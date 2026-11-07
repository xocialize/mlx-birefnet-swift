import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXNN
import MLXToolKit
import MLXProfiling
import BiRefNet

/// The conformant `matting` ModelPackage over the vendored BiRefNet core. One package, two quality modes
/// dispatched in `run(_:)` on `request.mode` (`fast` → general @1024, `best` → HR-matting @2048); each
/// returns a single-channel soft-alpha `Matte`. The engine constructs / loads / evicts it (C13) — it never
/// constructs itself.
@InferenceActor
public final class BiRefNetPackage: ModelPackage {
    public typealias Configuration = BiRefNetConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(weightLicense: .mit, portCodeLicense: .mit),   // BiRefNet MIT throughout
            // sourceRepo is the engine's download/marker key → point at the primary (fast) weights repo;
            // code provenance (vendored mnmly/mlx-swift-BiRefNet) lives in NOTICE/LICENSE.
            provenance: Provenance(sourceRepo: "mlx-community/BiRefNet-fp16", revision: "main", tier: 2),
            requirements: RequirementsManifest(
                // SPLIT FOOTPRINT (contract 1.14.0). RE-MEASURED 2026-11-07 (0.4.2, AB-A-0176, receipt in
                // Docs/receipts/footprint-20261107/). Each row is the **process phys_footprint** lifetime peak
                // (`/usr/bin/time -l`), one fresh process, run through the real MLXServeEngine
                // (`birefnet-smoke`, Release). Stack: mlx-swift 0.32.3, mlx-engine-swift 0.74.0, M5 Max 128 GB.
                // Inputs ran 1, 12, 24, 48 and 100 MP; the input size moves the peak by at most 0.25 GB.
                //   • fast@1024 : 7.32–7.57 GB (the engine's default pool cap is 2.15 GB here); 6.22 GB under a
                //     16 GB Mac's cap (0.64 GB) · MLX active peak 4.94 GB
                //   • best@2048 : 20.11–21.79 GB (2.15 GB cap); 19.91 GB under a 32 GB Mac's cap (1.19 GB)
                //     · MLX active peak 18.3 GB
                // The 2026-06-30 numbers (13.70 / 47.79 GB) were an UNCAPPED MLX buffer pool. With
                // `GPUCacheConfiguration(limit: .unmanaged)` the same smoke gives 14.58 / 47.04 GB today. The
                // cap is the engine's default since 0.21.0 (min(2 GB, 5% of budget)), so these declarations
                // assume an engine-managed pool. A host that runs `.unmanaged`, or writes MLX.Memory.cacheLimit
                // high itself, pays the old numbers.
                // Declared = the FAST (consumer) envelope as a split: residentBytes ~0.9 GB (both pipelines'
                // weights; best builds lazily and co-resides once requested); peakActivationBytes 7.2 GB. That
                // charges 8.1 GB, about 7% over the worst measured row, so fast admits on a 16 GB Mac (12.7 GB
                // budget). The engine reserves one shared transient across residents (serialized inference).
                //
                // best stays a RUNTIME-guarded variant (see `run()` / bestMinWorkingSet): mode is per-request,
                // so admission (at load, before the mode is known) can't reserve its peak. Making best a
                // first-class admitted variant = P1b (mode → PackageID); deferred (see EFFICIENCY-ADOPTION.md).
                footprints: [QuantFootprint(quant: .fp16,
                                            residentBytes: 900_000_000,         // both pipelines' weights floor
                                            peakActivationBytes: 7_200_000_000)], // fast: worst 7.57 GB phys, capped pool
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0))
            ),
            surfaces: [
                MattingContract.descriptor(
                    name: "birefnet",
                    summary: "BiRefNet foreground matting → single-channel soft-alpha matte. "
                        + "fast = general @1024 (~0.5s) · best = HR-matting @2048 (~2s).",
                    modes: [MattingContract.fast, MattingContract.best])
            ])
    }

    /// Device Metal working set best@2048 needs. RE-MEASURED 2026-11-07 (0.4.2, AB-A-0176): the process
    /// phys_footprint peak through the engine's managed pool is 21.79 GB at worst, across 1–100 MP inputs on a
    /// 2.15 GB cap, and 19.91 GB under a 32 GB Mac's 1.19 GB cap. The 2026-06-30 guard of 48 GB was the uncapped
    /// pool (47.04 GB today with `.unmanaged`). At 22 GB a 32 GB Mac clears it (working set ≈ 25 GB) and a 24 GB
    /// Mac (19.07 GB, AB-R-0551) refuses. Below this, `run(.best)` refuses rather than OOMs. Callers (e.g.
    /// EngineMatteProvider) catch `insufficientMemoryForBest` and fall back to fast.
    private static let bestMinWorkingSet: UInt64 = 22_000_000_000

    private let configuration: Configuration
    /// Built pipelines keyed by checkpoint role ("fast" / "best" / "lucida"). Keyed by ROLE rather
    /// than a fixed pair of fields, so a new checkpoint family costs no state here.
    private var pipelines: [String: BiRefNetPipeline] = [:]

    /// Test-facing seam for the engine's **INF gate** (C14): the loaded module graphs, keyed by
    /// tier. Reached via `@testable` — the conformance to `InferenceModeInspectable` lives in the
    /// test target so the shipping target takes no dependency on the conformance library.
    ///
    /// Every servable tier appears; tiers not yet built report `nil` (non-primary tiers build
    /// lazily), so a package that has not been `load()`-ed reports an empty graph — which INF-1
    /// treats as a failure, by design.
    var inferenceModeGraphs: [String: MLXNN.Module?] {
        var graphs: [String: MLXNN.Module?] = [:]
        for ckpt in configuration.servableCheckpoints {
            graphs[ckpt.role] = pipelines[ckpt.role]?.model
        }
        return graphs
    }

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    /// Warm the configuration's primary tier (`fast` for BiRefNet, the single tier for Lucida).
    /// Other tiers build lazily on first request, so a fast-only workflow never pays the 2048
    /// weight load.
    public func load() async throws {
        guard let primary = configuration.servableCheckpoints.first else { return }
        _ = try pipeline(for: primary)
    }

    public func unload() async {
        pipelines.removeAll()
        // Release the retained MLX buffer pool so eviction actually frees process RSS — dropping the refs
        // alone leaves the (large, @2048) activation buffers in MLX's cache, so phys_footprint doesn't drop
        // and R-MEM-1 can't reclaim. (Image-app acceptance run: process RSS grew monotonically across model
        // switches because unload() didn't clear the cache.)
        MLX.Memory.clearCache()
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before capability validation
        // (engine ≥ 0.27.0). Mid-run: the forward is ONE monolithic MLX eval (single lazy graph,
        // no iterative loop), so the real seams are pre-forward (after pipeline build/download)
        // and post-forward/pre-encode below — see CancellationTests for the cadence of record.
        try Task.checkCancellation()
        guard request.capability == .matting, let req = request as? MattingRequest else {
            throw BiRefNetError.unsupportedCapability(request.capability)
        }
        // Checkpoint identity + resolution come from the configuration's single choke point, so a
        // `mode` can only pick a tier WITHIN the registered family — never another family's weights
        // (a `.best` request on the Lucida PackageID resolves to Lucida's 1024 checkpoint, which is
        // what its descriptor advertises, not to BiRefNet's HR weights).
        let checkpoint = configuration.resolved(mode: req.mode)
        // Runtime memory guard: the 2048 tier peaks ~22 GB (managed pool), which the declared (1024) footprint
        // doesn't reserve — so refuse it on a device whose Metal working set can't hold it (rather
        // than OOM mid-forward). Keyed on the RESOLVED resolution, not on the requested mode, so it
        // tracks what will actually run. The caller (e.g. EngineMatteProvider) can catch this and
        // fall back to fast.
        if checkpoint.inputSize >= 2048 {
            let workingSet = MLX.GPU.deviceInfo().maxRecommendedWorkingSetSize
            if workingSet > 0 && workingSet < Self.bestMinWorkingSet {
                throw BiRefNetError.insufficientMemoryForBest(needsBytes: Self.bestMinWorkingSet,
                                                              deviceBytes: workingSet)
            }
        }
        let pipeline = try self.pipeline(for: checkpoint)
        // Pre-forward checkpoint: the lazy build above can be the first touch of a non-primary
        // tier's weights, so re-check before committing to the forward.
        try Task.checkCancellation()
        let input = try Self.decode(req.image)
        // Single-image matting forward is profiled (MLX_PROFILE=1). The pipeline self-evals its
        // outputs (predict + source-res resize), so one coarse region times the whole lazy graph
        // honestly; the core stays uninstrumented (single eval boundary, per the ddcolor precedent).
        // beginRun sits AFTER the lazy build so a first-request weight download can't skew the run.
        let prof = MLXProfiler.shared
        prof.beginRun("birefnet matting \(checkpoint.role) \(input.width)x\(input.height)")
        let prediction = prof.region("matting", "forward") {
            pipeline(input)                            // preprocess → forward → sigmoid → resize-to-source
        }
        prof.endRun(denominators: ["image": 1])
        let matteCG = try prediction.maskCGImage()     // 8-bit grayscale, source resolution
        try Task.checkCancellation()
        let png = try Self.encodePNG(matteCG)
        return MattingResponse(matte: Matte(format: .png, data: png,
                                            width: matteCG.width, height: matteCG.height,
                                            kind: .softAlpha))
    }

    // MARK: - Pipeline construction

    /// Cached-or-built pipeline for a resolved checkpoint.
    private func pipeline(
        for checkpoint: BiRefNetConfiguration.ResolvedCheckpoint
    ) throws -> BiRefNetPipeline {
        if let existing = pipelines[checkpoint.role] { return existing }
        let built = try buildPipeline(for: checkpoint)
        pipelines[checkpoint.role] = built
        return built
    }

    private func buildPipeline(
        for checkpoint: BiRefNetConfiguration.ResolvedCheckpoint
    ) throws -> BiRefNetPipeline {
        let url = try weightsURL(for: checkpoint)
        var cfg = BiRefNetConfig.swinLargeDefault
        cfg.inputSize = (width: checkpoint.inputSize, height: checkpoint.inputSize)
        return try BiRefNetPipeline.fromPretrained(url.path,
                                                   dtype: Self.dtype(configuration.quant),
                                                   config: cfg)
    }

    /// Resolve the weights file on disk. **No download happens here** — since contract 1.24 the
    /// ENGINE materializes every source `BiRefNetConfiguration.missingWeightSources(storeRoot:)`
    /// reports, before `load()` is ever called, with `WeightDownloadProgress` already bound so the
    /// host's prep UI shows a real `.downloading(fraction:)`. What remains is resolution plus an
    /// offline backstop: reaching the throw means no engine materialization ran (no store set, or a
    /// non-engine caller), which is a configuration error rather than something to paper over with a
    /// silent fetch.
    ///
    /// Accepts BOTH store layouts, matching the MS-2 probe: the hub-client snapshot
    /// (`snapshots/<commit>/…`) and the engine executor's flat layout.
    private func weightsURL(
        for checkpoint: BiRefNetConfiguration.ResolvedCheckpoint
    ) throws -> URL {
        if let override = checkpoint.override {
            guard FileManager.default.fileExists(atPath: override.path) else {
                throw BiRefNetError.weightsMissing(override)
            }
            return override
        }
        guard let root = configuration.modelsRootDirectory else { throw BiRefNetError.noModelStore }
        let store = ModelStore(root: root)
        let candidates = [store.snapshotDirectory(for: checkpoint.repo, revision: nil),
                          store.directory(for: checkpoint.repo)].compactMap { $0 }
        for dir in candidates {
            let url = dir.appendingPathComponent(configuration.weightsFile)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        throw BiRefNetError.weightsNotMaterialized(role: checkpoint.role, repo: checkpoint.repo)
    }

    private nonisolated static func dtype(_ q: Quant) -> DType {
        switch q {
        case .fp32: return .float32
        case .bf16: return .bfloat16
        default:    return .float16
        }
    }

    // MARK: - Image ⇄ Matte (canonical artifacts carry encoded bytes)

    private nonisolated static func decode(_ image: Image) throws -> CGImage {
        guard let src = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw BiRefNetError.decodeFailed
        }
        return cg
    }

    private nonisolated static func encodePNG(_ cg: CGImage) throws -> Data {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else {
            throw BiRefNetError.encodeFailed
        }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { throw BiRefNetError.encodeFailed }
        return data as Data
    }

    public enum BiRefNetError: Error {
        case unsupportedCapability(Capability)
        case noModelStore
        case weightsMissing(URL)
        /// A declared weight source is absent from the store — i.e. the engine's pre-`load()`
        /// materialization never ran. The offline backstop, not a fetch trigger.
        case weightsNotMaterialized(role: String, repo: String)
        case decodeFailed
        case encodeFailed
        /// `best` requested but the device's Metal working set can't hold the ~18 GB peak. Catch + retry `fast`.
        case insufficientMemoryForBest(needsBytes: UInt64, deviceBytes: UInt64)
    }
}

public extension BiRefNetPackage {
    /// Engine registration entry (license/eligibility gate at register; weights lazy on first run).
    nonisolated static var registration: PackageRegistration { .of(BiRefNetPackage.self) }
}
