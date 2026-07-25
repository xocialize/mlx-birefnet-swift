import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXNN
import MLXToolKit
import MLXProfiling
import Hub
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
                // SPLIT FOOTPRINT (contract 1.14.0) — RE-BASELINED 2026-06-30 to the in-app **process
                // phys_footprint** (the authoritative admission basis: it's what R-MEM-1 compares against and
                // what OOMs the process), measured in MLXEngineImage via MLXEngineTestKit with isolate +
                // clearCache. The original birefnet-smoke figures (fast peak 4.9 GB / best 18.3 GB) were MLX
                // working-set, which UNDER-read the real process peak by ~2.7× (the MLX buffer cache + process
                // overhead aren't in the MLX-peak metric). Clean phys_footprint (floor after clearCache → true
                // weights; peak = run high-water):
                //   • fast@1024 : floor 0.54 GB · peak 13.70 GB → transient ~13.2 GB
                //   • best@2048 : floor 0.61 GB · peak 47.79 GB → transient ~47.2 GB
                // Declared = the FAST (consumer) envelope as a split: residentBytes ~0.9 GB (both pipelines'
                // weights, best builds lazily + co-resides once requested); peakActivationBytes = the measured
                // fast transient. The engine reserves one shared transient across residents (serialized
                // inference), so this charges ~0.9 GB persistent + ~14 GB transient.
                //
                // best's measured split (NOT the admitted default; both modes are fp16 so QuantFootprint can't
                // carry it): residentBytes ~0.6 GB · peakActivationBytes ~47.2 GB. best stays a RUNTIME-guarded
                // variant (see `run()` / bestMinWorkingSet) — mode is per-request, so admission (at load,
                // before the mode is known) can't reserve its ~48 GB peak. The guard was raised 19.5 → 48 GB
                // on this re-measure (best does NOT fit 32 GB, is tight on 64 GB). Making best a first-class
                // admitted variant = P1b (mode → PackageID); deferred (see EFFICIENCY-ADOPTION.md outcome).
                footprints: [QuantFootprint(quant: .fp16,
                                            residentBytes: 900_000_000,         // both pipelines' weights floor
                                            peakActivationBytes: 14_000_000_000)], // measured fast transient (~13.2 GB + margin)
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

    /// Device Metal working set best@2048 needs. RE-BASELINED 2026-06-30 to the in-app phys_footprint peak
    /// (47.79 GB — the old 19.5 GB was from the MLX-peak smoke and ~2.4× too low, so best falsely admitted
    /// on 32 GB Macs). Below this, `run(.best)` refuses rather than OOMs: a 32 GB Mac fails, a 64 GB Mac
    /// clears it (tight). Callers (e.g. EngineMatteProvider) catch `insufficientMemoryForBest` → fall back to fast.
    private static let bestMinWorkingSet: UInt64 = 48_000_000_000

    private let configuration: Configuration
    private var fast: BiRefNetPipeline?
    private var best: BiRefNetPipeline?

    /// Test-facing seam for the engine's **INF gate** (C14): the loaded module graphs, keyed by
    /// tier. Reached via `@testable` — the conformance to `InferenceModeInspectable` lives in the
    /// test target so the shipping target takes no dependency on the conformance library.
    ///
    /// Both tiers are `nil` until requested (`best` builds lazily), so a package that has not been
    /// `load()`-ed reports an empty graph — which INF-1 treats as a failure, by design.
    var inferenceModeGraphs: [String: MLXNN.Module?] {
        ["fast": fast?.model, "best": best?.model]
    }

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    /// Warm the default (`fast`) tier; `best` builds lazily on its first request so a fast-only workflow
    /// never pays the 2048 weight load (or download).
    public func load() async throws {
        if fast == nil { fast = try await buildPipeline(best: false) }
    }

    public func unload() async {
        fast = nil; best = nil
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
        let useBest = req.mode == MattingContract.best
        // Runtime memory guard: best@2048 peaks ~18 GB, which the single declared (fast) footprint doesn't
        // reserve — so refuse best on a device whose Metal working set can't hold it (rather than OOM mid-
        // forward). The caller (e.g. EngineMatteProvider) can catch this and fall back to fast.
        if useBest {
            let workingSet = MLX.GPU.deviceInfo().maxRecommendedWorkingSetSize
            if workingSet > 0 && workingSet < Self.bestMinWorkingSet {
                throw BiRefNetError.insufficientMemoryForBest(needsBytes: Self.bestMinWorkingSet,
                                                              deviceBytes: workingSet)
            }
        }
        let pipeline = try await pipeline(best: useBest)
        // Pre-forward checkpoint: the pipeline build above can include a first-request weight
        // download (lazy best tier), so re-check before committing to the forward.
        try Task.checkCancellation()
        let input = try Self.decode(req.image)
        // Single-image matting forward is profiled (MLX_PROFILE=1). The pipeline self-evals its
        // outputs (predict + source-res resize), so one coarse region times the whole lazy graph
        // honestly; the core stays uninstrumented (single eval boundary, per the ddcolor precedent).
        // beginRun sits AFTER the lazy build so a first-request weight download can't skew the run.
        let prof = MLXProfiler.shared
        prof.beginRun("birefnet matting \(useBest ? "best" : "fast") \(input.width)x\(input.height)")
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

    private func pipeline(best useBest: Bool) async throws -> BiRefNetPipeline {
        if useBest {
            if best == nil { best = try await buildPipeline(best: true) }
            return best!
        }
        if fast == nil { fast = try await buildPipeline(best: false) }
        return fast!
    }

    private func buildPipeline(best useBest: Bool) async throws -> BiRefNetPipeline {
        let size = useBest ? 2048 : 1024
        let url = try await weightsURL(best: useBest)
        var cfg = BiRefNetConfig.swinLargeDefault
        cfg.inputSize = (width: size, height: size)
        return try BiRefNetPipeline.fromPretrained(url.path, dtype: Self.dtype(configuration.quant), config: cfg)
    }

    /// Resolve the weights file, **downloading from HF on first use** via the convention's native
    /// downloader (swift-transformers `HubApi`, pointed at the engine-stamped model-store root; idempotent —
    /// skips files already present). Download progress is forwarded to the engine's `WeightDownloadProgress`
    /// sink so the host's prep UI shows a real `.downloading(fraction:)`. A direct override URL (pre-resolved
    /// caller / CLI smoke) bypasses the network entirely.
    private func weightsURL(best useBest: Bool) async throws -> URL {
        if let override = useBest ? configuration.bestWeightsURL : configuration.fastWeightsURL {
            guard FileManager.default.fileExists(atPath: override.path) else {
                throw BiRefNetError.weightsMissing(override)
            }
            return override
        }
        let repo = useBest ? configuration.bestRepo : configuration.fastRepo
        let hub = HubApi(downloadBase: configuration.modelsRootDirectory)
        let dir = try await hub.snapshot(from: repo, matching: ["*.safetensors"]) { @Sendable progress in
            WeightDownloadProgress.report(fraction: progress.fractionCompleted)
        }
        let url = dir.appendingPathComponent(configuration.weightsFile)
        guard FileManager.default.fileExists(atPath: url.path) else { throw BiRefNetError.weightsMissing(url) }
        return url
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
