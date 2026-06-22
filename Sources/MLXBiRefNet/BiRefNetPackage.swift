import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXToolKit
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
                // Memory-harness measured (M-Max, 5.6 MP input; weights resident only ~424 MB fp16 — the
                // footprint is ~all activation high-water): fast@1024 peak 4.9 GB → recommend 6.2 GB ·
                // best@2048 peak 18.3 GB → recommend 22.2 GB. One fp16 footprint can't express both, and the
                // governor charges the largest-fitting footprint → declaring best (22 GB) would make even the
                // 6 GB fast tier inadmissible on <32 GB Macs. So declare the **fast (consumer) envelope** and
                // gate best with a RUNTIME memory guard (see `run()`); the proper fix is a config/per-mode
                // footprint (open engine enhancement — flagged to feed back). See MEMORY-REPORT.md.
                footprints: [QuantFootprint(quant: .fp16, residentBytes: 6_500_000_000)],
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

    /// Device Metal working set best@2048 needs (measured peak 18.3 GB + margin). Below this, `run(.best)`
    /// refuses rather than OOMs; ~32 GB+ Macs clear it, ≤16 GB don't.
    private static let bestMinWorkingSet: UInt64 = 19_500_000_000

    private let configuration: Configuration
    private var fast: BiRefNetPipeline?
    private var best: BiRefNetPipeline?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    /// Warm the default (`fast`) tier; `best` builds lazily on its first request so a fast-only workflow
    /// never pays the 2048 weight load (or download).
    public func load() async throws {
        if fast == nil { fast = try await buildPipeline(best: false) }
    }

    public func unload() async { fast = nil; best = nil }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard request.capability == .matting, let req = request as? MattingRequest else {
            throw BiRefNetError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()
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
        let input = try Self.decode(req.image)
        let prediction = pipeline(input)               // preprocess → forward → sigmoid → resize-to-source
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
