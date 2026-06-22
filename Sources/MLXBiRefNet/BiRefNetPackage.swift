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
            provenance: Provenance(sourceRepo: "mlx-community/birefnet-general-mlx", revision: "main", tier: 2),
            requirements: RequirementsManifest(
                // Smoke-measured fp16 peak (5.6 MP input, hard_fur_dog): fast@1024 ≈ 4.9 GB · best@2048 ≈ 18.3 GB.
                // load() warms `fast` (the consumer path) → declare the fast envelope here. NOTE: best@2048
                // transiently peaks ~18 GB — a per-mode footprint isn't expressible in one package; tracked as
                // a contract gap (feed back) + a runtime memory guard before best on tight machines. Proper
                // controlled-envelope MemoryProbe still owed for the C-memory gate.
                footprints: [QuantFootprint(quant: .fp16, residentBytes: 5_300_000_000)],
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
    }
}

public extension BiRefNetPackage {
    /// Engine registration entry (license/eligibility gate at register; weights lazy on first run).
    nonisolated static var registration: PackageRegistration { .of(BiRefNetPackage.self) }
}
