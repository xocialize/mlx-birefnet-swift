import Foundation
import MLXToolKit

/// Configuration for the BiRefNet `matting` package. Two converted weight sets back the two quality
/// modes — `MattingContract.fast` = general @1024 (~0.5 s) · `MattingContract.best` = HR-matting @2048
/// (~2 s) — both loading through the identical Swin-L + ASPP-Deformable arch (zero code difference).
/// Weights resolve under the engine's model store (`modelsRootDirectory` + repo dir + `weightsFile`).
public struct BiRefNetConfiguration: PackageConfiguration, ModelStorable {
    /// HF repo id for the converted MLX `.safetensors` backing the `fast` tier (general @1024).
    public var fastRepo: String
    /// HF repo id for the converted MLX `.safetensors` backing the `best` tier (HR-matting @2048).
    public var bestRepo: String
    /// Weights filename within each repo's local snapshot directory.
    public var weightsFile: String
    /// Inference dtype — also the declared footprint quant. fp16 is the validated runtime.
    public var quant: Quant
    /// Engine-injected model-store root (download dir + security-scoped bookmark).
    public var modelsRootDirectory: URL?
    /// Direct path to the `fast`/`best` weights, bypassing model-store resolution. For pre-resolved
    /// callers + dev/CLI smoke; nil = resolve under `modelsRootDirectory`/`fastRepo`/`bestRepo`.
    public var fastWeightsURL: URL?
    public var bestWeightsURL: URL?

    public init(fastRepo: String = "mlx-community/BiRefNet-fp16",
                bestRepo: String = "mlx-community/BiRefNet_HR-matting-fp16",
                weightsFile: String = "model.safetensors",
                quant: Quant = .fp16,
                modelsRootDirectory: URL? = nil,
                fastWeightsURL: URL? = nil,
                bestWeightsURL: URL? = nil) {
        self.fastRepo = fastRepo
        self.bestRepo = bestRepo
        self.weightsFile = weightsFile
        self.quant = quant
        self.modelsRootDirectory = modelsRootDirectory
        self.fastWeightsURL = fastWeightsURL
        self.bestWeightsURL = bestWeightsURL
    }
}

/// `QuantConfigured` (engine contract 1.14.0): the config already stores `var quant`, so an empty
/// conformance lets `MemoryGovernor` charge the matching declared `QuantFootprint` for the selected
/// quant instead of the largest-that-fits heuristic. (Both modes are fp16, so the quant match resolves
/// to the single declared fp16 footprint — the split below; per-mode activation is handled by the
/// `run()` best guard, see P1b note in EFFICIENCY-ADOPTION.md.)
extension BiRefNetConfiguration: QuantConfigured {}
