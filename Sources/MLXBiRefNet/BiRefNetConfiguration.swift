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

    public init(fastRepo: String = "xocialize/birefnet-general-mlx",
                bestRepo: String = "xocialize/birefnet-hr-matting-mlx",
                weightsFile: String = "model.safetensors",
                quant: Quant = .fp16,
                modelsRootDirectory: URL? = nil) {
        self.fastRepo = fastRepo
        self.bestRepo = bestRepo
        self.weightsFile = weightsFile
        self.quant = quant
        self.modelsRootDirectory = modelsRootDirectory
    }
}
