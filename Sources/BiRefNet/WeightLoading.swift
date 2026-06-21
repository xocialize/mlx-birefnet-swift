import Foundation
import MLX
import MLXNN

extension BiRefNet {

    /// Load a converted `.safetensors` produced by `birefnet-convert` into
    /// the model. All Conv2d weights must already be NHWC.
    ///
    /// - Parameters:
    ///   - url:    file URL pointing at the converted `model.safetensors`.
    ///   - dtype:  target dtype to cast every loaded array to (default `.float32`).
    ///   - strict: when `true`, the call fails if the checkpoint contains keys
    ///             with no matching parameter in the module tree.
    public func loadWeights(
        url: URL,
        dtype: DType = .float32,
        strict: Bool = true
    ) throws {
        let weights = try MLX.loadArrays(url: url)

        var remapped: [(String, MLXArray)] = []
        remapped.reserveCapacity(weights.count)
        for (key, value) in weights {
            remapped.append((key, value.asType(dtype)))
        }

        let params = ModuleParameters.unflattened(remapped)
        try update(parameters: params, verify: strict ? [.noUnusedKeys] : [])
    }
}
