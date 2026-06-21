import Foundation
import MLX
import MLXNN

/// Image → patch token embedding. Mirrors `PatchEmbed` from `swin_v1.py`.
///
/// In the Python reference (NCHW), the conv outputs `(B, embedDim, Wh, Ww)`,
/// the spatial map is flattened/transposed for the optional LayerNorm, and
/// the result is reshaped back to `(B, embedDim, Wh, Ww)`.
///
/// Here in NHWC we just leave the tensor as `(B, Wh, Ww, embedDim)` the whole
/// time — LayerNorm normalises over the last axis, which is the channel axis
/// in NHWC, so the math is identical without any reshape.
final class PatchEmbed: Module {

    let patchSize: Int
    let embedDim: Int
    let inChannels: Int

    @ModuleInfo(key: "proj") var proj: Conv2d
    @ModuleInfo(key: "norm") var norm: LayerNorm?

    init(patchSize: Int = 4, inChannels: Int = 3, embedDim: Int = 96, patchNorm: Bool = true) {
        self.patchSize = patchSize
        self.embedDim = embedDim
        self.inChannels = inChannels

        _proj.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: embedDim,
            kernelSize: .init(patchSize),
            stride: .init(patchSize),
            bias: true
        )
        _norm.wrappedValue = patchNorm ? LayerNorm(dimensions: embedDim) : nil
        super.init()
    }

    /// `x` is NHWC: `(B, H, W, inChannels)`. Returns `(B, Wh, Ww, embedDim)`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        // Pad the bottom / right edges to a multiple of patch size (matches the
        // Python which uses `F.pad`).
        let H = x.dim(1)
        let W = x.dim(2)
        let padH = (patchSize - H % patchSize) % patchSize
        let padW = (patchSize - W % patchSize) % patchSize
        if padH != 0 || padW != 0 {
            let widths: [IntOrPair] = [
                .init(0),
                .init((0, padH)),
                .init((0, padW)),
                .init(0),
            ]
            x = padded(x, widths: widths, mode: .constant)
        }
        x = proj(x)
        if let norm {
            x = norm(x)
        }
        return x
    }
}
