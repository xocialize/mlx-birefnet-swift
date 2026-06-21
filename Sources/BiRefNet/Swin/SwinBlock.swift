import Foundation
import MLX
import MLXNN

/// Two-layer MLP with GELU. Used inside `SwinTransformerBlock`. Matches the
/// PyTorch `Mlp` class. Note `Dropout` is dropped at inference (always 0.0
/// in the released checkpoint, and we don't run training).
final class SwinMlp: Module, UnaryLayer {

    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(inFeatures: Int, hiddenFeatures: Int) {
        _fc1.wrappedValue = Linear(inFeatures, hiddenFeatures, bias: true)
        _fc2.wrappedValue = Linear(hiddenFeatures, inFeatures, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return fc2(gelu(fc1(x)))
    }
}

/// One Swin Transformer block (W-MSA or SW-MSA).
final class SwinTransformerBlock: Module {

    let dim: Int
    let numHeads: Int
    let windowSize: Int
    let shiftSize: Int

    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: WindowAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: SwinMlp

    init(dim: Int, numHeads: Int, windowSize: Int, shiftSize: Int, mlpRatio: Float = 4.0,
         qkvBias: Bool = true) {
        self.dim = dim
        self.numHeads = numHeads
        self.windowSize = windowSize
        self.shiftSize = shiftSize
        precondition(shiftSize >= 0 && shiftSize < windowSize,
                     "shiftSize must be in [0, windowSize)")

        _norm1.wrappedValue = LayerNorm(dimensions: dim)
        _attn.wrappedValue = WindowAttention(
            dim: dim,
            windowSize: (windowSize, windowSize),
            numHeads: numHeads,
            qkvBias: qkvBias
        )
        _norm2.wrappedValue = LayerNorm(dimensions: dim)
        let mlpHidden = Int(Float(dim) * mlpRatio)
        _mlp.wrappedValue = SwinMlp(inFeatures: dim, hiddenFeatures: mlpHidden)
        super.init()
    }

    /// `x`: `(B, H*W, C)`. `maskMatrix`: `(numWindows, N, N)` for SW-MSA, or `nil`.
    func callAsFunction(_ x: MLXArray, H: Int, W: Int, maskMatrix: MLXArray?) -> MLXArray {
        let B = x.dim(0)
        let C = x.dim(2)
        precondition(x.dim(1) == H * W, "input feature has wrong size")

        let shortcut = x
        var v = norm1(x).reshaped([B, H, W, C])

        // Pad to a multiple of windowSize on the right/bottom (matches Python).
        let padR = (windowSize - W % windowSize) % windowSize
        let padB = (windowSize - H % windowSize) % windowSize
        if padR != 0 || padB != 0 {
            let widths: [IntOrPair] = [
                .init(0),
                .init((0, padB)),
                .init((0, padR)),
                .init(0),
            ]
            v = padded(v, widths: widths, mode: .constant)
        }
        let Hp = v.dim(1)
        let Wp = v.dim(2)

        // Cyclic shift for SW-MSA.
        let shiftedX: MLXArray
        let attnMask: MLXArray?
        if shiftSize > 0 {
            shiftedX = roll(roll(v, shift: -shiftSize, axis: 1), shift: -shiftSize, axis: 2)
            attnMask = maskMatrix
        } else {
            shiftedX = v
            attnMask = nil
        }

        // Partition windows: (nW*B, ws, ws, C) -> (nW*B, ws*ws, C)
        var windows = windowPartition(shiftedX, windowSize: windowSize)
        windows = windows.reshaped([-1, windowSize * windowSize, C])

        // W-MSA / SW-MSA
        let attnOut = attn(windows, mask: attnMask)

        // Merge windows back: (nW*B, ws, ws, C) -> (B, Hp, Wp, C)
        let attnWindows = attnOut.reshaped([-1, windowSize, windowSize, C])
        var merged = windowReverse(attnWindows, windowSize: windowSize, H: Hp, W: Wp)

        // Reverse cyclic shift.
        if shiftSize > 0 {
            merged = roll(roll(merged, shift: shiftSize, axis: 1), shift: shiftSize, axis: 2)
        }

        // Crop padding.
        if padR != 0 || padB != 0 {
            merged = merged[0..., ..<H, ..<W, 0...]
        }

        var y = merged.reshaped([B, H * W, C])

        // FFN. Drop-path is identity at inference.
        y = shortcut + y
        y = y + mlp(norm2(y))
        return y
    }
}
