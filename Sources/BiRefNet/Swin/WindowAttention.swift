import Foundation
import MLX
import MLXNN
import MLXFast

/// Window-based multi-head self-attention with relative position bias. Mirrors
/// `WindowAttention` from `swin_v1.py` (SDPA path, since the trained
/// checkpoint was produced with `config.SDPA_enabled = True`).
final class WindowAttention: Module {

    let dim: Int
    let numHeads: Int
    let headDim: Int
    let windowSize: (Int, Int)
    let scale: Float

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    /// `[(2*Wh - 1) * (2*Ww - 1), numHeads]`. Learnable.
    @ParameterInfo(key: "relative_position_bias_table") var relativePositionBiasTable: MLXArray

    /// `[Wh*Ww, Wh*Ww]` — integer indices. Computed at init time from the
    /// window size; the equivalent PyTorch buffer is dropped at conversion
    /// time so the checkpoint never carries it.
    let relativePositionIndex: MLXArray

    init(dim: Int, windowSize: (Int, Int), numHeads: Int, qkvBias: Bool = true) {
        self.dim = dim
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.windowSize = windowSize
        self.scale = 1.0 / Float(self.headDim).squareRoot()

        _qkv.wrappedValue = Linear(dim, dim * 3, bias: qkvBias)
        _proj.wrappedValue = Linear(dim, dim, bias: true)

        let nBias = (2 * windowSize.0 - 1) * (2 * windowSize.1 - 1)
        _relativePositionBiasTable.wrappedValue = MLXArray.zeros([nBias, numHeads])

        self.relativePositionIndex = WindowAttention.makeRelativePositionIndex(
            windowSize: windowSize)

        super.init()
    }

    private static func makeRelativePositionIndex(windowSize ws: (Int, Int)) -> MLXArray {
        // Mirrors the PyTorch buffer-build code exactly.
        let Wh = ws.0
        let Ww = ws.1
        var flat = [Int32](repeating: 0, count: Wh * Ww * Wh * Ww)
        let strideY = 2 * Ww - 1
        for y1 in 0..<Wh {
            for x1 in 0..<Ww {
                for y2 in 0..<Wh {
                    for x2 in 0..<Ww {
                        let dy = y1 - y2 + (Wh - 1)
                        let dx = x1 - x2 + (Ww - 1)
                        let idx = (y1 * Ww + x1) * (Wh * Ww) + (y2 * Ww + x2)
                        flat[idx] = Int32(dy * strideY + dx)
                    }
                }
            }
        }
        return MLXArray(flat, [Wh * Ww, Wh * Ww])
    }

    /// `x`: `(B_, N, C)` where `N = windowSize.0 * windowSize.1`.
    /// `mask`: `(numWindows, N, N)` for shifted-window attention, or `nil`.
    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let Bp = x.dim(0)
        let N = x.dim(1)
        let C = x.dim(2)

        // qkv: (B_, N, 3C) → (B_, N, 3, H, Dh) → (3, B_, H, N, Dh)
        var qkvOut = qkv(x).reshaped([Bp, N, 3, numHeads, headDim])
        qkvOut = qkvOut.transposed(2, 0, 3, 1, 4)
        let q = qkvOut[0]
        let k = qkvOut[1]
        let v = qkvOut[2]

        // Relative position bias: lookup -> (N, N, numHeads) -> (numHeads, N, N) -> (1, H, N, N)
        let indices = relativePositionIndex.reshaped([-1])
        var bias = relativePositionBiasTable[indices]                       // (N*N, H)
        bias = bias.reshaped([N, N, numHeads])                              // (N, N, H)
        bias = bias.transposed(2, 0, 1).expandedDimensions(axis: 0)         // (1, H, N, N)
        bias = bias.asType(q.dtype)

        var attnMask: MLXArray = bias
        if let mask {
            // mask is (nW, N, N); broadcast to (B_, H, N, N) by:
            //   expand on heads dim: (nW, 1, N, N), then add to bias.
            // But B_ = nW * B; the Python "SDPA_enabled" branch expands
            //   mask to (B_, 1, N, N) by repeating along the batch dim.
            // We mirror that here.
            let nW = mask.dim(0)
            let B = Bp / nW
            var m = mask.expandedDimensions(axis: 0)          // (1, nW, N, N)
            m = broadcast(m, to: [B, nW, N, N])               // (B, nW, N, N)
            m = m.reshaped([B * nW, 1, N, N]).asType(q.dtype) // (B_, 1, N, N)
            attnMask = bias + m
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: scale, mask: attnMask
        )

        let merged = out.transposed(0, 2, 1, 3).reshaped([Bp, N, C])
        return proj(merged)
    }
}
