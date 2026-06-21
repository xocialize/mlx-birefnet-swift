import Foundation
import MLX
import MLXNN

/// 2×2 spatial downsampling layer. Mirrors `PatchMerging` from `swin_v1.py`:
///
/// ```python
/// x0 = x[:, 0::2, 0::2, :]; x1 = x[:, 1::2, 0::2, :]
/// x2 = x[:, 0::2, 1::2, :]; x3 = x[:, 1::2, 1::2, :]
/// x = torch.cat([x0, x1, x2, x3], -1)   # 4 * C
/// x = self.norm(x); x = self.reduction(x)  # 2 * C
/// ```
final class PatchMerging: Module {

    let dim: Int

    @ModuleInfo(key: "reduction") var reduction: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(dim: Int) {
        self.dim = dim
        _reduction.wrappedValue = Linear(4 * dim, 2 * dim, bias: false)
        _norm.wrappedValue = LayerNorm(dimensions: 4 * dim)
        super.init()
    }

    /// Input `(B, H*W, C)` (token sequence). Output `(B, (H*W)/4 or padded equivalent, 2*C)`.
    func callAsFunction(_ x: MLXArray, H: Int, W: Int) -> MLXArray {
        let B = x.dim(0)
        let C = x.dim(2)
        precondition(x.dim(1) == H * W, "PatchMerging: input length mismatch")

        var v = x.reshaped([B, H, W, C])
        // Pad odd H/W (matches Python: `F.pad(x, (0, 0, 0, W % 2, 0, H % 2))`).
        let padH = H % 2
        let padW = W % 2
        if padH != 0 || padW != 0 {
            let widths: [IntOrPair] = [
                .init(0),
                .init((0, padH)),
                .init((0, padW)),
                .init(0),
            ]
            v = padded(v, widths: widths, mode: .constant)
        }
        let x0 = v[0..., .stride(by: 2), .stride(by: 2), 0...]
        let x1 = v[0..., .stride(from: 1, by: 2), .stride(by: 2), 0...]
        let x2 = v[0..., .stride(by: 2), .stride(from: 1, by: 2), 0...]
        let x3 = v[0..., .stride(from: 1, by: 2), .stride(from: 1, by: 2), 0...]
        var merged = concatenated([x0, x1, x2, x3], axis: -1)  // (B, H/2, W/2, 4C)
        let mH = merged.dim(1)
        let mW = merged.dim(2)
        merged = merged.reshaped([B, mH * mW, 4 * C])
        merged = norm(merged)
        merged = reduction(merged)
        return merged
    }
}
