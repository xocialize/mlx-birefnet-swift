import Foundation
import MLX
import MLXNN

/// Bilinear resize an NHWC tensor to an arbitrary `(targetH, targetW)`. Uses
/// `Upsample` with a per-axis scale factor (matches Python
/// `F.interpolate(..., mode='bilinear', align_corners=True)`).
///
/// Note: PyTorch BiRefNet uses `align_corners=True`. MLX's `Upsample` with
/// `.linear(alignCorners: true)` matches that geometry.
@inlinable
public func bilinearResize(_ x: MLXArray, _ targetH: Int, _ targetW: Int) -> MLXArray {
    let H = x.dim(1)
    let W = x.dim(2)
    if H == targetH && W == targetW { return x }
    let sh = Float(targetH) / Float(H)
    let sw = Float(targetW) / Float(W)
    let up = Upsample(scaleFactor: .array([sh, sw]),
                      mode: .linear(alignCorners: true))
    return up(x)
}

/// `b c (hg h) (wg w) -> b (c hg wg) h w`, written for NHWC tensors:
/// `(B, hg*h, wg*w, C) -> (B, h, w, C * hg * wg)`. Mirrors
/// `image2patches(..., transformation='b c (hg h) (wg w) -> b (c hg wg) h w')`
/// from the Python BiRefNet, which is the variant used in
/// `Decoder.forward` (NOT the one that explodes the batch axis).
///
/// `gridH`/`gridW` is the number of tiles per axis. The input H/W must be
/// divisible by them.
public func image2patches(_ x: MLXArray, gridH: Int, gridW: Int) -> MLXArray {
    let B = x.dim(0)
    let H = x.dim(1)
    let W = x.dim(2)
    let C = x.dim(3)
    precondition(H % gridH == 0 && W % gridW == 0,
                 "image2patches: H,W (\(H),\(W)) not divisible by grid (\(gridH),\(gridW))")
    let h = H / gridH
    let w = W / gridW
    // (B, hg*h, wg*w, C) -> (B, hg, h, wg, w, C)
    var out = x.reshaped([B, gridH, h, gridW, w, C])
    // -> (B, h, w, C, hg, wg)
    out = out.transposed(0, 2, 4, 5, 1, 3)
    // flatten last 3 dims -> (B, h, w, C*hg*wg)
    out = out.reshaped([B, h, w, C * gridH * gridW])
    return out
}
