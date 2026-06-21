import Foundation
import MLX

/// `(B, H, W, C) -> (numWindows*B, windowSize, windowSize, C)` — the exact
/// Python `window_partition`. Requires `H % windowSize == 0` and
/// `W % windowSize == 0`.
@inlinable
public func windowPartition(_ x: MLXArray, windowSize ws: Int) -> MLXArray {
    let B = x.dim(0)
    let H = x.dim(1)
    let W = x.dim(2)
    let C = x.dim(3)
    var v = x.reshaped([B, H / ws, ws, W / ws, ws, C])
    v = v.transposed(0, 1, 3, 2, 4, 5)
    v = v.reshaped([-1, ws, ws, C])
    return v
}

/// Inverse of ``windowPartition(_:windowSize:)``. `(numWindows*B, ws, ws, C) -> (B, H, W, C)`.
@inlinable
public func windowReverse(_ windows: MLXArray, windowSize ws: Int, H: Int, W: Int) -> MLXArray {
    let C = windows.dim(-1)
    var x = windows.reshaped([-1, H / ws, W / ws, ws, ws, C])
    x = x.transposed(0, 1, 3, 2, 4, 5)
    x = x.reshaped([-1, H, W, C])
    return x
}
