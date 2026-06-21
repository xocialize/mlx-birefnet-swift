import Foundation
import MLX
import MLXNN

/// One Swin "stage". `depth` transformer blocks + an optional `PatchMerging`
/// downsample at the end.
final class BasicLayer: Module {

    let dim: Int
    let depth: Int
    let windowSize: Int
    let shiftSize: Int
    let hasDownsample: Bool

    @ModuleInfo(key: "blocks") var blocks: [SwinTransformerBlock]
    @ModuleInfo(key: "downsample") var downsample: PatchMerging?

    init(dim: Int, depth: Int, numHeads: Int, windowSize: Int, mlpRatio: Float = 4.0,
         qkvBias: Bool = true, hasDownsample: Bool) {
        self.dim = dim
        self.depth = depth
        self.windowSize = windowSize
        self.shiftSize = windowSize / 2
        self.hasDownsample = hasDownsample

        var built: [SwinTransformerBlock] = []
        for i in 0..<depth {
            built.append(
                SwinTransformerBlock(
                    dim: dim,
                    numHeads: numHeads,
                    windowSize: windowSize,
                    shiftSize: (i % 2 == 0) ? 0 : windowSize / 2,
                    mlpRatio: mlpRatio,
                    qkvBias: qkvBias
                )
            )
        }
        _blocks.wrappedValue = built
        _downsample.wrappedValue = hasDownsample ? PatchMerging(dim: dim) : nil
        super.init()
    }

    /// Returns `(out, H, W, down, downH, downW)` mirroring the Python:
    /// `out` is the pre-downsample feature, `down` is the post-downsample
    /// feature (or `out` if there's no downsample), and the H/W pairs are
    /// the spatial sizes of each.
    struct Output {
        let out: MLXArray
        let H: Int
        let W: Int
        let down: MLXArray
        let downH: Int
        let downW: Int
    }

    func callAsFunction(_ x: MLXArray, H: Int, W: Int) -> Output {
        // Build the SW-MSA attention mask (Python: img_mask + window_partition).
        // The mask shape depends on the padded resolution (Hp, Wp).
        let Hp = ((H + windowSize - 1) / windowSize) * windowSize
        let Wp = ((W + windowSize - 1) / windowSize) * windowSize
        let attnMask = makeShiftedWindowMask(Hp: Hp, Wp: Wp)

        var v = x
        for blk in blocks {
            let mask: MLXArray? = (blk.shiftSize > 0) ? attnMask : nil
            v = blk(v, H: H, W: W, maskMatrix: mask)
        }

        if let downsample {
            let downed = downsample(v, H: H, W: W)
            let dH = (H + 1) / 2
            let dW = (W + 1) / 2
            return Output(out: v, H: H, W: W, down: downed, downH: dH, downW: dW)
        }
        return Output(out: v, H: H, W: W, down: v, downH: H, downW: W)
    }

    /// Builds the cyclic-shift attention mask exactly as in Python.
    /// Returns `(numWindows, ws*ws, ws*ws)` with `0` for same-region pairs
    /// and `-inf` for cross-region pairs.
    private func makeShiftedWindowMask(Hp: Int, Wp: Int) -> MLXArray {
        let ws = windowSize
        let ss = shiftSize
        var flat = [Int32](repeating: 0, count: Hp * Wp)
        var counter: Int32 = 0

        let hSlices: [(Int, Int)] = [
            (0, Hp - ws),
            (Hp - ws, Hp - ss),
            (Hp - ss, Hp),
        ]
        let wSlices: [(Int, Int)] = [
            (0, Wp - ws),
            (Wp - ws, Wp - ss),
            (Wp - ss, Wp),
        ]
        for (h0, h1) in hSlices {
            for (w0, w1) in wSlices {
                for y in h0..<h1 {
                    for x in w0..<w1 {
                        flat[y * Wp + x] = counter
                    }
                }
                counter += 1
            }
        }
        let imgMask = MLXArray(flat, [1, Hp, Wp, 1])
        var maskWindows = windowPartition(imgMask, windowSize: ws)  // (nW, ws, ws, 1)
        maskWindows = maskWindows.reshaped([-1, ws * ws])
        let diff = maskWindows.expandedDimensions(axis: 1) - maskWindows.expandedDimensions(axis: 2)
        // diff: (nW, N, N), 0 where same region, nonzero otherwise.
        let neg = MLXArray(-1e9 as Float)
        let zero = MLXArray(Float(0))
        let mask = MLX.which(diff .!= 0, neg, zero)
        return mask
    }
}
