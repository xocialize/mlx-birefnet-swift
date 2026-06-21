import Foundation
import MLX
import MLXNN

/// BiRefNet inference-side top-level model.
///
/// Encoder = Swin backbone (called twice for `mul_scl_ipt='cat'`).
/// Bottleneck = a single `BasicDecBlk` (`squeeze_module`).
/// Decoder = `BiRefNetDecoder`.
///
/// Forward returns single-channel logits at the input resolution. Apply
/// `sigmoid` to get a probability mask.
public final class BiRefNet: Module {

    public let config: BiRefNetConfig

    @ModuleInfo(key: "bb") public var bb: SwinTransformer
    @ModuleInfo(key: "squeeze_module") public var squeezeModule: [BasicDecBlk]
    @ModuleInfo(key: "decoder") public var decoder: BiRefNetDecoder

    public init(config: BiRefNetConfig = .swinLargeDefault) {
        self.config = config

        _bb.wrappedValue = SwinTransformer(config.backbone)

        // squeeze_module: `BasicDecBlk_x1` ⇒ a sequence of 1 BasicDecBlk that
        // maps (deepest channel + cxt sum) → deepest channel.
        let lateral = config.lateralChannels
        let cxtSum = config.contextChannels.reduce(0, +)
        let deepCh = lateral.first ?? 0
        precondition(deepCh > 0, "lateralChannels empty")
        var sqz: [BasicDecBlk] = []
        if config.squeezeBlockEnabled {
            for _ in 0..<config.squeezeBlockCount {
                sqz.append(BasicDecBlk(
                    inChannels: deepCh + cxtSum,
                    outChannels: deepCh,
                    hasDecAtt: config.decoderAttention != .none
                ))
            }
        }
        _squeezeModule.wrappedValue = sqz

        _decoder.wrappedValue = BiRefNetDecoder(config: config)

        super.init()
    }

    /// Encoder forward with `mul_scl_ipt='cat'` and context concatenation.
    /// Inputs/outputs in NHWC.
    ///
    /// The full-res and half-res backbone passes are independent until the
    /// per-level `merge`, so we dispatch the half-res call on a fresh GPU
    /// stream. With two queues in flight MLX overlaps the half-res
    /// transformer work with the tail of the full-res pass on a single
    /// device. Measured win on Swin-L @ 1024²: ~25 ms off the encoder share.
    func forwardEnc(_ x: MLXArray) -> [MLXArray] {
        // (x1, x2, x3, x4) at full resolution — runs on the default stream.
        let full = bb(x)
        precondition(full.count == 4)

        // mul_scl_ipt='cat': also run on half-res input and concat per level.
        var x1 = full[0]
        var x2 = full[1]
        var x3 = full[2]
        var x4 = full[3]
        if config.multiScaleInput != .none {
            let H = x.dim(1)
            let W = x.dim(2)
            // Half-res path on a new GPU stream so it overlaps with the
            // full-res path. The merge below introduces the cross-stream
            // dependency; MLX inserts the right barrier at eval time.
            let half = Stream.withNewDefaultStream(device: .gpu) { () -> [MLXArray] in
                let xHalf = bilinearResize(x, H / 2, W / 2)
                return bb(xHalf)
            }

            func merge(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
                let bUp = bilinearResize(b, a.dim(1), a.dim(2))
                switch config.multiScaleInput {
                case .cat: return concatenated([a, bUp], axis: -1)
                case .add: return a + bUp
                case .none: return a
                }
            }
            x1 = merge(x1, half[0])
            x2 = merge(x2, half[1])
            x3 = merge(x3, half[2])
            x4 = merge(x4, half[3])
        }

        // Context skip: concatenate upsampled x1/x2/x3 (last cxtNum of them, reversed)
        // onto x4 along channel axis.
        // Python: cxt = lateral[1:][::-1][-cxtNum:]  ⇒ for cxtNum=3 (Swin):
        //   cxt = reversed([x2_C, x3_C, x4_C])[-3:] = [x4_C, x3_C, x2_C][-3:] = [x4_C, x3_C, x2_C]
        // Wait — Python uses ascending order x1,x2,x3,x4 with channels
        //   lateralChannels = [x4_C, x3_C, x2_C, x1_C]   # deep first
        // Then cxt = lateral[1:][::-1][-cxtNum:] = [x3_C, x2_C, x1_C][::-1][-3:] = [x1_C, x2_C, x3_C]
        // The concatenation order in forward_enc is fixed:
        //   torch.cat([F.interp(x1, x4_size), F.interp(x2, x4_size), F.interp(x3, x4_size)][-cxtNum:], x4)
        // So with cxtNum=3 we use [x1, x2, x3] (all three) resized to x4.
        if config.cxtNum > 0 {
            let H4 = x4.dim(1); let W4 = x4.dim(2)
            let raw: [MLXArray] = [x1, x2, x3].map { bilinearResize($0, H4, W4) }
            let used = Array(raw.suffix(config.cxtNum))
            x4 = concatenated(used + [x4], axis: -1)
        }

        return [x, x1, x2, x3, x4]
    }

    /// Forward. `x` is `(B, H, W, 3)` NHWC. Returns `(B, H, W, 1)` logits.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var features = forwardEnc(x)
        // squeeze_module operates on x4 (last feature). In Python the
        // squeeze_module is an `nn.Sequential` of BasicDecBlk_x1 blocks.
        if !squeezeModule.isEmpty {
            var x4 = features[4]
            for blk in squeezeModule {
                x4 = blk(x4)
            }
            features[4] = x4
        }
        return decoder(features)
    }
}
