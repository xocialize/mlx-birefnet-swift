import Foundation
import MLX
import MLXNN

/// Pass-through `dec_att` (used while the real `ASPPDeformable` is being
/// implemented). Has no parameters.
public final class IdentityAtt: Module, UnaryLayer {
    public override init() { super.init() }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { x }
}

/// Mirrors `BasicDecBlk` from `decoder_blocks.py`:
///
/// ```
/// conv_in (3x3) → bn_in → ReLU → [dec_att?] → conv_out (3x3) → bn_out
/// ```
///
/// `bn_*` is real `BatchNorm` when the checkpoint was trained with
/// `batch_size > 1` (the default), otherwise `Identity`. We always allocate
/// the BN since the released BiRefNet ckpts were trained with `batch_size=8`.
public final class BasicDecBlk: Module {

    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "bn_in") var bnIn: BatchNorm
    @ModuleInfo(key: "dec_att") var decAtt: any UnaryLayer
    @ModuleInfo(key: "conv_out") var convOut: Conv2d
    @ModuleInfo(key: "bn_out") var bnOut: BatchNorm

    /// `hasDecAtt` mirrors the Python `if config.dec_att != '': self.dec_att = ...`.
    /// When `false`, no `dec_att.*` keys exist in the checkpoint.
    let hasDecAtt: Bool

    public init(
        inChannels: Int,
        outChannels: Int,
        interChannels: Int = 64,
        decoderAttention: any UnaryLayer = IdentityAtt(),
        hasDecAtt: Bool = true
    ) {
        self.hasDecAtt = hasDecAtt
        _convIn.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: interChannels,
            kernelSize: .init(3), stride: .init(1), padding: .init(1), bias: true
        )
        _bnIn.wrappedValue = BatchNorm(featureCount: interChannels)
        // If the caller didn't pass a concrete dec_att but says we have one,
        // default to ASPPDeformable on `interChannels` (matches the Python
        // BiRefNet config `dec_att='ASPPDeformable'`).
        if hasDecAtt, decoderAttention is IdentityAtt {
            _decAtt.wrappedValue = ASPPDeformable(inChannels: interChannels)
        } else {
            _decAtt.wrappedValue = decoderAttention
        }
        _convOut.wrappedValue = Conv2d(
            inputChannels: interChannels, outputChannels: outChannels,
            kernelSize: .init(3), stride: .init(1), padding: .init(1), bias: true
        )
        _bnOut.wrappedValue = BatchNorm(featureCount: outChannels)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = convIn(x)
        y = bnIn(y)
        y = relu(y)
        if hasDecAtt {
            y = decAtt(y)
        }
        y = convOut(y)
        y = bnOut(y)
        return y
    }
}
