import Foundation
import MLX
import MLXNN

/// Two 3×3 convolutions with no activation between them. Used as
/// `ipt_blk{1..5}` in the BiRefNet decoder.
public final class SimpleConvs: Module, UnaryLayer {

    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    public init(inChannels: Int, outChannels: Int, interChannels: Int = 64) {
        _conv1.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: interChannels,
            kernelSize: .init(3),
            stride: .init(1),
            padding: .init(1),
            bias: true
        )
        _convOut.wrappedValue = Conv2d(
            inputChannels: interChannels,
            outputChannels: outChannels,
            kernelSize: .init(3),
            stride: .init(1),
            padding: .init(1),
            bias: true
        )
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return convOut(conv1(x))
    }
}
