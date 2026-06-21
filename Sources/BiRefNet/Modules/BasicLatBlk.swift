import Foundation
import MLX
import MLXNN

/// Mirrors `BasicLatBlk` from `lateral_blocks.py`: a single 1×1 Conv2d.
public final class BasicLatBlk: Module, UnaryLayer {

    @ModuleInfo(key: "conv") var conv: Conv2d

    public init(inChannels: Int, outChannels: Int, kernelSize: Int = 1, stride: Int = 1,
                padding: Int = 0) {
        _conv.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: .init(kernelSize),
            stride: .init(stride),
            padding: .init(padding),
            bias: true
        )
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return conv(x)
    }
}
