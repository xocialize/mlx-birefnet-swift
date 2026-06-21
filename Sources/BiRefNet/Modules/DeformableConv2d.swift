import Foundation
import MLX
import MLXNN

/// Modulated deformable conv (DCNv2). Mirrors
/// `models/modules/deform_conv.py::DeformableConv2d` from BiRefNet.
///
/// Three sub-convs:
/// - `offset_conv`     : Conv2d(Cin → 2·kH·kW, k=kernelSize, padding=padding)
/// - `modulator_conv`  : Conv2d(Cin →   kH·kW, k=kernelSize, padding=padding)
/// - `regular_conv`    : Conv2d(Cin → Cout,    k=kernelSize, padding=padding, bias=false)
///
/// At inference, `regular_conv` is NOT invoked as a regular Conv2d — its
/// weight tensor is passed straight into `deformConv2dForward`. The Conv2d
/// shell is allocated only so the parameter tree matches the PyTorch
/// checkpoint.
public final class DeformableConv2d: Module {

    let kernelSizeHW: (Int, Int)
    let padding: Int

    @ModuleInfo(key: "offset_conv") var offsetConv: Conv2d
    @ModuleInfo(key: "modulator_conv") var modulatorConv: Conv2d
    @ModuleInfo(key: "regular_conv") var regularConv: Conv2d

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int = 3,
        padding: Int = 1
    ) {
        self.kernelSizeHW = (kernelSize, kernelSize)
        self.padding = padding

        let k = kernelSize * kernelSize
        _offsetConv.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: 2 * k,
            kernelSize: .init(kernelSize), stride: .init(1), padding: .init(padding), bias: true)
        _modulatorConv.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: 1 * k,
            kernelSize: .init(kernelSize), stride: .init(1), padding: .init(padding), bias: true)
        _regularConv.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: .init(kernelSize), stride: .init(1), padding: .init(padding), bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let offset = offsetConv(x)
        let modulator = 2.0 * sigmoid(modulatorConv(x))
        return deformConv2dForward(
            input: x,
            offset: offset,
            mask: modulator,
            weight: regularConv.weight,
            kernelSize: kernelSizeHW,
            padding: padding
        )
    }
}
