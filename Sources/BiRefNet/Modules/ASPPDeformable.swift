import Foundation
import MLX
import MLXNN

/// Single ASPP-Deformable branch: DCN → BN → ReLU. Bias on the deformable
/// conv is always `False` per the Python (`_ASPPModuleDeformable`).
public final class ASPPModuleDeformable: Module, UnaryLayer {

    @ModuleInfo(key: "atrous_conv") var atrousConv: DeformableConv2d
    @ModuleInfo(key: "bn") var bn: BatchNorm

    public init(inChannels: Int, outChannels: Int, kernelSize: Int, padding: Int) {
        _atrousConv.wrappedValue = DeformableConv2d(
            inChannels: inChannels, outChannels: outChannels,
            kernelSize: kernelSize, padding: padding
        )
        _bn.wrappedValue = BatchNorm(featureCount: outChannels)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = atrousConv(x)
        y = bn(y)
        return relu(y)
    }
}

/// Mirrors `ASPPDeformable` from `aspp.py`:
///
/// - 1×1 deformable branch on the input
/// - N parallel deformable branches at `parallel_block_sizes = [1, 3, 7]`
/// - global average pooling branch (avg → 1×1 conv → BN → ReLU → upsample)
/// - concat → 1×1 conv → BN → ReLU → dropout
public final class ASPPDeformable: Module, UnaryLayer {

    let inChannelster: Int

    @ModuleInfo(key: "aspp1") var aspp1: ASPPModuleDeformable
    @ModuleInfo(key: "aspp_deforms") var asppDeforms: [ASPPModuleDeformable]
    @ModuleInfo(key: "global_avg_pool") var globalAvgPool: [UnaryLayer]
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "bn1") var bn1: BatchNorm

    public init(
        inChannels: Int,
        outChannels: Int? = nil,
        parallelBlockSizes: [Int] = [1, 3, 7]
    ) {
        let outCh = outChannels ?? inChannels
        let inCh = 256  // `in_channelster` in Python
        inChannelster = inCh

        _aspp1.wrappedValue = ASPPModuleDeformable(
            inChannels: inChannels, outChannels: inCh, kernelSize: 1, padding: 0)

        var deforms: [ASPPModuleDeformable] = []
        for k in parallelBlockSizes {
            deforms.append(ASPPModuleDeformable(
                inChannels: inChannels, outChannels: inCh,
                kernelSize: k, padding: k / 2))
        }
        _asppDeforms.wrappedValue = deforms

        // global_avg_pool is `nn.Sequential(AdaptiveAvgPool2d(1) → Conv2d → BN → ReLU)`.
        // PyTorch serialises it with keys `global_avg_pool.0`, `.1`, `.2`, `.3`. The
        // AdaptiveAvgPool has no params (slot 0); Conv2d slot 1; BN slot 2; ReLU slot 3.
        // We mirror that with `[UnaryLayer]` and a no-op pool head, so PyTorch keys
        // `.1.weight`, `.2.weight`, etc. map cleanly.
        _globalAvgPool.wrappedValue = [
            IdentityAtt(),   // placeholder for AdaptiveAvgPool2d (handled in forward)
            Conv2d(inputChannels: inChannels, outputChannels: inCh,
                   kernelSize: .init(1), stride: .init(1), padding: .init(0), bias: false),
            BatchNorm(featureCount: inCh),
            ReLU(),
        ]

        let concatCh = inCh * (2 + parallelBlockSizes.count)  // 1×1 + N deforms + GAP
        _conv1.wrappedValue = Conv2d(
            inputChannels: concatCh, outputChannels: outCh,
            kernelSize: .init(1), stride: .init(1), padding: .init(0), bias: false)
        _bn1.wrappedValue = BatchNorm(featureCount: outCh)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x1 = aspp1(x)
        let deformOuts = asppDeforms.map { $0(x) }

        // Global average pool over spatial dims.
        let pooled = x.mean(axes: [1, 2], keepDims: true)  // (B, 1, 1, Cin)
        var gap = pooled
        // skip slot 0 (the AdaptiveAvgPool2d placeholder)
        for i in 1..<globalAvgPool.count {
            gap = globalAvgPool[i](gap)
        }
        // Upsample to x1 spatial.
        gap = bilinearResize(gap, x1.dim(1), x1.dim(2))

        let pieces: [MLXArray] = [x1] + deformOuts + [gap]
        var y = concatenated(pieces, axis: -1)
        y = conv1(y)
        y = bn1(y)
        return relu(y)
        // Dropout (Python `nn.Dropout(0.5)`) is identity at inference.
    }
}
