import Foundation
import MLX
import MLXNN

/// BiRefNet `Decoder`. Inference-side port of `models/birefnet.py::Decoder`.
///
/// Training-only outputs (`m4/m3/m2`, GT/pred branches under `out_ref`) are
/// dropped, but the `gdt_convs_*` + `gdt_attn_*` sigmoid attention IS on the
/// inference path and is preserved.
///
/// Shapes (NHWC, with `mul_scl_ipt='cat'` doubling the lateral channels for
/// the Swin-L config):
/// - Decoder input `features = [x, x1, x2, x3, x4]`:
///   - `x`  : (B, H,    W,    3)          original image
///   - `x1` : (B, H/4,  W/4,  3072)       deepest (already squeeze-blocked + cxt-cat)
///   - `x2` : (B, H/8,  W/8,  1536)
///   - `x3` : (B, H/16, W/16, 768)
///   - `x4` : (B, H/32, W/32, 384)
/// Wait — the Python order is (x, x1, x2, x3, x4) with x1 shallowest and x4 deepest.
/// We follow Python.
public final class BiRefNetDecoder: Module {

    public let config: BiRefNetConfig

    // Lateral 1×1 convs (deepest-side context skip into each level).
    @ModuleInfo(key: "lateral_block4") var lateralBlock4: BasicLatBlk
    @ModuleInfo(key: "lateral_block3") var lateralBlock3: BasicLatBlk
    @ModuleInfo(key: "lateral_block2") var lateralBlock2: BasicLatBlk

    // Decoder blocks (deepest → shallowest).
    @ModuleInfo(key: "decoder_block4") var decoderBlock4: BasicDecBlk
    @ModuleInfo(key: "decoder_block3") var decoderBlock3: BasicDecBlk
    @ModuleInfo(key: "decoder_block2") var decoderBlock2: BasicDecBlk
    @ModuleInfo(key: "decoder_block1") var decoderBlock1: BasicDecBlk

    // Image-patch injection branches (`dec_ipt`). Five of them, one per stage.
    @ModuleInfo(key: "ipt_blk1") var iptBlk1: SimpleConvs
    @ModuleInfo(key: "ipt_blk2") var iptBlk2: SimpleConvs
    @ModuleInfo(key: "ipt_blk3") var iptBlk3: SimpleConvs
    @ModuleInfo(key: "ipt_blk4") var iptBlk4: SimpleConvs
    @ModuleInfo(key: "ipt_blk5") var iptBlk5: SimpleConvs

    // Final 1×1 conv to a single-channel logit map.
    @ModuleInfo(key: "conv_out1") var convOut1: Conv2d

    // out_ref: gdt convs + attention. `gdt_convs_{2,3,4} = Sequential(Conv 3x3 → BN → ReLU)`,
    // and `gdt_convs_attn_{2,3,4} = Sequential(Conv 1x1)`. The pred branch
    // (`gdt_convs_pred_*`) is training-only; we keep it allocated for weight
    // loading completeness but never call it.
    @ModuleInfo(key: "gdt_convs_2") var gdtConvs2: [UnaryLayer]
    @ModuleInfo(key: "gdt_convs_3") var gdtConvs3: [UnaryLayer]
    @ModuleInfo(key: "gdt_convs_4") var gdtConvs4: [UnaryLayer]
    @ModuleInfo(key: "gdt_convs_attn_2") var gdtConvsAttn2: [UnaryLayer]
    @ModuleInfo(key: "gdt_convs_attn_3") var gdtConvsAttn3: [UnaryLayer]
    @ModuleInfo(key: "gdt_convs_attn_4") var gdtConvsAttn4: [UnaryLayer]
    @ModuleInfo(key: "gdt_convs_pred_2") var gdtConvsPred2: [UnaryLayer]
    @ModuleInfo(key: "gdt_convs_pred_3") var gdtConvsPred3: [UnaryLayer]
    @ModuleInfo(key: "gdt_convs_pred_4") var gdtConvsPred4: [UnaryLayer]

    // ms_supervision: conv_ms_spvn_{2,3,4} — training-only output heads.
    // Allocated to consume the checkpoint keys but never called.
    @ModuleInfo(key: "conv_ms_spvn_2") var convMsSpvn2: Conv2d
    @ModuleInfo(key: "conv_ms_spvn_3") var convMsSpvn3: Conv2d
    @ModuleInfo(key: "conv_ms_spvn_4") var convMsSpvn4: Conv2d

    /// Per-stage in/out channel widths derived from the Python:
    /// `bb_neck_out_channels = lateralChannels.copy()`
    /// `dec_blk_out_channels = bb_neck_out_channels[1:] + [bb_neck_out_channels[-1] // 2]`
    /// (use_pyramid_neck is False for Swin)
    /// `ipt_blk_out_channels = [N_dec_ipt]*4` with `N_dec_ipt = 64`
    /// `dec_blk_in_channels = [bb_neck[i] + ipt_out[max(0,i-1)] for i in range(4)]`
    let bbNeckOut: [Int]
    let decBlkOut: [Int]
    let iptOut: [Int]
    let iptIn: [Int]   // image-patch input channel widths per stage (depends on dec_ipt_split)

    static let N_DEC_IPT = 64

    public init(config: BiRefNetConfig) {
        self.config = config

        // ----- channel arithmetic -----
        let lateral = config.lateralChannels  // doubled by mul_scl_ipt='cat'
        let _bbNeck = lateral
        let _decBlkOut = [lateral[1], lateral[2], lateral[3], lateral[3] / 2]
        let _iptOut = [Int](repeating: Self.N_DEC_IPT, count: 4)
        let _iptIn: [Int] = config.decoderInputSplit
            ? [3072, 768, 192, 48, 3]
            : [Int](repeating: 3, count: 5)
        let decBlkIn = (0..<lateral.count).map { i in
            lateral[i] + _iptOut[Swift.max(0, i - 1)]
        }
        self.bbNeckOut = _bbNeck
        self.decBlkOut = _decBlkOut
        self.iptOut = _iptOut
        self.iptIn = _iptIn

        // ----- lateral 1x1 convs (lateral_block4: lateral[1] -> decBlkOut[0], etc.) -----
        _lateralBlock4.wrappedValue = BasicLatBlk(inChannels: lateral[1], outChannels: _decBlkOut[0])
        _lateralBlock3.wrappedValue = BasicLatBlk(inChannels: lateral[2], outChannels: _decBlkOut[1])
        _lateralBlock2.wrappedValue = BasicLatBlk(inChannels: lateral[3], outChannels: _decBlkOut[2])

        // ----- four decoder blocks -----
        let hasDecAtt = config.decoderAttention != .none
        _decoderBlock4.wrappedValue = BasicDecBlk(
            inChannels: decBlkIn[0], outChannels: _decBlkOut[0], hasDecAtt: hasDecAtt)
        _decoderBlock3.wrappedValue = BasicDecBlk(
            inChannels: decBlkIn[1], outChannels: _decBlkOut[1], hasDecAtt: hasDecAtt)
        _decoderBlock2.wrappedValue = BasicDecBlk(
            inChannels: decBlkIn[2], outChannels: _decBlkOut[2], hasDecAtt: hasDecAtt)
        _decoderBlock1.wrappedValue = BasicDecBlk(
            inChannels: decBlkIn[3], outChannels: _decBlkOut[3], hasDecAtt: hasDecAtt)

        // ----- ipt blocks (image-patch injection) -----
        _iptBlk5.wrappedValue = SimpleConvs(inChannels: _iptIn[0], outChannels: _iptOut[0])
        _iptBlk4.wrappedValue = SimpleConvs(inChannels: _iptIn[1], outChannels: _iptOut[0])
        _iptBlk3.wrappedValue = SimpleConvs(inChannels: _iptIn[2], outChannels: _iptOut[1])
        _iptBlk2.wrappedValue = SimpleConvs(inChannels: _iptIn[3], outChannels: _iptOut[2])
        _iptBlk1.wrappedValue = SimpleConvs(inChannels: _iptIn[4], outChannels: _iptOut[3])

        // ----- final conv_out1 -----
        let convOutIn = _decBlkOut[3] + (config.decoderInputInjection ? _iptOut[3] : 0)
        _convOut1.wrappedValue = Conv2d(
            inputChannels: convOutIn, outputChannels: 1,
            kernelSize: .init(1), stride: .init(1), padding: .init(0), bias: true
        )

        // ----- ms_supervision heads (training-only path, kept for weight loading) -----
        _convMsSpvn4.wrappedValue = Conv2d(
            inputChannels: _decBlkOut[0], outputChannels: 1,
            kernelSize: .init(1), stride: .init(1), padding: .init(0), bias: true)
        _convMsSpvn3.wrappedValue = Conv2d(
            inputChannels: _decBlkOut[1], outputChannels: 1,
            kernelSize: .init(1), stride: .init(1), padding: .init(0), bias: true)
        _convMsSpvn2.wrappedValue = Conv2d(
            inputChannels: _decBlkOut[2], outputChannels: 1,
            kernelSize: .init(1), stride: .init(1), padding: .init(0), bias: true)

        // ----- gdt convs (out_ref) -----
        let N = 16
        func makeGdt(in c: Int) -> [UnaryLayer] {
            [
                Conv2d(inputChannels: c, outputChannels: N,
                       kernelSize: .init(3), stride: .init(1), padding: .init(1), bias: true),
                BatchNorm(featureCount: N),
                ReLU(),
            ]
        }
        func make1x1(in c: Int) -> [UnaryLayer] {
            [
                Conv2d(inputChannels: c, outputChannels: 1,
                       kernelSize: .init(1), stride: .init(1), padding: .init(0), bias: true),
            ]
        }
        _gdtConvs4.wrappedValue = makeGdt(in: _decBlkOut[0])
        _gdtConvs3.wrappedValue = makeGdt(in: _decBlkOut[1])
        _gdtConvs2.wrappedValue = makeGdt(in: _decBlkOut[2])
        _gdtConvsAttn4.wrappedValue = make1x1(in: N)
        _gdtConvsAttn3.wrappedValue = make1x1(in: N)
        _gdtConvsAttn2.wrappedValue = make1x1(in: N)
        _gdtConvsPred4.wrappedValue = make1x1(in: N)
        _gdtConvsPred3.wrappedValue = make1x1(in: N)
        _gdtConvsPred2.wrappedValue = make1x1(in: N)

        super.init()
    }

    /// Image-patches concatenation step. Mirrors Python:
    ///
    /// ```python
    /// patches_batch = image2patches(x, patch_ref=x4,
    ///                               transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
    /// x4 = torch.cat((x4, self.ipt_blkN(F.interpolate(patches_batch, size=x4.shape[2:], ...))), 1)
    /// ```
    ///
    /// In NHWC: stack tiles along the channel axis, resize to the target
    /// feature spatial size, run through `iptBlk`, concatenate on `C`.
    private func injectImagePatches(_ x: MLXArray, feature f: MLXArray, iptBlk: SimpleConvs) -> MLXArray {
        let H = x.dim(1); let W = x.dim(2)
        let fH = f.dim(1); let fW = f.dim(2)
        let patches: MLXArray
        if config.decoderInputSplit {
            let gridH = H / fH
            let gridW = W / fW
            patches = image2patches(x, gridH: gridH, gridW: gridW)
        } else {
            patches = x
        }
        let resized = bilinearResize(patches, fH, fW)
        let projected = iptBlk(resized)
        return concatenated([f, projected], axis: -1)
    }

    /// Forward. `features = [x, x1, x2, x3, x4]`. Returns single-channel
    /// logits at the resolution of `x`.
    public func callAsFunction(_ features: [MLXArray]) -> MLXArray {
        precondition(features.count == 5, "Decoder expects [x, x1, x2, x3, x4]")
        let x  = features[0]
        var x1 = features[1]
        var x2 = features[2]
        var x3 = features[3]
        var x4 = features[4]
        _ = x1; _ = x2; _ = x3  // mutability sanity (pyramid neck is off)

        // Stage 4 ----------------------------------------------------------
        if config.decoderInputInjection {
            x4 = injectImagePatches(x, feature: x4, iptBlk: iptBlk5)
        }
        var p4 = decoderBlock4(x4)
        if config.outRef {
            let g = runSequential(gdtConvs4, on: p4)
            let attn = sigmoid(runSequential(gdtConvsAttn4, on: g))
            p4 = p4 * attn
        }

        // Up to stage 3 ----------------------------------------------------
        var p4Up = bilinearResize(p4, x3.dim(1), x3.dim(2))
        var p3 = p4Up + lateralBlock4(x3)
        if config.decoderInputInjection {
            p3 = injectImagePatches(x, feature: p3, iptBlk: iptBlk4)
        }
        p3 = decoderBlock3(p3)
        if config.outRef {
            let g = runSequential(gdtConvs3, on: p3)
            let attn = sigmoid(runSequential(gdtConvsAttn3, on: g))
            p3 = p3 * attn
        }
        _ = p4Up

        // Up to stage 2 ----------------------------------------------------
        var p3Up = bilinearResize(p3, x2.dim(1), x2.dim(2))
        var p2 = p3Up + lateralBlock3(x2)
        if config.decoderInputInjection {
            p2 = injectImagePatches(x, feature: p2, iptBlk: iptBlk3)
        }
        p2 = decoderBlock2(p2)
        if config.outRef {
            let g = runSequential(gdtConvs2, on: p2)
            let attn = sigmoid(runSequential(gdtConvsAttn2, on: g))
            p2 = p2 * attn
        }
        _ = p3Up

        // Up to stage 1 ----------------------------------------------------
        var p2Up = bilinearResize(p2, x1.dim(1), x1.dim(2))
        var p1 = p2Up + lateralBlock2(x1)
        if config.decoderInputInjection {
            p1 = injectImagePatches(x, feature: p1, iptBlk: iptBlk2)
        }
        p1 = decoderBlock1(p1)
        _ = p2Up

        // Up to image resolution -------------------------------------------
        var pOut = bilinearResize(p1, x.dim(1), x.dim(2))
        if config.decoderInputInjection {
            pOut = injectImagePatches(x, feature: pOut, iptBlk: iptBlk1)
        }
        return convOut1(pOut)
    }
}

/// Lightweight `nn.Sequential` runner — applies each `UnaryLayer` in order.
@inlinable
public func runSequential(_ layers: [UnaryLayer], on x: MLXArray) -> MLXArray {
    var v = x
    for l in layers { v = l(v) }
    return v
}
