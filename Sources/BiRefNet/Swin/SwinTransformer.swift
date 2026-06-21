import Foundation
import MLX
import MLXNN

/// Swin Transformer v1 backbone. Outputs four feature maps in NHWC,
/// matching `outIndices = (0,1,2,3)` from the Python reference.
public final class SwinTransformer: Module {

    public let spec: SwinSpec

    @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
    @ModuleInfo(key: "layers") var layers: [BasicLayer]

    // One LayerNorm per output stage. Named `norm0`, `norm1`, `norm2`, `norm3`
    // in the PyTorch state_dict (registered via `add_module`).
    @ModuleInfo(key: "norm0") var norm0: LayerNorm
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "norm3") var norm3: LayerNorm

    public init(spec: SwinSpec) {
        self.spec = spec

        _patchEmbed.wrappedValue = PatchEmbed(
            patchSize: spec.patchSize,
            inChannels: spec.inChannels,
            embedDim: spec.embedDim,
            patchNorm: true
        )

        let nLayers = spec.depths.count
        var built: [BasicLayer] = []
        for i in 0..<nLayers {
            let dim = spec.embedDim * (1 << i)
            built.append(
                BasicLayer(
                    dim: dim,
                    depth: spec.depths[i],
                    numHeads: spec.numHeads[i],
                    windowSize: spec.windowSize,
                    mlpRatio: spec.mlpRatio,
                    qkvBias: spec.qkvBias,
                    hasDownsample: i < nLayers - 1
                )
            )
        }
        _layers.wrappedValue = built

        let chs = (0..<nLayers).map { spec.embedDim * (1 << $0) }
        _norm0.wrappedValue = LayerNorm(dimensions: chs[0])
        _norm1.wrappedValue = LayerNorm(dimensions: chs[1])
        _norm2.wrappedValue = LayerNorm(dimensions: chs[2])
        _norm3.wrappedValue = LayerNorm(dimensions: chs[3])

        super.init()
    }

    /// Convenience: build for a named Swin variant.
    public convenience init(_ backbone: BiRefNetConfig.Backbone) {
        self.init(spec: SwinSpec.forBackbone(backbone))
    }

    /// Input: `(B, H, W, 3)` NHWC float tensor (already normalised).
    /// Output: four NHWC feature maps `[x1, x2, x3, x4]` ordered from shallow to deep.
    public func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        let feat = patchEmbed(x)                       // (B, Wh, Ww, C)
        let B = feat.dim(0)
        var Wh = feat.dim(1)
        var Ww = feat.dim(2)
        var tokens = feat.reshaped([B, Wh * Ww, feat.dim(3)])

        let norms: [LayerNorm] = [norm0, norm1, norm2, norm3]
        var outs: [MLXArray] = []
        for (i, layer) in layers.enumerated() {
            let o = layer(tokens, H: Wh, W: Ww)
            let normalised = norms[i](o.out)
            let ch = normalised.dim(-1)
            outs.append(normalised.reshaped([B, o.H, o.W, ch]))
            tokens = o.down
            Wh = o.downH
            Ww = o.downW
        }
        return outs
    }
}
