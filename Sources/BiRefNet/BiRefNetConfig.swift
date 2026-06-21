import Foundation
import MLX

/// Frozen subset of the Python `Config` that is actually consumed by the
/// inference path of BiRefNet. The defaults match `models/birefnet.py` with
/// `bb='swin_v1_l'`, `mul_scl_ipt='cat'`, `cxt_num=3`, `dec_att='ASPPDeformable'`.
public struct BiRefNetConfig: Sendable {

    public enum Backbone: String, Sendable {
        case swinV1L = "swin_v1_l"
        case swinV1B = "swin_v1_b"
        case swinV1S = "swin_v1_s"
        case swinV1T = "swin_v1_t"
    }

    public enum MultiScaleInput: String, Sendable {
        case none = ""
        case add
        case cat
    }

    public enum DecoderAttention: String, Sendable {
        case none = ""
        case aspp = "ASPP"
        case asppDeformable = "ASPPDeformable"
    }

    public var backbone: Backbone = .swinV1L
    public var multiScaleInput: MultiScaleInput = .cat
    public var cxtNum: Int = 3
    public var decoderAttention: DecoderAttention = .asppDeformable
    public var decoderBlock: String = "BasicDecBlk"  // we only ship BasicDecBlk
    public var lateralBlock: String = "BasicLatBlk"
    public var decoderInputInjection: Bool = true
    public var decoderInputSplit: Bool = true
    public var squeezeBlockEnabled: Bool = true     // "BasicDecBlk_x1"
    public var squeezeBlockCount: Int = 1
    public var msSupervision: Bool = true            // only relevant for training outputs
    public var outRef: Bool = true                   // gdt_convs / gdt_attn are on inference path
    /// Checkpoint was trained with batch_size > 1 ⇒ BN layers are present and
    /// have real running statistics (not Identity).
    public var batchNormEnabled: Bool = true
    /// Default training input resolution. Inference can run other sizes.
    public var inputSize: (width: Int, height: Int) = (1024, 1024)

    /// Per-stage channel widths for the chosen backbone (BEFORE `mul_scl_ipt`
    /// doubling). Ordered from deepest to shallowest, matching
    /// `lateral_channels_in_collection[bb]` in Python.
    public var lateralChannelsRaw: [Int] {
        switch backbone {
        case .swinV1L: return [1536, 768, 384, 192]
        case .swinV1B: return [1024, 512, 256, 128]
        case .swinV1S: return [768, 384, 192, 96]
        case .swinV1T: return [768, 384, 192, 96]
        }
    }

    /// Per-stage channel widths AFTER `mul_scl_ipt='cat'` doubling, when enabled.
    public var lateralChannels: [Int] {
        let raw = lateralChannelsRaw
        return multiScaleInput == .cat ? raw.map { $0 * 2 } : raw
    }

    /// Context channels appended to the deepest feature map. Mirrors
    /// `config.cxt = lateral[1:][::-1][-cxtNum:]` from Python.
    public var contextChannels: [Int] {
        guard cxtNum > 0 else { return [] }
        let trimmed = Array(lateralChannels.dropFirst()).reversed()
        return Array(trimmed.suffix(cxtNum))
    }

    public init() {}

    public static let swinLargeDefault: BiRefNetConfig = BiRefNetConfig()
}

// MARK: - Swin Backbone Specs

public struct SwinSpec: Sendable {
    public let embedDim: Int
    public let depths: [Int]
    public let numHeads: [Int]
    public let windowSize: Int
    public let mlpRatio: Float = 4.0
    public let patchSize: Int = 4
    public let inChannels: Int = 3
    public let qkvBias: Bool = true

    public static let swinV1L = SwinSpec(
        embedDim: 192, depths: [2, 2, 18, 2], numHeads: [6, 12, 24, 48], windowSize: 12
    )
    public static let swinV1B = SwinSpec(
        embedDim: 128, depths: [2, 2, 18, 2], numHeads: [4, 8, 16, 32], windowSize: 12
    )
    public static let swinV1S = SwinSpec(
        embedDim: 96, depths: [2, 2, 18, 2], numHeads: [3, 6, 12, 24], windowSize: 7
    )
    public static let swinV1T = SwinSpec(
        embedDim: 96, depths: [2, 2, 6, 2], numHeads: [3, 6, 12, 24], windowSize: 7
    )

    public static func forBackbone(_ bb: BiRefNetConfig.Backbone) -> SwinSpec {
        switch bb {
        case .swinV1L: return .swinV1L
        case .swinV1B: return .swinV1B
        case .swinV1S: return .swinV1S
        case .swinV1T: return .swinV1T
        }
    }
}
