// Public weight conversion API: PyTorch safetensors → MLX-Swift safetensors.
//
// Pulled out of the CLI's Convert.swift so the library can be embedded by
// applications (e.g., the demo in Examples/) that want to download an
// official BiRefNet checkpoint and convert it inline without shelling out
// to a separate binary.

import Foundation
import MLX

public enum BiRefNetWeights {

    public struct ConversionStats: Sendable {
        public let totalInputTensors: Int
        public let totalOutputTensors: Int
        public let droppedTensors: Int
    }

    /// Convert an official BiRefNet PyTorch `.safetensors` checkpoint into a
    /// key-remapped, NHWC-transposed `.safetensors` that the Swift `BiRefNet`
    /// module tree loads with `verify: [.noUnusedKeys]` passing.
    ///
    /// Steps:
    /// 1. Strip optional checkpoint prefixes (`model.`, `module.`) added by
    ///    HF wrappers.
    /// 2. Drop deterministic / PyTorch-only buffers (`relative_position_index`,
    ///    `num_batches_tracked`) that the Swift port either recomputes locally
    ///    or doesn't carry.
    /// 3. Remap `decoder.conv_out1.0.{w,b}` → `decoder.conv_out1.{w,b}`
    ///    (Python wraps the final Conv2d in `nn.Sequential`; the Swift port
    ///    uses a bare `Conv2d`).
    /// 4. Transpose every 4-D tensor from PyTorch NCHW Conv2d layout
    ///    `(out, in, kH, kW)` to MLX NHWC `(out, kH, kW, in)`.
    /// 5. Cast every tensor to `dtype`.
    ///
    /// The converter pins to the CPU device so it can run from binaries
    /// without a Metal-capable runtime (e.g. `swift build` output binaries).
    public static func convert(
        inputURL: URL,
        outputURL: URL,
        dtype: DType = .float32,
        verbose: Bool = false,
        progress: ((Double) -> Void)? = nil
    ) throws -> ConversionStats {
        // Conversion is pure file I/O + reshape + cast and must NOT pin the
        // GPU. Use the scoped form so the rest of the process keeps its
        // existing default device — otherwise an in-process caller (the
        // demo app) would silently switch to CPU after conversion and the
        // custom Metal kernel would refuse to dispatch later
        // ("[metal_kernel] Only supports the GPU").
        try Device.withDefaultDevice(.cpu) {
            try convertInner(
                inputURL: inputURL, outputURL: outputURL,
                dtype: dtype, verbose: verbose, progress: progress)
        }
    }

    private static func convertInner(
        inputURL: URL,
        outputURL: URL,
        dtype: DType,
        verbose: Bool,
        progress: ((Double) -> Void)?
    ) throws -> ConversionStats {
        let raw = try MLX.loadArrays(url: inputURL)

        var converted: [String: MLXArray] = [:]
        converted.reserveCapacity(raw.count)
        var dropped = 0
        var seen = 0

        for (origKey, tensor) in raw {
            seen += 1
            if let progress { progress(Double(seen) / Double(raw.count) * 0.5) }

            var key = origKey
            for prefix in ["model.", "module."] {
                if key.hasPrefix(prefix) { key.removeFirst(prefix.count) }
            }
            if key.hasSuffix(".relative_position_index")
                || key.hasSuffix(".num_batches_tracked")
            {
                dropped += 1
                if verbose { print("DROP   \(origKey)") }
                continue
            }
            key = remapConvOut1(key)

            var value = tensor
            if value.ndim == 4 {
                value = value.transposed(0, 2, 3, 1)
            }
            if value.dtype != dtype {
                value = value.asType(dtype)
            }
            if verbose { print("KEEP   \(origKey) → \(key) shape=\(value.shape)") }
            converted[key] = value
        }

        // Force materialisation so safetensors gets concrete bytes.
        eval(Array(converted.values))
        if let progress { progress(0.9) }

        try MLX.save(arrays: converted, url: outputURL)
        if let progress { progress(1.0) }

        return ConversionStats(
            totalInputTensors: raw.count,
            totalOutputTensors: converted.count,
            droppedTensors: dropped
        )
    }

    /// `decoder.conv_out1.0.weight` → `decoder.conv_out1.weight`.
    private static func remapConvOut1(_ key: String) -> String {
        guard key.contains("conv_out1.0.") else { return key }
        return key.replacingOccurrences(of: "conv_out1.0.", with: "conv_out1.")
    }
}
