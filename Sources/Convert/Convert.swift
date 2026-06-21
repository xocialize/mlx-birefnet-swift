import Foundation
import ArgumentParser
import MLX
import BiRefNet

/// Converts an official BiRefNet PyTorch safetensors checkpoint into a
/// key-remapped, NHWC-transposed `.safetensors` consumable by the Swift
/// `BiRefNet` module tree. Thin wrapper around
/// `BiRefNetWeights.convert(inputURL:outputURL:dtype:)` from the
/// `BiRefNetSegmenter` library.
///
/// **Input requirement**: a `.safetensors` file. If you only have a
/// `pytorch_model.bin` (pickle), convert it first with:
///
/// ```bash
/// python3 - <<'PY'
/// import torch, safetensors.torch as st
/// sd = torch.load("pytorch_model.bin", map_location="cpu", weights_only=True)
/// for p in ["model.", "module."]:
///     sd = {(k[len(p):] if k.startswith(p) else k): v for k, v in sd.items()}
/// st.save_file({k: v.contiguous() for k, v in sd.items()}, "model.safetensors")
/// PY
/// ```
@main
struct ConvertCommand: ParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "birefnet-convert",
        abstract: "Convert official BiRefNet PyTorch weights to MLX-Swift safetensors."
    )

    @Argument(help: "Path to the input PyTorch safetensors checkpoint.")
    var input: String

    @Argument(help: "Path to write the converted MLX safetensors.")
    var output: String

    @Option(name: .long, help: "Target dtype for the converted weights (float32 / float16 / bfloat16).")
    var dtype: String = "float32"

    @Flag(help: "Print every key being processed.")
    var verbose: Bool = false

    mutating func run() throws {
        let targetDType: DType
        switch dtype.lowercased() {
        case "float32", "fp32", "f32": targetDType = .float32
        case "float16", "fp16", "f16": targetDType = .float16
        case "bfloat16", "bf16": targetDType = .bfloat16
        default: throw ValidationError("unknown dtype: \(dtype)")
        }

        FileHandle.standardError.write("Loading \(input) …\n".data(using: .utf8)!)
        let stats = try BiRefNetWeights.convert(
            inputURL: URL(fileURLWithPath: input),
            outputURL: URL(fileURLWithPath: output),
            dtype: targetDType,
            verbose: verbose
        )
        let line = "Loaded \(stats.totalInputTensors) tensors. Wrote "
            + "\(stats.totalOutputTensors) (dropped \(stats.droppedTensors)) to \(output).\n"
        FileHandle.standardError.write(line.data(using: .utf8)!)
    }
}
