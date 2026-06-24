# mlx-birefnet-swift

The MLXEngine **`matting`** package over [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) — high-quality foreground matting (soft-alpha cutouts) on Apple Silicon.

Takes an `Image` and returns a single-channel `Matte` (0 = background … 1 = foreground) at source resolution — usable as a cutout source *and* as a reusable weight-map signal for other capabilities. A thin conformance wrapper (`MLXBiRefNet`) over a vendored Swift/MLX core (`BiRefNet`).

## Modes

One package, two quality tiers — dispatched on `MattingRequest.mode` (same Swin-L + ASPP-Deformable architecture, different weights + input size):

| Mode | Checkpoint | Input | Weights | Peak (fp16) |
|---|---|---|---|---|
| `MattingContract.fast` (default) | general | 1024² | `mlx-community/BiRefNet-fp16` | ~4.9 GB |
| `MattingContract.best` | HR-matting | 2048² | `mlx-community/BiRefNet_HR-matting-fp16` | ~18.3 GB |

Weights are fp16 (~420 MB each, IoU 0.9905) and **auto-download** on first use via `HubApi` into the engine's model store — no manual URLs. `best` is a pro-tier footprint: the package declares the `fast` envelope and applies a **runtime guard** on `best` (`BiRefNetError.insufficientMemoryForBest` when the device's recommended Metal working set is too small), so a consumer adapter can fall back to `fast`.

## Usage

```swift
import MLXServeCore
import MLXBiRefNet

let engine = MLXServeEngine()
try await engine.register(BiRefNetPackage.registration, configuration: BiRefNetConfiguration())

// fast (default)
let resp = try await engine.run(MattingRequest(image: photo)) as! MattingResponse
// resp.matte — soft-alpha grayscale .png, same dimensions as the input

// best tier
let hi = try await engine.run(
    MattingRequest(image: photo, mode: MattingContract.best)) as! MattingResponse
```

## Products

| Product | What it is |
|---|---|
| `MLXBiRefNet` | the engine-consumable `matting` ModelPackage (`BiRefNetPackage` + `BiRefNetConfiguration`) |
| `BiRefNet` | the vendored Swift/MLX core (Swin-L backbone + decoder) |
| `birefnet-convert` | PyTorch / HF → MLX weight converter (executable) |
| `birefnet-smoke` | real-forward gate over `BiRefNetPackage` (executable) |

## Consuming it

Public + version-tagged on github.com/xocialize. Add by tagged URL:
`.package(url: "https://github.com/xocialize/mlx-birefnet-swift", from: "0.1.0")`, then import `MLXBiRefNet` (the conformant `matting` package). Builds standalone — its engine contract (`MLXToolKit`) and other dependencies are tagged-URL net deps, no local checkouts.

Requirements: macOS 26+ (Apple Silicon, Metal GPU).

## License

MIT throughout (port + wrapper). The `BiRefNet` core is vendored from [mnmly/mlx-swift-BiRefNet](https://github.com/mnmly/mlx-swift-BiRefNet) (MIT, Hiroaki Yamane); the original BiRefNet model is by [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet) (MIT). See `LICENSE` and `NOTICE` for attribution.
