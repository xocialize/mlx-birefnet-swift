# mlx-birefnet-swift

The MLXEngine **`matting`** package over [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) — high-quality foreground matting (soft-alpha cutouts) on Apple Silicon.

Takes an `Image` and returns a single-channel `Matte` (0 = background … 1 = foreground) at source resolution — usable as a cutout source *and* as a reusable weight-map signal for other capabilities. A thin conformance wrapper (`MLXBiRefNet`) over a vendored Swift/MLX core (`BiRefNet`).

## Checkpoint families (PackageIDs)

Several checkpoints share this one core — the converted key sets are byte-identical (754 tensors in →
687 out), so a new checkpoint is a configuration + manifest, never a port. Each **family** registers
under its own PackageID so license, provenance and footprint stay separable; a request's `mode`
selects a tier *within* the registered family.

| PackageID | Family | Tier (`mode`) | Input | Weights | Peak (fp16) |
|---|---|---|---|---|---|
| `birefnet` | BiRefNet | `MattingContract.fast` (default) | 1024² | `mlx-community/BiRefNet-fp16` | ~4.9 GB |
| `birefnet` | BiRefNet | `MattingContract.best` | 2048² | `mlx-community/BiRefNet_HR-matting-fp16` | ~18.3 GB |
| `lucida-matting` | Lucida (BiRefNet_HR fine-tune) | *none — single tier* | 1024² | `mlx-community/Lucida-fp16` | ~4.9 GB |

Weights are fp16 (~420 MB each) and materialize on first use **by the engine** from each
configuration's declared `WeightSourcing` sources, into the model store — no manual URLs and no
package-local downloader (contract 1.24). `best` is a pro-tier footprint: the package declares the
1024 envelope and applies a **runtime guard** on the 2048 tier
(`BiRefNetError.insufficientMemoryForBest` when the device's recommended Metal working set is too
small), so a consumer adapter can fall back to `fast`.

### Which family for which image

Lucida trains and runs at 1024, so it costs the *fast* tier's memory while leading BiRefNet-HR
overall on the author's benchmark (MAE 0.0257 vs 0.0346). It is **not** a replacement — on that same
benchmark the split is:

| Better on Lucida | Better on BiRefNet-HR |
|---|---|
| camouflage (0.0270 vs 0.0752) · print/design (0.0235 vs 0.0544) · text & logos (0.0091 vs 0.0207) · illustration (0.0092 vs 0.0157) · transparency (0.0358 vs 0.0687) | hair (0.0048 vs 0.0093) · thin structures (0.0196 vs 0.0322) · complex edges (0.0385 vs 0.0484) |

So: photographic portrait/hair work → `birefnet`; non-photographic, hard-edge, text or illustration
work → `lucida-matting`.

> **Benchmark caveat.** Those numbers are the author's own — 203 self-selected images, 9 self-defined
> categories, MAE only (which flatters soft-alpha smoothness), and the published table names
> "Lucida-v7" while the repo ships one unversioned `model.safetensors`. Treat as directional. Also
> not ported: the author's colour-decontamination and edge-refinement passes live in their GitHub
> repo, not the checkpoint — this package emits the raw sigmoid alpha for every family.

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

Lucida is a separate PackageID, registered alongside — both can be resident and selected per request:

```swift
let lucidaID = try await engine.register(
    Lucida.registration, configuration: .lucida())

let cutout = try await engine.run(
    MattingRequest(image: logo), package: lucidaID) as! MattingResponse
```

`Lucida.registration` refuses a `.birefnet` configuration (and vice-versa the family's `resolved`
mapping ignores off-axis modes), so a registration can never publish one family's identity over
another's weights.

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
