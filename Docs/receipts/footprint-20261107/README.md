# BiRefNet footprints re-measured: fast 7.6 GB, best 21.8 GB (0.4.2, AB-A-0176)

2026-11-07, Mac Studio M5 Max 128 GB, macOS 27.2. Release `birefnet-smoke` through the real `MLXServeEngine`:
register, run, the engine constructs the package. Stack: mlx-swift 0.32.3 (the version Pixel Agency Editor, Desktop
and Gaea's forge plugin resolve), mlx-engine-swift 0.74.0. One fresh process per row. **Peak = the process's
`phys_footprint` lifetime high-water** (`/usr/bin/time -l`, "peak memory footprint"), the number R-MEM-1 and the
OS go by. Run script: `run.sh`; raw output in `logs/`.

## The ladder: the engine's default MLX pool cap (2.15 GB on this Mac)

| Input | fast (1024²) | best (2048²) |
|---|---|---|
| 1 MP (1152×864) | 7.35 GB | 21.68 GB |
| 12 MP (4000×3000) | 7.36 GB | 21.69 GB |
| 24 MP (5656×4242) | 7.32 GB | 20.11 GB |
| 48 MP (8000×6000) | 7.52 GB | 21.76 GB |
| 100 MP (11548×8660) | 7.57 GB | 21.79 GB |

- **The input size doesn't drive the peak.** The forward runs at the fixed 1024² or 2048² grid. The size-dependent
  work (decode, the mask's bilinear resize to source resolution, render, PNG encode) adds at most about 0.25 GB at
  100 MP.
- **MLX active peak:** 4,941 MB (fast) and 17,777–18,305 MB (best). The difference to phys is the capped buffer pool
  plus process overhead.
- **Every matte is non-uniform** (gray mean 0.094–0.098, range 0–1). Smoke's silent-failure check passed.

## Controls: what the pool policy does (48 MP input)

| Row | Pool cap | Peak phys |
|---|---|---|
| fast, engine default | 2.15 GB | 7.47 GB |
| fast, 16 GB Mac's cap (5% of 12.7 GB) | 0.64 GB | **6.22 GB** |
| fast, **unmanaged** | none | **14.58 GB** |
| best, 32 GB Mac's cap (5% of ~23.7 GB) | 1.19 GB | 19.91 GB |
| best, **unmanaged** | none | **47.04 GB** |

**The unmanaged rows reproduce the 2026-06-30 baseline** (13.70 / 47.79 GB). June's in-app measurement ran on an
uncapped pool. The engine has capped the pool by default since 0.21.0 (`GPUCacheConfiguration.automatic` =
min(2 GB, 5% of budget)). Engine-hosted runs (Pixel Agency Editor's ForgeCore, Desktop, Gaea) pay the capped
numbers, which is what PixelAgencyEditor measured (7.85 / 21.81 GB in-app, AB-R-0518 / AB-R-0519).

## Declared in 0.4.2

- **fast**, and Lucida (same graph and resolution; byte-identical MLX memory, per its manifest):
  - BiRefNet `residentBytes` 0.9 GB + `peakActivationBytes` **7.2 GB** (was 14 GB) = an 8.1 GB charge, ~7% over the
    worst row.
  - Lucida 0.6 + 7.2 GB.
  - Both now **admit on a 16 GB Mac**: the budget is 12.7 GB, and the charge was 14.9 GB before.
  - `testFastAdmitsOnSixteenGigabytesAndCoversTheMeasuredPeak` pins both bounds.
- **best**: `bestMinWorkingSet` **22 GB** (was 48 GB), just over the worst capped row (21.79 GB).
  - A 32 GB Mac (working set ≈ 25 GB) now runs Best.
  - A 24 GB Mac (19.07 GB, AB-R-0551) still refuses, and the caller falls back to fast.

**Caveat:** the declarations assume an engine-managed pool. A host that constructs the engine with
`GPUCacheConfiguration(limit: .unmanaged)`, or writes `MLX.Memory.cacheLimit` high after the engine does, pays the
unmanaged numbers.

**Not measured here:** a real 16 GB Mac. The 0.64 GB-cap row emulates its pool cap on this Mac, not its memory
pressure.

**About the v0.4.1 tag:** it was pushed by mistake on 1198aab, the same commit as v0.4.0. It carries none of this
change; these declarations shipped as **v0.4.2**.
