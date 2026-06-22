# BiRefNet — memory report (C-memory conformance)

Methodology: memory-harness (peak **active** unified memory during the real forward, not weight size).
Measured via `birefnet-smoke` (one variant per process = a clean peak) → `MEM` line: resident floor
(`activeMemory` after `clearCache()`, ≈ weights), peak (`GPU.snapshot().peakMemory`), recommend
(`peak×1.2 + 256 MB`), vs the device `maxRecommendedWorkingSetSize`.

**Box:** Apple M-series Max, 128 GB unified (workingSet ~107 GB). **Input:** 2048×2731 (5.6 MP), fp16.
`peakActive` is largely machine-independent; `workingSet` is local.

| variant | model input | resident floor (weights) | peak active | recommend (×1.2+256) |
|---|---|---|---|---|
| `fast` (general)     | 1024² | 423 MB | **4,941 MB** | **6,185 MB** |
| `best` (HR-matting)  | 2048² | 425 MB | **18,305 MB** | **22,222 MB** |

## Findings

- **Footprint is ~all activation.** Weights resident is only ~424 MB (fp16) for both tiers; the peak is
  dominated by the Swin-L forward activations and scales with model input size (1024 vs 2048). Declaring
  weight size would under-reserve by ~10–40×.
- **best is a pro-tier (~32 GB+) feature.** Its ~18.3 GB peak needs a Metal working set ≥ ~19.5 GB; a 16 GB
  Mac (workingSet ~11 GB) can't hold it. fast (~6 GB) is consumer-viable.

## Declaration + guard (one fp16 footprint can't express both)

`MemoryGovernor` charges the largest declared footprint that fits the budget, and footprints key on **quant**
(both tiers are fp16) — so a single manifest can't carry per-mode footprints. Declaring best (22 GB) would
make even the 6 GB fast tier inadmissible on <32 GB Macs. Therefore:

- **Declared `QuantFootprint(.fp16, 6.5 GB)`** = the fast (consumer) envelope → fast admits broadly.
- **Runtime guard in `run(.best)`** refuses best (`insufficientMemoryForBest`) when
  `maxRecommendedWorkingSetSize < 19.5 GB`, rather than OOM mid-forward; callers can catch + fall back to fast.
- **Open engine enhancement (flagged to feed back):** config/per-mode footprints so a variant package can
  declare both tiers and the governor admits the selected one. Until then, this declare-fast + guard-best is
  the safe, product-viable compromise.
