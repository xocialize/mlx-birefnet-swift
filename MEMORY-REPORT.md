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

## Declaration + guard (SPLIT footprint, contract 1.14.0 — efficiency adoption 2026-06-30)

The open enhancement this report flagged **shipped** (engine 0.14.0 / contract 1.14.0:
`QuantFootprint.peakActivationBytes` + the engine's single shared transient reserve across residents).
Re-measured via `birefnet-smoke` through the real `MLXServeEngine` — peaks identical to the table above.
Both tiers are fp16, so `QuantFootprint` (keyed on quant) still can't carry per-mode figures; the package
declares the **fast (consumer) envelope as a SPLIT** and keeps the best runtime guard:

- **Declared `QuantFootprint(.fp16, residentBytes: 0.9 GB, peakActivationBytes: 4.4 GB)`** — persistent
  weights floor (both pipelines' ~424 MB; best builds lazily and co-resides once requested) + the fast
  transient activation peak (4,941 − 423 MB). Replaces the old flat 6.5 GB. Because the engine reserves
  **one** shared transient across all residents (serialized inference), BiRefNet now co-resides with the
  rest of the optimizer chain on the weights while sharing a single activation reserve — engine charge
  drops from **6.5 GB flat → ~0.9 GB resident** + a shared ~4.4 GB transient.
- **`BiRefNetConfiguration: QuantConfigured`** — so the governor charges the matching declared fp16
  footprint (not the largest-that-fits heuristic).
- **best's measured split (documented, NOT admitted):** residentBytes ~0.5 GB · peakActivationBytes
  ~17.9 GB (18,305 − 425 MB). best stays a **runtime-guarded** variant: mode is per-request, so admission
  (at load, before the mode is known) can't reserve its ~18 GB peak.
- **Runtime guard in `run(.best)`** refuses best (`insufficientMemoryForBest`) when
  `maxRecommendedWorkingSetSize < 19.5 GB`, rather than OOM mid-forward; callers can catch + fall back to
  fast. **Kept** — it's the device-capability check that lets "best when capable" work for the per-request
  mode.
- **P1b (deferred):** promoting mode to a registration-time `PackageID` axis (two packages each conforming
  to `FootprintConfigured`, so best is first-class admitted/evicted) is a coordinated change — the PROD
  consumer contract (`EngineMatteProvider`) relies on `req.mode` + the `insufficientMemoryForBest` fallback.
  See EFFICIENCY-ADOPTION.md outcome.
