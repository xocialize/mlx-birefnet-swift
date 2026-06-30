# Efficiency Adoption Brief — `mlx-birefnet-swift` (BiRefNet, `matting`)

> **For a session-specific agent.** Self-contained: audit + tasks to adopt the MLXEngine
> library-efficiency contract (engine 1.14.0). Load the `mlx-swift-integration` skill and read
> `references/package-efficiency.md` (incl. "Gotchas & measurement" from the LTX run) +
> `references/memory-harness.md` first. Brief shape follows the LTX template. Audited 2026-06-30.

## Why this one matters
BiRefNet is the package that **motivated the 1.14 split feature.** Its own manifest comment says:
*"One fp16 footprint can't express both [modes] … the proper fix is a config/per-mode footprint (open
engine enhancement — flagged to feed back)."* That enhancement shipped (`FootprintConfigured`
`residentBytesHint`/`peakActivationBytesHint` + the serialized transient reserve). This adoption closes
that loop and is the showcase for the **per-mode** lever LTX didn't exercise.

## Package at a glance
- **Wrapper:** `BiRefNetPackage` (`Sources/MLXBiRefNet/`), core `BiRefNet`. Capability `matting`. **Single-component** (Swin-L + ASPP-Deformable in one forward) — *not* a multi-stage pipeline.
- **Two modes, per-request** (`req.mode`): `fast` = general @1024 (~0.5 s) · `best` = HR-matting @2048 (~2 s). One config (`fastRepo`+`bestRepo`); pipelines built lazily, `best` only on first best request.
- **Measured (MEMORY-REPORT.md):** weights resident only **~424 MB fp16** — the footprint is **~all activation**: fast peak **4.9 GB**, best peak **18.3 GB**. So this is a *pure activation/transient* story — exactly what the split was built for.
- **Home:** `mlxengine-image/PROD/` (note: the registry row says WIP — fix to PROD/✅).

## Engine dependency status
- `Package.swift` pins `mlx-engine-swift` **`from: "0.10.0"`**, resolved **0.10.0**. **P0 = `swift package update`** → 0.14.0 (the pin already admits it). No manifest edit for the dep. If a consuming app workspace also references it, bump that `Package.resolved` too (the two-repo trap from the LTX run).

## Audit vs. the four levers

| Lever | State | Finding | Priority |
|---|---|---|---|
| Engine dep | 🟡 | resolves 0.10.0; re-resolve to 0.14.0 | **P0** |
| 1. Split footprint | ❌ | single flat `QuantFootprint(.fp16, 6.5 GB)` + a runtime guard for best; the "open enhancement" workaround | **P1** |
| 2. mmap/lazy load | 🟢 | weights tiny (~424 MB) via `fromPretrained`; eager-cast cost negligible here | note only |
| 3. Per-stage evict | ➖ | single-component — no multi-stage pipeline to stage (minor: `best` pipeline never evicted once built) | P3 (optional) |
| 4. BudgetAware | ➖ | fp16 is the validated runtime; no real memory/quality dtype lever | defer |

---

## P1 — Declare the split (the headline; choose a path)

Weights are ~424 MB and the rest is activation, so the split is dramatic. **Conform the config to
`QuantConfigured`** first (trivial — it already has `quant`; one extension), so the engine charges the
selected quant. Then choose how to express the two modes:

### P1a — Split the fast envelope, keep the best runtime guard  (recommended, low-risk)
Mode is **per-request**, so admission (at load, before mode is known) can't reserve best's 18.3 GB.
Declare the **fast/consumer envelope as a split**, keep the existing best runtime guard:
- `QuantFootprint(.fp16, residentBytes: ~0.9 GB /* both pipelines' weights */, peakActivationBytes: ~4.4 GB /* fast peak − weights */)`.
- Net: persistent **~0.9 GB** + transient **~4.4 GB** replaces the flat 6.5 GB — and because the engine
  now reserves **one shared transient** across residents, BiRefNet co-resides with the rest of the
  optimizer chain (IQA + restore + upscale) without each baking in its own activation. This is the
  direct win for the optimizer family.
- `best` stays guarded by `bestMinWorkingSet` (≈19.5 GB) in `run()` — correct, since the engine can't
  know at load that a best request is coming. Keep it.

### P1b — Promote mode to config/PackageID for full per-mode footprints  (design upgrade; decide with owner)
The "proper" fix the original comment wanted. Make mode a **registration-time** axis: register
`birefnet-fast` and `birefnet-best` as two `PackageID`s, each config conforming to **`FootprintConfigured`**:
- fast → `residentBytesHint ~0.5 GB`, `peakActivationBytesHint ~4.4 GB`
- best → `residentBytesHint ~0.5 GB`, `peakActivationBytesHint ~17.9 GB`
The engine then admits/evicts **best correctly** (first-class, not runtime-guarded), and the app routes
via `PackageID`. Cost: changes the mode model from per-request to per-registration — the matting UX must
pick a backer instead of passing `req.mode`. **Decision for the package owner:** keep the simple
per-request mode (P1a) or make best a first-class admitted variant (P1b). P1a is the safe default;
recommend P1b only if a real consumer needs best admitted on a constrained tier.

**Measure first (light — this is image, ~0.5–2 s runs, no GPU-watchdog risk):** re-run the memory
harness for the clean weights floor (step 1) and fast/best peaks (step 2) to confirm the ~0.9 / ~4.4 /
~17.9 numbers above; MEMORY-REPORT.md already has the peaks. Keep the provenance comments.

## P3 (optional) — evict the `best` pipeline after use
`best` is built lazily and never dropped (`unload()` clears both, but a long-lived resident keeps both
pipelines' ~0.9 GB after one best run). On tight devices, drop `best` after its run + `clearCache()`.
Marginal (weights are tiny); do only if a memory-sensitive consumer asks.

## Defer — P2 (n/a, single-component), P4 (BudgetAware: fp16 is the runtime, no lever).

## Already good (don't regress)
- Native HF downloader → engine model store (`modelsRootDirectory`) + `WeightDownloadProgress` forwarding.
- `ModelStorable`; lazy best-tier build; cancellation honored; the best runtime guard (`insufficientMemoryForBest`) — keep it under P1a.

## Definition of done
- [ ] `swift package update` → engine 0.14.0 (+ any consuming app workspace `Package.resolved`).
- [ ] Config conforms to `QuantConfigured` (trivial; it has `quant`).
- [ ] Split declared (P1a) — `residentBytes` (weights) + `peakActivationBytes` (fast activation), re-measured; best runtime guard kept. (Or P1b if the owner chooses mode-as-PackageID.)
- [ ] Parity/smoke gates green; in-app validation in the **image** testing app (fast + best), recording the split + peak. Confirm the engine charge drops from 6.5 GB → ~0.9 GB resident.
- [ ] BudgetAware deferred (note); P3 deferred unless requested.
- [ ] Update the BiRefNet row in `mlx-engine-swift/docs/model-registry.md`: Home → image/PROD, Avail ✅, Eff ✅ (or 🔵 if P1b is left open), Eng 0.14.0.

## Validation + reporting (this run also surveys the IMAGE testing app)
Validate in the image-category testing app and, as with LTX, **report the testing-app gaps** against the
per-category harness checklist (split readout, `transientReserveBytes` row, admissibility/tier seam,
phase-tagged trace, model-store grant, headless autorun) — see the `mlxengine-implementation` skill,
topic 7. BiRefNet is ideal for exercising the **admissibility/tier seam** (does best fit a 16 GB tier?)
and the **per-mode** display. Report effort (P1a is small; P1b is the variable) + the gap list.

---

## Outcome (executed 2026-06-30 — P1a done)

**P0** — `swift package update mlx-engine-swift` → resolved **0.10.0 → 0.14.0** (the `from: "0.10.0"` pin
admitted it; no manifest edit). Package builds clean against the 1.14 contract.

**QuantConfigured** — `BiRefNetConfiguration` now conforms via an empty extension (it already stored
`var quant`), so the governor charges the matching declared fp16 footprint.

**P1a — split declared.** Re-measured via `birefnet-smoke` through the real `MLXServeEngine` at the
documented envelope (`hard_fur_dog.jpg`, 2048×2731 / 5.6 MP, fp16) — peaks identical to MEMORY-REPORT:
- fast : floor **423 MB** · peak **4,941 MB** · valid matte (gray mean 0.387, full 0…1 range)
- best : floor **425 MB** · peak **18,305 MB** · valid matte (gray mean 0.388, full 0…1 range)

Declared `QuantFootprint(.fp16, residentBytes: 900 MB, peakActivationBytes: 4.4 GB)` (fast peak − floor).
**Engine charge: 6.5 GB flat → ~0.9 GB resident + a shared ~4.4 GB transient.** best's measured split
(resident ~0.5 GB, peakActivation ~17.9 GB) is documented in a manifest comment but NOT admitted; the
`insufficientMemoryForBest` runtime guard (`bestMinWorkingSet` 19.5 GB) is **kept** — it's the device
check that makes "best when capable" work for the per-request mode. Regression tests added
(`testSplitFootprintDeclared`, `testQuantConfigured`); all CLI gates green.

**P3 / P4 — deferred.** P3 (evict best after use) skipped (weights tiny; marginal). P4 (BudgetAware)
deferred — fp16 is the validated runtime, no real dtype/quality lever to trade.

### P1b — recommended follow-up (deferred; coordinated change for later)

P1b would promote mode from a **per-request** axis (`req.mode`) to a **registration-time** `PackageID`
axis: register `birefnet-fast` and `birefnet-best` as two PackageIDs, each config conforming to
`FootprintConfigured` (fast: resident ~0.5 GB / peakActivation ~4.4 GB; best: resident ~0.5 GB /
peakActivation ~17.9 GB). The engine would then **admit/evict best first-class** (not runtime-guarded),
and the matting UX would route by PackageID.

**Why deferred:** it's a breaking change to the consumer contract. The PROD consumer (`EngineMatteProvider`)
relies on passing `req.mode` and catching `insufficientMemoryForBest` to fall back to fast — a per-request
model. Switching to mode-as-PackageID means the UX must *pick a backer at registration* instead of passing
a mode per request, and the fallback path changes from a catch to a re-route. That coordination (engine +
the matting consumer + UX) is out of scope for the per-package efficiency sweep. P1a (split the fast
envelope, keep the best guard) captures the co-residency win with zero contract churn; recommend P1b only
when a real consumer needs best **admitted** on a constrained tier (the admissibility/tier seam below).

### Image testing-app survey

See the run report. BiRefNet is the ideal package for the **admissibility/tier seam** ("does best fit a
16 GB tier?") because of its 3.6× per-mode activation gap — whether the image testing app can express that
seam (and the split readout / transientReserve row / per-mode display) is captured in the gap list.
