# BiRefNet `matting` ModelPackage — C0–C13 conformance review

Reviewed 2026-06-22 against `EngineeringDocs/MLXEngineDocs/conformance.md` (contract 1.5.0). The
`MLXServeConformance` harness is still a placeholder, so this is a manual reviewer pass; evidence cites
`Sources/MLXBiRefNet/{BiRefNetPackage,BiRefNetConfiguration}.swift`. Parity is **not** a conformance item
(owned by mlx-porting) — BiRefNet IoU 0.9905; correct mattes proven offline (smoke) + live (BRIDGE-017).

**Verdict: PASS (C0–C13).** Offline contract tests green; full engine path (register→run) live-verified.

| C | Item | Verdict | Evidence |
|---|---|---|---|
| **C0** | Contract version | ✅ | `PackageManifest(contractVersion:)` defaults to `ContractVersion.current` (1.5.0 — the version that introduced `.matting`). |
| **C1** | Capability registration | ✅ | One canonical capability `.matting`; one `manifest.surfaces` entry (`MattingContract.descriptor`), one model. No fusion. |
| **C2** | Canonical schema | ✅ | I/O = canonical `MattingRequest` (`image`, `preferredKind`, `mode`) → `MattingResponse(matte:)`; output artifact = `.matte`. Uses the contract's own types verbatim, not a bespoke shape. |
| **C3** | Canonical artifact I/O | ✅ | Input `Image` (PNG bytes) → output `Matte` (PNG bytes), both serialized round-trip; no live-tensor/IOSurface fork. |
| **C4** | Mode-as-parameter | ✅ | `fast`/`best` are `MattingContract.fast/.best` **Modes** on `req.mode`; `run()` dispatches on them. NOT separate surfaces. |
| **C5** | metaData hygiene | ✅ | `metaData` unused. The quality tier is a **canonical Mode**, not smuggled — strong pass. |
| **C6** | Specialty declaration | ✅ | `specialties: []` — none claimed, none misused (no governed-vocab term applies to a sole matting model). |
| **C7** | Weight license gate | ✅ | `weightLicense: .mit` (ZhengPeng7/BiRefNet MIT) — permissive; admitted live under `.permissiveOnly`. |
| **C8** | Port-code license gate | ✅ | `portCodeLicense: .mit` (vendored mnmly core MIT + wrapper) — permissive, consistent with the weight layer. |
| **C9** | PackageConfiguration | ✅ | `BiRefNetConfiguration: PackageConfiguration, ModelStorable` (Codable+Sendable); holds session-stable values (repos/quant/weight URLs/store root). Per-request `mode`/`image` ride the request, not config. |
| **C10** | Requirements manifest | ✅* | `footprints: [.fp16 6.5 GB]` (empirically measured — see MEMORY-REPORT.md), `requiredBackends: [.metalGPU]`, `minMacOS 26`. *Nuance: one fp16 footprint can't express fast(6 GB)/best(18 GB); declared the fast envelope + a runtime guard refuses best on devices that can't hold it (per-mode footprint = flagged engine enhancement). |
| **C11** | MCPBridge introspection | ✅ | Surface schema is the canonical `MattingContract.descriptor(...)` (`ToolDescriptor` + params + `supportedModes`) — introspectable, no reverse-engineering. |
| **C12** | Forward-compat discipline | ✅ | No closed-`Capability` switch; `run()` uses `request.capability == .matting` else-throws (a new case just routes to the error path). |
| **C13** | Runtime governance | ✅* | Engine-constructed via `PackageRegistration.of` + `nonisolated init` (no compute in init); `@InferenceActor`-isolated `load/run/unload` (compiler-enforced); `unload()` nils both pipelines (working set freed); no private queue. *Nuance: cancellation is checked at the forward boundaries (`Task.checkCancellation` before + after the forward) — a single matting forward is atomic (no intra-kernel token/frame boundary to preempt), unlike an autoregressive surface. |

## Open items (not blockers; tracked)

- **Per-mode footprint** (C10 nuance): config-aware/per-mode footprints so a variant package declares
  fast+best separately — fed to MLXEngine ENHANCEMENTS §3.1 (BiRefNet is the "affordable-tier-blocked" case).
- **Best-tier memory**: best@2048 ~18 GB is a 32 GB+ feature; guarded at runtime, adapter falls back to fast.
