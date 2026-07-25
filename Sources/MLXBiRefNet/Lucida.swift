// Lucida as an alternate `matting` PackageID on the BiRefNet core.
//
// Lucida is a fine-tune of ZhengPeng7/BiRefNet_HR with NO architecture change: the converted key
// set is byte-identical to both upstream bases (754 tensors in → 687 out, 0 keys/shapes differing),
// and the resolved arch config matches `BiRefNetConfig.swinLargeDefault` exactly (swin_v1_l,
// lateral [1536,768,384,192], mul_scl_ipt=cat, cxt_num=3, ASPPDeformable, BasicDecBlk_x1, BN
// present, refine=''). So there is no port and no new core — `BiRefNetPackage` runs the Lucida
// snapshot as-is via `BiRefNetConfiguration.variant == .lucida`. This registration carries only the
// distinct manifest (provenance + surface identity + license posture).
//
// WHY ITS OWN PackageID rather than a third `mode` on the birefnet surface: the manifest is
// per-registration, so license, provenance (`sourceRepo` is the engine's marker key) and footprint
// would otherwise be shared. Keeping them separable means a future license posture on Lucida cannot
// gate BiRefNet's PROD consumers (EngineMatteProvider, the TRELLIS.2 multi-view front door), and the
// storage panel credits the right repo. Same shape as FireRed-on-QwenImageEdit and Klein
// base-vs-distilled.
//
// TIER POSITIONING (measured, not claimed). Lucida trains and runs at 1024, so it lands in the FAST
// tier's memory envelope while the author's benchmark puts it ahead of BiRefNet-HR overall
// (MAE 0.0257 vs 0.0346). It is NOT a replacement: on that same benchmark BiRefNet-HR is better on
// hair (0.0048 vs 0.0093), thin structures (0.0196 vs 0.0322) and complex edges (0.0385 vs 0.0484),
// while Lucida leads on camouflage (0.0270 vs 0.0752), print/design (0.0235 vs 0.0544), text/logo
// (0.0091 vs 0.0207) and illustration (0.0092 vs 0.0157). Route photographic portrait/hair work to
// the birefnet PackageID and non-photographic/illustration work here.
//
// BENCHMARK CAVEAT: those figures are the author's own — 203 self-selected images, 9 self-defined
// categories, MAE only (which flatters soft-alpha smoothness), and the published table names
// "Lucida-v7" while the repo ships one unversioned model.safetensors. Directional, not decisive;
// promotion to a default tier wants our own A/B on real consumer input.
//
// NOT PORTED: the author's colour-decontamination and edge-refinement passes live in their GitHub
// repo, not the checkpoint. This package emits the raw sigmoid alpha, as it does for every tier, so
// the full upstream quality claim is not reproduced here.

import Foundation
import MLXToolKit
import BiRefNet

/// The Lucida checkpoint's engine identity. Register with `BiRefNetConfiguration.lucida()`.
public enum Lucida {

    public static var manifest: PackageManifest {
        PackageManifest(
            // C7 weights + C8 port. Upstream declares MIT for the weights and MIT for the code, so
            // MIT is what we declare — inventing a stricter label than the actual grant would be
            // just as wrong as ignoring one. PROVENANCE CAVEAT worth carrying: the model card says
            // the training data "mix[es] MIT-licensed and research-only datasets" (P3M-10k, COD10K,
            // DIS5K, plus the CC-BY-4.0 ToonOut set). That is a data-derivation question, not a
            // defect in the grant we received — but it is the same shape as the MobileWan C7 finding,
            // so a legal read is owed before this becomes a shipping default rather than an opt-in
            // tier. Flagged in the registry row too.
            license: LicenseDeclaration(weightLicense: .mit, portCodeLicense: .mit),
            // The engine's download/marker key → the Lucida weight repo, NOT BiRefNet's.
            provenance: Provenance(sourceRepo: "mlx-community/Lucida-fp16",
                                   revision: "main", tier: 2),
            requirements: RequirementsManifest(
                // SPLIT FOOTPRINT (contract 1.14.0), on the in-app **process phys_footprint** basis
                // the birefnet row was re-baselined to — that is what R-MEM-1 admits against and what
                // OOMs the process, and an MLX working-set figure under-reads it ~2.7×.
                //
                // Lucida runs the IDENTICAL graph at the IDENTICAL resolution as the birefnet `fast`
                // tier, so the fast tier's in-app measurement transfers directly. Confirmed rather
                // than assumed: `birefnet-smoke` reports byte-identical memory for both families
                // (floor 423 MB / MLX-peak 4941 MB), and that 4941 MB is the same MLX-peak (4.9 GB)
                // whose in-app phys equivalent was measured at floor 0.54 GB / peak 13.70 GB.
                //   • residentBytes 0.6 GB  = ONE 1024 graph's weights floor (measured phys 0.54 GB
                //     + margin). Lower than the birefnet package's 0.9 GB, which covers both tiers.
                //   • peakActivationBytes 14 GB = 13.70 peak − 0.54 floor ≈ 13.2 GB transient + margin.
                footprints: [QuantFootprint(quant: .fp16,
                                            residentBytes: 600_000_000,
                                            peakActivationBytes: 14_000_000_000)],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0))
            ),
            surfaces: [
                MattingContract.descriptor(
                    name: "lucida-matting",
                    summary: "Lucida foreground matting (BiRefNet_HR fine-tune) → single-channel "
                        + "soft-alpha matte @1024, its trained resolution (~0.5s). Tuned for "
                        + "non-photographic and hard-edge subjects: camouflage, transparency, "
                        + "text/logos, print design, illustration. For photographic hair and thin "
                        + "structures prefer the `birefnet` package.",
                    // No modes: one checkpoint at one resolution. `BiRefNetConfiguration.resolved`
                    // therefore ignores any mode a caller sends rather than reaching another
                    // family's weights.
                    modes: [])
            ])
    }

    /// Registration reusing the BiRefNet package type under the Lucida manifest — the
    /// multi-package-per-capability route (PackageID = "lucida-matting").
    public static var registration: PackageRegistration {
        PackageRegistration(manifest: manifest) { config in
            guard let typed = config as? BiRefNetConfiguration else {
                throw PackageError.configurationMismatch(
                    expected: String(describing: BiRefNetConfiguration.self),
                    got: String(describing: type(of: config)))
            }
            // Guard the pairing: this manifest's identity, footprint and marker repo all describe
            // the Lucida checkpoint, so registering it with a `.birefnet` configuration would
            // publish Lucida's identity over BiRefNet's weights.
            guard typed.variant == .lucida else {
                throw PackageError.configurationMismatch(
                    expected: "BiRefNetConfiguration(variant: .lucida)",
                    got: "BiRefNetConfiguration(variant: .\(typed.variant.rawValue))")
            }
            return BiRefNetPackage(configuration: typed)
        }
    }
}
