import Foundation
import MLXToolKit

/// Configuration for the BiRefNet-family `matting` packages.
///
/// One configuration type backs several CHECKPOINT FAMILIES, selected by ``Variant``. Every family
/// loads through the identical Swin-L + ASPP-Deformable graph — the converted key sets are
/// byte-identical (754 tensors in, 687 out, verified against both upstream bases) — so a new
/// checkpoint is a configuration + manifest, never a port. Weights resolve under the engine's model
/// store (`modelsRootDirectory` + repo dir + `weightsFile`), and the engine materializes anything
/// missing before `load()` (contract 1.24).
///
/// - ``Variant/birefnet``: the two-tier original — `MattingContract.fast` = general @1024 (~0.5 s) ·
///   `MattingContract.best` = HR-matting @2048 (~2 s).
/// - ``Variant/lucida``: a BiRefNet_HR fine-tune, single tier @1024 (its trained resolution),
///   registered under its own PackageID so license, provenance and footprint stay separable.
public struct BiRefNetConfiguration: PackageConfiguration, ModelStorable {

    /// Which checkpoint family this configuration serves.
    ///
    /// Family is fixed at REGISTRATION (one PackageID per family). A request's `mode` selects a
    /// quality TIER **within** the family — it must never be able to reach another family's
    /// weights, or a caller asking `lucida-matting` for `best` would silently be served BiRefNet's
    /// HR checkpoint under Lucida's identity. ``resolved(mode:)`` is the single place that mapping
    /// happens, so it cannot drift.
    public enum Variant: String, Codable, Sendable, CaseIterable {
        case birefnet
        case lucida
    }

    /// Checkpoint family. Defaults to `.birefnet` — also the value assumed when decoding a
    /// configuration serialized before this axis existed (see `init(from:)`).
    public var variant: Variant
    /// HF repo id for the converted MLX `.safetensors` backing the `fast` tier (general @1024).
    public var fastRepo: String
    /// HF repo id for the converted MLX `.safetensors` backing the `best` tier (HR-matting @2048).
    public var bestRepo: String
    /// HF repo id for the Lucida checkpoint (single tier @1024).
    public var lucidaRepo: String
    /// Weights filename within each repo's local snapshot directory.
    public var weightsFile: String
    /// Inference dtype — also the declared footprint quant. fp16 is the validated runtime, and for
    /// Lucida it is also the *best* one: measured against the PyTorch oracle at 1024², fp16 scores
    /// sigmoid cos 0.999995 while bf16 fails outright at 0.9958 (8-bit mantissa too coarse for this
    /// graph), which is why no bf16 Lucida variant is published.
    public var quant: Quant
    /// Engine-injected model-store root (download dir + security-scoped bookmark).
    public var modelsRootDirectory: URL?
    /// Direct paths to weights, bypassing model-store resolution. For pre-resolved callers +
    /// dev/CLI smoke; nil = resolve under `modelsRootDirectory` + the variant's repo.
    public var fastWeightsURL: URL?
    public var bestWeightsURL: URL?
    public var lucidaWeightsURL: URL?

    public init(variant: Variant = .birefnet,
                fastRepo: String = "mlx-community/BiRefNet-fp16",
                bestRepo: String = "mlx-community/BiRefNet_HR-matting-fp16",
                lucidaRepo: String = "mlx-community/Lucida-fp16",
                weightsFile: String = "model.safetensors",
                quant: Quant = .fp16,
                modelsRootDirectory: URL? = nil,
                fastWeightsURL: URL? = nil,
                bestWeightsURL: URL? = nil,
                lucidaWeightsURL: URL? = nil) {
        self.variant = variant
        self.fastRepo = fastRepo
        self.bestRepo = bestRepo
        self.lucidaRepo = lucidaRepo
        self.weightsFile = weightsFile
        self.quant = quant
        self.modelsRootDirectory = modelsRootDirectory
        self.fastWeightsURL = fastWeightsURL
        self.bestWeightsURL = bestWeightsURL
        self.lucidaWeightsURL = lucidaWeightsURL
    }

    /// Convenience for the Lucida family — pair with `Lucida.registration`.
    public static func lucida(quant: Quant = .fp16,
                              modelsRootDirectory: URL? = nil,
                              weightsURL: URL? = nil) -> BiRefNetConfiguration {
        BiRefNetConfiguration(variant: .lucida, quant: quant,
                              modelsRootDirectory: modelsRootDirectory,
                              lucidaWeightsURL: weightsURL)
    }

    // MARK: - Checkpoint resolution

    /// A resolved checkpoint: which repo, which explicit override (if any), and the resolution the
    /// model runs at.
    public struct ResolvedCheckpoint: Sendable, Equatable {
        /// Stable role tag — doubles as the `WeightSource.role` and the tier label in logs.
        public let role: String
        public let repo: String
        public let override: URL?
        public let inputSize: Int
    }

    /// Map (variant, mode) → checkpoint. THE choke point for checkpoint identity: repo and
    /// resolution are decided together, so no caller can pair one family's weights with another's
    /// resolution.
    ///
    /// For `.lucida` the mode is deliberately IGNORED — the family ships one checkpoint trained at
    /// 1024, its descriptor advertises no modes, and the contract lets a package honor only the
    /// modes it advertises. Ignoring is the safe direction; falling through to `bestRepo` would
    /// serve BiRefNet's weights under Lucida's PackageID.
    public func resolved(mode: Mode?) -> ResolvedCheckpoint {
        switch variant {
        case .lucida:
            return ResolvedCheckpoint(role: "lucida", repo: lucidaRepo,
                                      override: lucidaWeightsURL, inputSize: 1024)
        case .birefnet:
            if mode == MattingContract.best {
                return ResolvedCheckpoint(role: "best", repo: bestRepo,
                                          override: bestWeightsURL, inputSize: 2048)
            }
            return ResolvedCheckpoint(role: "fast", repo: fastRepo,
                                      override: fastWeightsURL, inputSize: 1024)
        }
    }

    /// Every checkpoint this configuration can be asked to serve. First entry is the tier `load()`
    /// warms; the whole list is what a fresh machine must materialize.
    public var servableCheckpoints: [ResolvedCheckpoint] {
        switch variant {
        case .lucida:   return [resolved(mode: nil)]
        case .birefnet: return [resolved(mode: nil), resolved(mode: MattingContract.best)]
        }
    }

    // MARK: - Codable (decode-safe across the variant axis)

    private enum CodingKeys: String, CodingKey {
        case variant, fastRepo, bestRepo, lucidaRepo, weightsFile, quant
        case modelsRootDirectory, fastWeightsURL, bestWeightsURL, lucidaWeightsURL
    }

    /// Hand-written so configurations serialized BEFORE the variant/lucida keys existed still
    /// decode: every new key is optional-with-default. Synthesized `Codable` would have made
    /// `variant` required and broken any persisted PROD registration.
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = BiRefNetConfiguration()
        variant = try c.decodeIfPresent(Variant.self, forKey: .variant) ?? d.variant
        fastRepo = try c.decodeIfPresent(String.self, forKey: .fastRepo) ?? d.fastRepo
        bestRepo = try c.decodeIfPresent(String.self, forKey: .bestRepo) ?? d.bestRepo
        lucidaRepo = try c.decodeIfPresent(String.self, forKey: .lucidaRepo) ?? d.lucidaRepo
        weightsFile = try c.decodeIfPresent(String.self, forKey: .weightsFile) ?? d.weightsFile
        quant = try c.decodeIfPresent(Quant.self, forKey: .quant) ?? d.quant
        modelsRootDirectory = try c.decodeIfPresent(URL.self, forKey: .modelsRootDirectory)
        fastWeightsURL = try c.decodeIfPresent(URL.self, forKey: .fastWeightsURL)
        bestWeightsURL = try c.decodeIfPresent(URL.self, forKey: .bestWeightsURL)
        lucidaWeightsURL = try c.decodeIfPresent(URL.self, forKey: .lucidaWeightsURL)
    }
}

/// `QuantConfigured` (engine contract 1.14.0): the config already stores `var quant`, so an empty
/// conformance lets `MemoryGovernor` charge the matching declared `QuantFootprint` for the selected
/// quant instead of the largest-that-fits heuristic. (All tiers are fp16, so the quant match
/// resolves to the single declared fp16 footprint; per-mode activation is handled by the `run()`
/// best guard — see the P1b note in EFFICIENCY-ADOPTION.md.)
extension BiRefNetConfiguration: QuantConfigured {}

/// `WeightSourcing` (engine contract 1.19.0; engine-EXECUTED since 1.24.0). Declares what a fresh
/// machine needs, so first-run materialization is checkable offline (MAT-1..5) and the ENGINE — not
/// this package — performs the download before `load()`.
///
/// BEHAVIOR NOTE for `.birefnet`: both tiers are declared, so a fresh install now materializes the
/// 2048 `best` checkpoint up front (~+440 MB) instead of lazily on its first `best` request. That is
/// the honest declaration — the package *can* be asked for `best`, and with the package-local
/// downloader removed an undeclared tier would fail on a fresh store (the MSS-7 silent-degradation
/// class this retrofit exists to kill). Restoring laziness is exactly what P1b (mode → PackageID)
/// buys: as its own PackageID, `best`'s weights would only be declared when that package is
/// registered. `.lucida` declares one source and is unaffected.
extension BiRefNetConfiguration: WeightSourcing {

    public var weightSources: [WeightSource] {
        servableCheckpoints.map {
            WeightSource(role: $0.role, repo: $0.repo, matching: [weightsFile])
        }
    }

    /// Honors the explicit `*WeightsURL` escape hatches BEFORE probing the store (the documented
    /// MS-2 override shape) — a pre-resolved caller or CLI smoke run has no missing sources even
    /// against an empty store.
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let overridden = Set(servableCheckpoints.filter {
            guard let url = $0.override else { return false }
            return FileManager.default.fileExists(atPath: url.path)
        }.map(\.role))
        return defaultMissingWeightSources(storeRoot: storeRoot)
            .filter { !overridden.contains($0.role) }
    }
}
