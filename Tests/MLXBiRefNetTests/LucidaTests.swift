// LucidaTests.swift — the Lucida sibling checkpoint's contract surface, offline (no MLX kernels).
//
// Covers what a checkpoint-swap registration can get wrong without ever failing to build:
//   • the MAT gate (MAT-1..5) for BOTH families, now that the package-local downloader is gone and
//     the engine executes materialization from the declaration;
//   • Codable back-compat — a configuration serialized before the `variant` axis must still decode;
//   • the (variant, mode) → checkpoint mapping, including the cross-family guard that a `best`
//     request on the Lucida PackageID cannot reach BiRefNet's HR weights;
//   • manifest identity: distinct surface, distinct provenance marker repo, no advertised modes;
//   • CAN-1..3 on the new PackageID.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXBiRefNet

final class LucidaTests: XCTestCase {

    // MARK: - Helpers

    /// A configuration whose every servable checkpoint has an existing explicit weights file, so
    /// `missingWeightSources` must report nothing even with no store — the MAT-5 "satisfied" side.
    private func satisfied(_ variant: BiRefNetConfiguration.Variant) throws -> BiRefNetConfiguration {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("birefnet-mat-\(variant.rawValue)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        func stub(_ name: String) throws -> URL {
            let url = dir.appendingPathComponent(name)
            try Data("stub".utf8).write(to: url)
            return url
        }
        switch variant {
        case .lucida:
            return BiRefNetConfiguration(variant: .lucida,
                                         lucidaWeightsURL: try stub("lucida.safetensors"))
        case .birefnet:
            return BiRefNetConfiguration(variant: .birefnet,
                                         fastWeightsURL: try stub("fast.safetensors"),
                                         bestWeightsURL: try stub("best.safetensors"))
        }
    }

    // MARK: - MAT-1..5 (engine-executed materialization, contract 1.24.0)

    func testMATGateLucida() throws {
        let report = MaterializationConformance.check(
            freshConfiguration: BiRefNetConfiguration.lucida(),
            satisfiedConfiguration: try satisfied(.lucida))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testMATGateBiRefNet() throws {
        let report = MaterializationConformance.check(
            freshConfiguration: BiRefNetConfiguration(),
            satisfiedConfiguration: try satisfied(.birefnet))
        XCTAssertTrue(report.passed, report.summary)
    }

    /// Declared sources, per family. Lucida is one source; BiRefNet declares BOTH tiers — with the
    /// package-local downloader removed, an undeclared tier would simply fail on a fresh store
    /// instead of being fetched (the silent-degradation class the retrofit removes).
    func testWeightSourceDeclarations() {
        let lucida = BiRefNetConfiguration.lucida()
        XCTAssertEqual(lucida.weightSources.map(\.role), ["lucida"])
        XCTAssertEqual(lucida.weightSources.first?.repo, "mlx-community/Lucida-fp16")
        XCTAssertEqual(lucida.weightSources.first?.matching, ["model.safetensors"])

        let birefnet = BiRefNetConfiguration()
        XCTAssertEqual(birefnet.weightSources.map(\.role), ["fast", "best"])
        XCTAssertEqual(birefnet.weightSources.map(\.repo),
                       ["mlx-community/BiRefNet-fp16", "mlx-community/BiRefNet_HR-matting-fp16"])
        // No family may declare another family's weights.
        XCTAssertFalse(birefnet.weightSources.contains { $0.repo.contains("Lucida") })
        XCTAssertFalse(lucida.weightSources.contains { $0.repo.contains("BiRefNet") })
    }

    /// Fresh-machine posture (MAT-4): no store, no overrides ⇒ everything missing.
    func testMissingSourcesOnFreshMachine() {
        XCTAssertEqual(BiRefNetConfiguration.lucida().missingWeightSources(storeRoot: nil).count, 1)
        XCTAssertEqual(BiRefNetConfiguration().missingWeightSources(storeRoot: nil).count, 2)
    }

    /// Explicit override paths are honored before the store probe, so a pre-resolved caller or CLI
    /// smoke run reports nothing missing even against an empty store.
    func testOverridesSatisfyMissingSet() throws {
        for variant in BiRefNetConfiguration.Variant.allCases {
            let cfg = try satisfied(variant)
            XCTAssertTrue(cfg.missingWeightSources(storeRoot: nil).isEmpty,
                          "\(variant.rawValue): overrides should satisfy the missing set")
        }
    }

    // MARK: - Codable back-compat across the variant axis

    /// A configuration persisted BEFORE `variant`/`lucidaRepo` existed must still decode, defaulting
    /// to `.birefnet`. Synthesized Codable would have made `variant` required and broken any
    /// persisted PROD registration.
    func testPreVariantConfigurationStillDecodes() throws {
        let legacy = """
            {"fastRepo":"mlx-community/BiRefNet-fp16",
             "bestRepo":"mlx-community/BiRefNet_HR-matting-fp16",
             "weightsFile":"model.safetensors","quant":"fp16"}
            """.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(BiRefNetConfiguration.self, from: legacy)
        XCTAssertEqual(decoded.variant, .birefnet)
        XCTAssertEqual(decoded.fastRepo, "mlx-community/BiRefNet-fp16")
        XCTAssertEqual(decoded.lucidaRepo, "mlx-community/Lucida-fp16", "new key takes its default")
        XCTAssertEqual(decoded.quant, .fp16)
    }

    func testRoundTripPreservesVariant() throws {
        let cfg = BiRefNetConfiguration.lucida()
        let back = try JSONDecoder().decode(
            BiRefNetConfiguration.self, from: JSONEncoder().encode(cfg))
        XCTAssertEqual(back.variant, .lucida)
        XCTAssertEqual(back.resolved(mode: nil).repo, cfg.resolved(mode: nil).repo)
    }

    // MARK: - (variant, mode) → checkpoint, and the cross-family guard

    func testBiRefNetModeMapping() {
        let cfg = BiRefNetConfiguration()
        let fast = cfg.resolved(mode: MattingContract.fast)
        XCTAssertEqual(fast.role, "fast")
        XCTAssertEqual(fast.inputSize, 1024)
        XCTAssertEqual(fast.repo, cfg.fastRepo)

        let best = cfg.resolved(mode: MattingContract.best)
        XCTAssertEqual(best.role, "best")
        XCTAssertEqual(best.inputSize, 2048)
        XCTAssertEqual(best.repo, cfg.bestRepo)

        // No mode ⇒ the fast tier, unchanged from the pre-variant behaviour.
        XCTAssertEqual(cfg.resolved(mode: nil).role, "fast")
    }

    /// THE regression guard for the cross-checkpoint bug: Lucida ships one checkpoint at 1024 and
    /// advertises no modes, so ANY mode — including `best`, and an unknown tag — must resolve to
    /// Lucida's own weights at 1024. Falling through to `bestRepo` would serve BiRefNet's HR
    /// checkpoint under Lucida's PackageID, identity and license row.
    func testLucidaIgnoresModeAndNeverReachesBiRefNetWeights() {
        let cfg = BiRefNetConfiguration.lucida()
        for mode in [nil, MattingContract.fast, MattingContract.best, Mode("something-else")] {
            let r = cfg.resolved(mode: mode)
            XCTAssertEqual(r.role, "lucida", "mode \(mode.map(String.init(describing:)) ?? "nil")")
            XCTAssertEqual(r.inputSize, 1024, "mode \(mode.map(String.init(describing:)) ?? "nil")")
            XCTAssertEqual(r.repo, cfg.lucidaRepo, "mode \(mode.map(String.init(describing:)) ?? "nil")")
            XCTAssertNotEqual(r.repo, cfg.bestRepo)
            XCTAssertNotEqual(r.repo, cfg.fastRepo)
        }
        XCTAssertEqual(cfg.servableCheckpoints.count, 1)
    }

    func testBiRefNetServesBothTiers() {
        XCTAssertEqual(BiRefNetConfiguration().servableCheckpoints.map(\.role), ["fast", "best"])
    }

    // MARK: - Manifest identity

    func testLucidaManifestShape() {
        let m = Lucida.manifest
        XCTAssertEqual(m.license.weightLicense, .mit)
        XCTAssertEqual(m.license.portCodeLicense, .mit)
        XCTAssertTrue(m.requirements.requiredBackends.contains(.metalGPU))
        XCTAssertEqual(m.surfaces.count, 1)

        let surface = m.surfaces[0]
        XCTAssertEqual(surface.capability, .matting)
        XCTAssertEqual(surface.name, "lucida-matting")
        // One checkpoint, one resolution ⇒ advertises no tiers (see resolved(mode:)).
        XCTAssertTrue(surface.supportedModes.isEmpty)

        // Distinct engine identity from the birefnet package: the marker/download key must be
        // Lucida's repo, or the storage panel and provenance marker credit the wrong row.
        XCTAssertEqual(m.provenance.sourceRepo, "mlx-community/Lucida-fp16")
        XCTAssertNotEqual(m.provenance.sourceRepo, BiRefNetPackage.manifest.provenance.sourceRepo)
        XCTAssertNotEqual(surface.name, BiRefNetPackage.manifest.surfaces[0].name)
    }

    /// Split footprint (contract 1.14.0). Lucida runs at 1024 only, so its resident floor is ONE
    /// graph's weights — below the birefnet package's floor, which covers both tiers.
    func testLucidaSplitFootprint() {
        let fp = Lucida.manifest.requirements.footprints.first { $0.quant == .fp16 }
        XCTAssertNotNil(fp, "fp16 footprint must be declared")
        XCTAssertGreaterThan(fp!.peakActivationBytes, 0,
                             "transient must be declared so the engine can reserve it")
        let birefnetFloor = BiRefNetPackage.manifest.requirements.footprints
            .first { $0.quant == .fp16 }!.residentBytes
        XCTAssertLessThan(fp!.residentBytes, birefnetFloor,
                          "one 1024 graph must declare a smaller floor than the two-tier package")
    }

    /// bf16 must NOT be offered: measured against the PyTorch oracle at 1024², bf16 fails at
    /// sigmoid cos 0.9958 where fp16 reaches 0.999995.
    func testOnlyFP16Published() {
        XCTAssertEqual(Lucida.manifest.requirements.footprints.map(\.quant), [.fp16])
    }

    // MARK: - Registration pairing

    func testRegistrationBuildsPackageFromLucidaConfiguration() throws {
        let package = try Lucida.registration.makePackage(BiRefNetConfiguration.lucida())
        XCTAssertTrue(package is BiRefNetPackage)
    }

    /// Registering the Lucida manifest against a `.birefnet` configuration would publish Lucida's
    /// identity, footprint and marker repo over BiRefNet's weights — refuse it.
    func testRegistrationRejectsMismatchedVariant() {
        XCTAssertThrowsError(try Lucida.registration.makePackage(BiRefNetConfiguration())) { error in
            guard case PackageError.configurationMismatch = error else {
                return XCTFail("expected configurationMismatch, got \(error)")
            }
        }
    }

    func testRegistrationRejectsForeignConfigurationType() {
        struct Other: PackageConfiguration {}
        XCTAssertThrowsError(try Lucida.registration.makePackage(Other()))
    }

    // MARK: - CAN-1..3 on the new PackageID

    func testCANGatePreCancelledRunLucida() async {
        let package = BiRefNetPackage(configuration: .lucida())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: MattingRequest(image: Image(format: .png, data: Data())))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANCadenceDeclarationLucida() {
        XCTAssertTrue(CancellationConformance.longRunImplied(by: Lucida.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: Lucida.manifest,
            posture: .cadence([
                .init(phase: .encode, unit: .frame),
                .init(phase: .postprocess, unit: .frame),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
