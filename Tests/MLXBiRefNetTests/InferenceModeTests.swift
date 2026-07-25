// InferenceModeTests.swift — BiRefNet through the engine's INF gate (C14).
//
// This package is the gate's motivating defect (2026-07-25): every `BatchNorm2d` in
// `squeeze_module` / `ASPPDeformable` / the decoder ran on per-image batch statistics because
// `MLXNN.Module.training` defaults to `true`, so the checkpoint's `running_mean`/`running_var` were
// never read and were overwritten on every forward. The matte still looked like a matte; the PROD
// fast tier over-segmented by 68 % (foreground fraction 0.42 vs the PyTorch oracle's 0.25) and
// e2e logits cosine vs the oracle was 0.264. `BiRefNetPipeline.init` now calls `model.train(false)`
// at the single construction choke point, which took cosine to 0.99999.
//
// POSTURE OF RECORD: `.moduleGraph` — BiRefNet is an `MLXNN.Module` tree (Swin encoder is
// LayerNorm-only and was unaffected; the decoder's BatchNorms were not).
//
// What is offline here: the seam is wired to the REAL stored pipelines, and a package that has not
// been loaded reports an empty graph — which INF-1 must fail rather than pass vacuously. INF-1's
// green assertion needs weights, so it lives in the live gate lane (`birefnet-smoke`), not here.

import XCTest
import MLXNN
import MLXServeConformance
import MLXServeConformanceNN
import BiRefNet
@testable import MLXBiRefNet

// The conformance lives in the TEST target: `InferenceModeInspectable` is a test-facing seam, so
// the shipping target takes no dependency on the conformance library (the same placement the CAN
// and MAT gates use). No `@retroactive` needed — the test target and MLXBiRefNet are the same
// Swift package, so this is not a cross-package retroactive conformance.
extension BiRefNetPackage: InferenceModeInspectable {
    public func inferenceModeFlags() -> [InferenceModeConformance.ModuleTrainingFlag] {
        InferenceModeConformance.flags(of: inferenceModeGraphs)
    }
}

final class InferenceModeTests: XCTestCase {

    /// INF-2 — the posture declaration, and the proof that the seam reads real state: an
    /// unloaded package reports NO modules, which INF-1 fails. If this ever passes, the seam has
    /// stopped reaching the stored pipelines (or someone made it return a constant).
    func testUnloadedPackageFailsINF1() async {
        let pkg = BiRefNetPackage(configuration: BiRefNetConfiguration())
        let report = await InferenceModeConformance.check(pkg, posture: .moduleGraph)
        XCTAssertFalse(report.passed, "an unloaded package must not pass INF-1:\n\(report.summary)")
        XCTAssertTrue(report.summary.contains("no modules observed"), report.summary)
    }

    /// The choke point itself, exercised without weights: `BiRefNetPipeline.init` is what every
    /// load path funnels through (`fromPretrained`, `BiRefNet.segment`, direct init), and it must
    /// leave the graph in inference mode. Building the architecture is cheap — MLX arrays are lazy
    /// and nothing is evaluated here.
    func testPipelineInitIsTheInferenceModeChokePoint() {
        let model = BiRefNet(config: .swinLargeDefault)

        // Before: the MLXNN default that caused the defect.
        let before = InferenceModeConformance.check(
            flags: InferenceModeConformance.flags(of: model), posture: .moduleGraph)
        XCTAssertFalse(before.passed, "a fresh BiRefNet must be in training mode (MLXNN default)")
        XCTAssertTrue(before.summary.contains("(BatchNorm)"),
                      "the failure should name the running-statistic layers:\n\(before.summary)")

        // After: constructing the pipeline is the only thing that happens.
        _ = BiRefNetPipeline(model: model)
        let after = InferenceModeConformance.check(
            flags: InferenceModeConformance.flags(of: model), posture: .moduleGraph)
        XCTAssertTrue(after.passed, "BiRefNetPipeline.init must set inference mode:\n\(after.summary)")
    }

    /// The decoder BatchNorms are the ones that actually carried the defect — assert they exist and
    /// are reached by the walk, so a refactor that renames or re-nests them can't quietly drop them
    /// out of the gate's scope.
    func testDecoderBatchNormsAreInScope() {
        let flags = InferenceModeConformance.flags(of: BiRefNet(config: .swinLargeDefault))
        let batchNorms = flags.filter { InferenceModeConformance.isTrainingSensitive(type: $0.type) }
        XCTAssertFalse(batchNorms.isEmpty,
                       "BiRefNet must expose BatchNorms to the walk — if this is empty the gate is "
                       + "watching nothing")
        XCTAssertTrue(batchNorms.contains { $0.path.contains("decoder") },
                      "expected decoder BatchNorms; got \(batchNorms.prefix(5).map(\.path))")
    }
}
