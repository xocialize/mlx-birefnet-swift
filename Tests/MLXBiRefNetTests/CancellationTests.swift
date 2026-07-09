// CancellationTests.swift — BiRefNet through the engine's CAN gate (offline, no MLX kernels).
// CAN-1/2 drive the real run() pre-cancelled (the entry checkpoint fires before capability
// validation, weights, or the best-tier memory guard); CAN-3 is the document of record for the
// checkpoint cadence. BiRefNet is a CAN-3 judgment case: the forward is ONE monolithic MLX eval
// (single encoder-decoder lazy graph, ~0.5 s fast@1024 / ~2 s best@2048 — no denoise loop, no
// tiling), yet peakActivationBytes 14 GB ≥ 2 GB makes the manifest long-run implied. So the
// declared cadence names the REAL seams run() has, once per matted frame each:
//   • entry checkpoint (first act of run(), BiRefNetPackage.run)
//   • pre-forward, after pipeline build — a first-request best-tier weight download can
//     precede the forward (BiRefNetPackage.run, after `pipeline(best:)`)
//   • post-forward / pre-PNG-encode (BiRefNetPackage.run, after `maskCGImage()`)
// No per-step checkpoints are fabricated — there is no iterative loop to hang them on.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXBiRefNet

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation or weights are touched, so this is offline-safe.
        let package = BiRefNetPackage(configuration: BiRefNetConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: MattingRequest(image: Image(format: .png, data: Data())))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // peakActivationBytes 14 GB ≥ 2 GB ⇒ long-run implied — the sub-second exemption is
        // not available (and best@2048 genuinely runs ~2 s at a ~48 GB phys peak).
        XCTAssertTrue(CancellationConformance.longRunImplied(by: BiRefNetPackage.manifest))

        let report = CancellationConformance.checkCadence(
            manifest: BiRefNetPackage.manifest,
            posture: .cadence([
                // Pre-forward seam: once per frame, after pipeline build/possible weight
                // download, before committing to the monolithic encoder-decoder eval.
                .init(phase: .encode, unit: .frame),
                // Post-forward seam: once per frame, between the forward's eval and PNG encode.
                .init(phase: .postprocess, unit: .frame),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
