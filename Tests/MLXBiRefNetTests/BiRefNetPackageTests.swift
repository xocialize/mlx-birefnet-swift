import XCTest
import MLXToolKit
@testable import MLXBiRefNet

/// Contract-level checks for the BiRefNet `matting` package — no kernel runs here (the first real forward
/// happens in the app/CLI harness). Asserts the manifest is well-formed + license-admissible + the surface
/// declares the two quality modes.
final class BiRefNetPackageTests: XCTestCase {
    func testManifestShape() {
        let m = BiRefNetPackage.manifest
        XCTAssertEqual(m.license.weightLicense, .mit)
        XCTAssertEqual(m.license.portCodeLicense, .mit)
        XCTAssertTrue(m.requirements.requiredBackends.contains(.metalGPU))
        XCTAssertEqual(m.surfaces.count, 1)
        let surface = m.surfaces[0]
        XCTAssertEqual(surface.capability, .matting)
        XCTAssertTrue(surface.supportedModes.contains(MattingContract.fast))
        XCTAssertTrue(surface.supportedModes.contains(MattingContract.best))
    }

    /// Split footprint (contract 1.14.0): the declared fp16 footprint carries BOTH halves — a non-zero
    /// transient activation peak (so the engine can reserve the shared transient) and a resident floor far
    /// below the old flat 6.5 GB. Regression guard for the efficiency adoption (P1a).
    func testSplitFootprintDeclared() {
        let m = BiRefNetPackage.manifest
        let fp = m.requirements.footprints.first { $0.quant == .fp16 }
        XCTAssertNotNil(fp, "fp16 footprint must be declared")
        XCTAssertGreaterThan(fp!.peakActivationBytes, 0, "transient must be declared so the engine can reserve it")
        XCTAssertLessThan(fp!.residentBytes, 6_500_000_000, "resident floor must drop below the old flat charge")
    }

    /// `QuantConfigured` opt-in so the governor charges the matching declared `QuantFootprint`.
    func testQuantConfigured() {
        XCTAssertEqual((BiRefNetConfiguration() as? QuantConfigured)?.quant, .fp16)
    }

    func testConfigurationDefaults() {
        let c = BiRefNetConfiguration()
        XCTAssertEqual(c.weightsFile, "model.safetensors")
        XCTAssertEqual(c.quant, .fp16)
        XCTAssertFalse(c.fastRepo.isEmpty)
        XCTAssertFalse(c.bestRepo.isEmpty)
    }
}
