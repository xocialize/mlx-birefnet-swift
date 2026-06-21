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

    func testConfigurationDefaults() {
        let c = BiRefNetConfiguration()
        XCTAssertEqual(c.weightsFile, "model.safetensors")
        XCTAssertEqual(c.quant, .fp16)
        XCTAssertFalse(c.fastRepo.isEmpty)
        XCTAssertFalse(c.bestRepo.isEmpty)
    }
}
