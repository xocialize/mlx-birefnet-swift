import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXToolKit
import MLXServeCore
import MLXBiRefNet

// birefnet-smoke <image> <fast.safetensors> <best.safetensors> <out.png> [fast|best]
// Drives the package through the REAL MLXServeEngine (register → run) — proving the full Stage-2 path:
// license-gate admission (MIT/MIT), device eligibility (C10), engine-constructs-the-package (C13), and the
// matte forward. Reports timing + peak GPU memory + matte gray-stats so a uniform (silent) matte is caught.

enum SmokeError: Error { case usage, badImage, badResponse }

func encodePNG(_ cg: CGImage) throws -> Data {
    let data = NSMutableData()
    guard let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil)
    else { throw SmokeError.badImage }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { throw SmokeError.badImage }
    return data as Data
}

/// Decode a grayscale PNG matte and report normalized mean/min/max — a near-uniform result is the
/// classic "looks fine, is wrong" silent failure.
func grayStats(_ png: Data) -> (mean: Double, min: Double, max: Double) {
    guard let src = CGImageSourceCreateWithData(png as CFData, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return (0, 0, 0) }
    let w = cg.width, h = cg.height
    var buf = [UInt8](repeating: 0, count: w * h)
    let cs = CGColorSpaceCreateDeviceGray()
    guard let ctx = CGContext(data: &buf, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w,
                              space: cs, bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return (0, 0, 0) }
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    var sum = 0.0, lo = 255, hi = 0
    for v in buf { sum += Double(v); lo = min(lo, Int(v)); hi = max(hi, Int(v)) }
    return (sum / Double(buf.count) / 255.0, Double(lo) / 255.0, Double(hi) / 255.0)
}

@main
struct Smoke {
    static func main() async {
        let a = CommandLine.arguments
        guard a.count >= 5 else {
            FileHandle.standardError.write(Data(
                "usage: birefnet-smoke <image> <fast.safetensors> <best.safetensors> <out.png> [fast|best]\n".utf8))
            exit(2)
        }
        let imagePath = a[1], fastW = a[2], bestW = a[3], outPath = a[4]
        let quality = a.count > 5 ? a[5] : "fast"
        do {
            guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: imagePath) as CFURL, nil),
                  let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw SmokeError.badImage }
            let image = Image(format: .png, data: try encodePNG(cg), width: cg.width, height: cg.height)

            // "hub" sentinel → nil override → exercise the live HubApi download (post-upload verify).
            let config = BiRefNetConfiguration(
                fastWeightsURL: fastW == "hub" ? nil : URL(fileURLWithPath: fastW),
                bestWeightsURL: bestW == "hub" ? nil : URL(fileURLWithPath: bestW))

            // Full engine path: register (license gate + C10 eligibility) → run (engine constructs/loads
            // the package, C13; first run lazily loads weights).
            let engine = MLXServeEngine()
            await engine.useModelStore(ModelStore(root: nil))
            let t0 = Date()
            let id = try await engine.register(BiRefNetPackage.registration, configuration: config)
            let tReg = Date().timeIntervalSince(t0)

            let mode: Mode = quality == "best" ? MattingContract.best : MattingContract.fast
            let t1 = Date()
            let resp = try await engine.run(MattingRequest(image: image, preferredKind: .softAlpha, mode: mode),
                                            package: id)
            let tRun = Date().timeIntervalSince(t1)
            let tLoad = tReg   // register is the cheap gate; weights load inside the first run

            guard let matte = (resp as? MattingResponse)?.matte else { throw SmokeError.badResponse }
            try matte.data.write(to: URL(fileURLWithPath: outPath))

            let s = grayStats(matte.data)
            let peakMB = Double(MLX.GPU.snapshot().peakMemory) / 1_048_576
            print(String(format: "OK %@ → %@ | %dx%d kind=%@ | reg %.2fs run %.2fs | peakGPU %.0f MB | "
                + "gray mean %.3f [%.3f…%.3f]", quality, outPath, matte.width ?? -1, matte.height ?? -1,
                matte.kind.rawValue, tLoad, tRun, peakMB, s.mean, s.min, s.max))

            // --- Memory report (this variant; one process = a clean peak). Methodology per the
            // memory-harness: peak active during the real forward + resident floor (active after freeing
            // activations) + the device's recommendedWorkingSet ceiling. Recommend = peak×1.2 + 256 MB.
            MLX.GPU.clearCache()
            let floorMB = Double(MLX.GPU.snapshot().activeMemory) / 1_048_576
            let workingSetMB = Double(MLX.GPU.deviceInfo().maxRecommendedWorkingSetSize) / 1_048_576
            let recommendMB = peakMB * 1.2 + 256
            print(String(format: "MEM %@ | floor %.0f MB · peak %.0f MB · recommend %.0f MB (×1.2+256) | "
                + "workingSet %.0f MB · exceeds %@", quality, floorMB, peakMB, recommendMB, workingSetMB,
                recommendMB > workingSetMB ? "YES" : "no"))
            if s.max - s.min < 0.02 {
                FileHandle.standardError.write(Data(
                    "WARN: matte near-uniform (mean \(s.mean)) — possible silent failure\n".utf8))
                exit(3)
            }
        } catch {
            FileHandle.standardError.write(Data("FAILED: \(error)\n".utf8))
            exit(1)
        }
    }
}
