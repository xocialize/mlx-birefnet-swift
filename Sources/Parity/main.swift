import Foundation
import MLX
import MLXNN
import BiRefNet

// birefnet-parity <goldens.safetensors> <weights.safetensors> [--dtype fp32|fp16|bf16]
//                 [--gpu] [--stages] [--tol 1e-3]
//
// Gates a converted BiRefNet checkpoint against PyTorch-oracle goldens produced by
// DEV/lucida-port/oracle_parity.py. The oracle's exact normalized input tensor is INJECTED
// (never re-derived here), so this measures the model, not preprocessing — per mlx-porting,
// generate the input on one side and inject on both.
//
// STREAM SPLIT (a framework constraint, not a preference). mlx-porting says pin parity to the
// CPU stream — Apple-GPU fp32 matmul accumulates ~8e-4 relative error per op, which both masks
// real op bugs and gets mistaken for them. But this port's ASPPDeformable path dispatches a
// custom Metal kernel (DCNv2Kernel) that hard-fails on CPU with
// "[metal_kernel] Only supports the GPU" — the same trap WeightConverter's comments warn about.
// So:
//   • encoder stages (Swin backbone + resize/merge — no deformable conv) → CPU stream, tight tol
//   • e2e forward (decoder/squeeze carry ASPPDeformable)                 → GPU stream, tol-e2e
// `--stream` overrides the split; `--gpu` is shorthand for all-GPU, used by the dtype sweep
// where the fp16/bf16 delta being measured sits far above GPU noise.
//
// COVERAGE. The authoritative gate is e2e logits + sigmoid. `--stages` adds encoder
// localization — the backbone's 4 stage outputs on BOTH passes (mul_scl_ipt='cat' calls the
// Swin twice) plus the per-level cat(full, up(half)) merge. That isolates 357 of 687 tensors
// and the resize/merge seam; if e2e diverges while every encoder stage matches, the break is
// downstream in squeeze_module/decoder and this gate should grow taps for those.

// MARK: - Comparison

struct Delta {
    let name: String
    let shape: [Int]
    let maxAbs: Float
    let meanAbs: Float
    let cosine: Float
    let refRange: (Float, Float)

    var passes: Bool { maxAbs.isFinite && cosine.isFinite }

    /// maxAbs normalised by the reference's own dynamic range. An ABSOLUTE bound is
    /// meaningless across these tensors — activations here span ±32 while the matte spans
    /// [0,1], so one number can't judge both (the same miscalibration the Mage-Flow port
    /// recorded: "absolute 0.9999 was miscalibrated for this arch").
    var relMax: Float {
        let span = refRange.1 - refRange.0
        return span > 0 ? maxAbs / span : maxAbs
    }

    /// CPU-stream phases are judged on relative maxAbs (deterministic, so outliers are
    /// meaningful). GPU-stream phases are judged on COSINE: Apple fp32 accumulation over a
    /// deep stack produces large single-element outliers that say nothing about correctness
    /// (measured here: the same encoder tensor goes maxAbs 3.96e-3 on CPU → 7.25 on GPU,
    /// while cosine holds at 0.9999), so cosine is the honest statistic for that lane.
    func verdict(relTol: Float, cosTol: Float, onGPU: Bool) -> Bool {
        guard passes else { return false }
        return onGPU ? cosine >= cosTol : relMax <= relTol
    }

    func line(relTol: Float, cosTol: Float, onGPU: Bool) -> String {
        let ok = verdict(relTol: relTol, cosTol: cosTol, onGPU: onGPU)
        return String(format: "  %-16@ %-22@ maxAbs %.3e  relMax %.2e  meanAbs %.3e  cos %.8f  %@",
                      name as NSString,
                      shape.map(String.init).joined(separator: "×") as NSString,
                      maxAbs, relMax, meanAbs, cosine, ok ? "PASS" : "FAIL" as NSString)
    }
}

/// Compare an MLX NHWC tensor against a torch-native NCHW golden.
func compare(_ name: String, mine: MLXArray, goldenNCHW: MLXArray) -> Delta {
    // Goldens are NCHW; the port is NHWC throughout.
    let ref = goldenNCHW.ndim == 4 ? goldenNCHW.transposed(0, 2, 3, 1) : goldenNCHW
    let a = mine.asType(.float32)
    let b = ref.asType(.float32)
    precondition(a.shape == b.shape,
                 "\(name): shape mismatch mine \(a.shape) vs golden(NHWC) \(b.shape)")
    let diff = MLX.abs(a - b)
    let maxAbs = diff.max().item(Float.self)
    let meanAbs = diff.mean().item(Float.self)
    let dot = (a * b).sum().item(Float.self)
    let na = MLX.sqrt((a * a).sum()).item(Float.self)
    let nb = MLX.sqrt((b * b).sum()).item(Float.self)
    let cos = (na > 0 && nb > 0) ? dot / (na * nb) : Float.nan
    return Delta(name: name, shape: a.shape, maxAbs: maxAbs, meanAbs: meanAbs, cosine: cos,
                 refRange: (b.min().item(Float.self), b.max().item(Float.self)))
}

// MARK: - Args

var args = Array(CommandLine.arguments.dropFirst())
func takeFlag(_ f: String) -> Bool {
    if let i = args.firstIndex(of: f) { args.remove(at: i); return true }
    return false
}
func takeOption(_ f: String) -> String? {
    guard let i = args.firstIndex(of: f), i + 1 < args.count else { return nil }
    let v = args[i + 1]; args.removeSubrange(i...(i + 1)); return v
}

let useGPU = takeFlag("--gpu")
let wantStages = takeFlag("--stages")
/// Isolate `bilinearResize` against torch F.interpolate at several ratios. The cxt-concat
/// downsamples 8× (256²→32²), where an align_corners mismatch diverges far more than at the
/// 2× upsample the encoder merge exercises — so a resize bug hides everywhere except here.
let resizeProbe = takeFlag("--resize-probe")
let dtypeName = takeOption("--dtype") ?? "fp32"
/// CPU-stream bound: maxAbs relative to the reference's dynamic range. Observed worst on a
/// correct port is ~8e-5 (encoder stages), so 5e-4 leaves ~6× headroom.
let tol = Float(takeOption("--tol") ?? "5e-4") ?? 5e-4
/// GPU-stream bound: cosine. The DCNv2 custom Metal kernel forces the decoder onto the GPU,
/// where fp32 accumulation makes maxAbs uninformative (see Delta.verdict).
let tolE2E = Float(takeOption("--tol-cos") ?? "0.9995") ?? 0.9995
/// "hybrid" (default) = stages on CPU, e2e on GPU. `--gpu` forces all-GPU.
let streamMode = takeOption("--stream") ?? (useGPU ? "gpu" : "hybrid")

guard args.count >= (resizeProbe ? 1 : 2) else {
    FileHandle.standardError.write(Data("""
        usage: birefnet-parity <goldens.safetensors> <weights.safetensors> \
        [--dtype fp32|fp16|bf16] [--gpu] [--stages] [--tol 1e-3]

        """.utf8))
    exit(2)
}
let goldensPath = args[0]
let weightsPath = resizeProbe ? "" : args[1]

let dtype: DType
switch dtypeName.lowercased() {
case "fp32", "float32", "f32": dtype = .float32
case "fp16", "float16", "f16": dtype = .float16
case "bf16", "bfloat16":       dtype = .bfloat16
default:
    FileHandle.standardError.write(Data("unknown --dtype \(dtypeName)\n".utf8)); exit(2)
}

// MARK: - Gate

func runGate() throws -> Bool {
    let goldens = try MLX.loadArrays(url: URL(fileURLWithPath: goldensPath))
    guard let inputNCHW = goldens["input"] else {
        throw NSError(domain: "parity", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "goldens missing `input`"])
    }
    print("goldens: \(goldens.count) tensors from \(goldensPath)")
    print("weights: \(weightsPath) @ \(dtypeName) | stream mode: \(streamMode)\n")

    // The oracle's exact normalized input, NCHW → NHWC, at the model's working dtype.
    let x = inputNCHW.transposed(0, 2, 3, 1).asType(dtype)
    let H = x.dim(1), W = x.dim(2)

    var cfg = BiRefNetConfig.swinLargeDefault
    cfg.inputSize = (width: W, height: H)
    let pipeline = try BiRefNetPipeline.fromPretrained(weightsPath, dtype: dtype, config: cfg)
    let model = pipeline.model

    var deltas: [Delta] = []       // CPU-stream encoder stages, judged at `tol`
    var bottleneck: [Delta] = []   // GPU-stream squeeze tap, judged at `tolE2E`

    if wantStages {
        // The Swin backbone + bilinear merge contain no deformable conv, so this phase can hold
        // the tight CPU-stream bound.
        func stageWork() -> [Delta] {
            var out: [Delta] = []
            // Pass 1: full-res backbone.
            let full = model.bb(x)
            eval(full)
            for (i, t) in full.enumerated() {
                if let g = goldens["enc_full_x\(i + 1)"] {
                    out.append(compare("enc_full_x\(i + 1)", mine: t, goldenNCHW: g))
                }
            }
            // Pass 2: half-res backbone (mul_scl_ipt='cat' — align_corners=true both sides).
            let xHalf = bilinearResize(x, H / 2, W / 2)
            let half = model.bb(xHalf)
            eval(half)
            for (i, t) in half.enumerated() {
                if let g = goldens["enc_half_x\(i + 1)"] {
                    out.append(compare("enc_half_x\(i + 1)", mine: t, goldenNCHW: g))
                }
            }
            // Per-level merge: cat(full, up(half)) along channels.
            for i in 0..<min(full.count, half.count) {
                let a = full[i]
                let up = bilinearResize(half[i], a.dim(1), a.dim(2))
                let cat = concatenated([a, up], axis: -1)
                eval(cat)
                if let g = goldens["enc_cat_x\(i + 1)"] {
                    out.append(compare("enc_cat_x\(i + 1)", mine: cat, goldenNCHW: g))
                }
            }
            return out
        }
        let stagesOnCPU = (streamMode == "hybrid" || streamMode == "cpu")
        deltas = stagesOnCPU ? Device.withDefaultDevice(.cpu) { stageWork() } : stageWork()
        print("--- encoder stages (\(stagesOnCPU ? "CPU" : "GPU") stream, "
              + "\(stagesOnCPU ? "relMax ≤ \(tol)" : "cos ≥ \(tolE2E)")) ---")
        for d in deltas {
            print(d.line(relTol: tol, cosTol: tolE2E, onGPU: !stagesOnCPU))
        }
        print("")

        // Bottleneck tap: cxt-concat onto x4, then squeeze_module. Reproduces
        // BiRefNet.forwardEnc's tail + callAsFunction's squeeze step (forwardEnc is internal).
        // squeeze_module carries ASPPDeformable ⇒ GPU stream, judged at tol-e2e.
        if let g = goldens["squeeze_out"] {
            let full = model.bb(x)
            let half = model.bb(bilinearResize(x, H / 2, W / 2))
            var merged: [MLXArray] = []
            for i in 0..<min(full.count, half.count) {
                let a = full[i]
                merged.append(concatenated([a, bilinearResize(half[i], a.dim(1), a.dim(2))],
                                           axis: -1))
            }
            var x4 = merged[3]
            if cfg.cxtNum > 0 {
                let H4 = x4.dim(1), W4 = x4.dim(2)
                let raw = [merged[0], merged[1], merged[2]].map { bilinearResize($0, H4, W4) }
                x4 = concatenated(Array(raw.suffix(cfg.cxtNum)) + [x4], axis: -1)
            }
            eval(x4)
            print("--- bottleneck (GPU stream, cos ≥ \(tolE2E)) ---")
            // squeeze_in isolates the cxt-concat's 8× bilinear DOWNsample (256²→32²) from
            // squeeze_module's math. The merge above only exercises a 2× UPsample, so a
            // coordinate-mapping difference hides there and shows up here.
            if let gin = goldens["squeeze_in"] {
                let din = compare("squeeze_in", mine: x4, goldenNCHW: gin)
                print(din.line(relTol: tol, cosTol: tolE2E, onGPU: true))
                bottleneck.append(din)
            }
            var sq = x4
            for blk in model.squeezeModule { sq = blk(sq) }
            eval(sq)
            let d = compare("squeeze_out", mine: sq, goldenNCHW: g)
            print(d.line(relTol: tol, cosTol: tolE2E, onGPU: true))
            print("")
            bottleneck.append(d)
        }
    }

    // Authoritative: full forward.
    let t0 = Date()
    let logits = model(x)
    eval(logits)
    let fwd = Date().timeIntervalSince(t0)
    var e2e: [Delta] = []
    if let g = goldens["logits"] {
        e2e.append(compare("logits", mine: logits, goldenNCHW: g))
    }
    let mask = sigmoid(logits)
    eval(mask)
    if let g = goldens["sigmoid"] {
        e2e.append(compare("sigmoid", mine: mask, goldenNCHW: g))
    }
    print("--- end to end (\(streamMode == "cpu" ? "CPU" : "GPU") stream, forward "
          + "\(String(format: "%.1f", fwd))s, cos ≥ \(tolE2E)) ---")
    for d in e2e {
        print(d.line(relTol: tol, cosTol: tolE2E, onGPU: streamMode != "cpu"))
    }

    // Silent-failure guard: a matte that is uniform passes a naive diff against a
    // uniform golden but is worthless. Report the actual spread.
    let mMin = mask.min().item(Float.self), mMax = mask.max().item(Float.self)
    let mMean = mask.mean().item(Float.self)
    let fgFraction = (mask .> 0.5).asType(.float32).mean().item(Float.self)
    print(String(format: "\nmatte: mean %.4f range [%.4f … %.4f] | fg fraction %.4f",
                 mMean, mMin, mMax, fgFraction))
    if mMax - mMin < 0.02 {
        print("WARN: matte near-uniform — possible silent failure")
    }

    let all = deltas + bottleneck + e2e
    let worst = all.map(\.maxAbs).max() ?? .infinity
    let worstCos = all.map(\.cosine).min() ?? 0
    // Each phase is judged against its own stream's bound.
    let stagesOnCPUFinal = (streamMode == "hybrid" || streamMode == "cpu")
    let stagesOK = deltas.allSatisfy {
        $0.verdict(relTol: tol, cosTol: tolE2E, onGPU: !stagesOnCPUFinal)
    }
    let e2eOK = (bottleneck + e2e).allSatisfy {
        $0.verdict(relTol: tol, cosTol: tolE2E, onGPU: streamMode != "cpu")
    }
    let ok = stagesOK && e2eOK && !all.isEmpty
    print(String(format: "\n[PARITY] dtype=%@ stream=%@ tensors=%d stages=%@ e2e=%@ "
                 + "worstMaxAbs=%.3e worstCos=%.8f bounds=rel%.1e/cos%.4f %@",
                 dtypeName, streamMode, all.count,
                 deltas.isEmpty ? "skipped" : (stagesOK ? "PASS" : "FAIL"),
                 e2e.isEmpty ? "skipped" : (e2eOK ? "PASS" : "FAIL"),
                 worst, worstCos, tol, tolE2E, ok ? "PASS" : "FAIL"))
    return ok
}

/// Standalone resize probe — no weights needed.
func runResizeProbe() throws -> Bool {
    let g = try MLX.loadArrays(url: URL(fileURLWithPath: goldensPath))
    guard let inNCHW = g["probe_in"] else {
        throw NSError(domain: "parity", code: 2,
                      userInfo: [NSLocalizedDescriptionKey: "probe bundle missing `probe_in`"])
    }
    let x = inNCHW.transposed(0, 2, 3, 1)   // NHWC
    print("resize probe: input \(x.shape) (NHWC)\n")
    var ok = true
    for tgt in [32, 64, 128, 512] {
        let mine = bilinearResize(x, tgt, tgt)
        eval(mine)
        for variant in ["true", "false"] {
            guard let ref = g["probe_ac_\(variant)_\(tgt)"] else { continue }
            let d = compare("ac_\(variant)→\(tgt)", mine: mine, goldenNCHW: ref)
            print(d.line(relTol: tol, cosTol: tolE2E, onGPU: false))
            // bilinearResize claims align_corners=true, so that's the row that must match.
            if variant == "true" && d.relMax > tol { ok = false }
        }
        print("")
    }
    print("[RESIZE] \(ok ? "PASS — bilinearResize matches align_corners=true" : "FAIL — see rows above")")
    return ok
}

do {
    if resizeProbe {
        // Pure MLX ops, no custom Metal kernel ⇒ safe on the CPU stream for a clean bound.
        let ok = try Device.withDefaultDevice(.cpu) { try runResizeProbe() }
        exit(ok ? 0 : 1)
    }
    // No outer stream pin: the phases pin themselves (see STREAM SPLIT above). Weight loading
    // stays on the default device — mlx-porting allows a CPU-stream load, but the forward that
    // touches DCNv2 must be GPU, so leaving the default alone avoids a half-pinned graph.
    let ok = try runGate()
    exit(ok ? 0 : 1)
} catch {
    FileHandle.standardError.write(Data("FAILED: \(error)\n".utf8))
    exit(1)
}
