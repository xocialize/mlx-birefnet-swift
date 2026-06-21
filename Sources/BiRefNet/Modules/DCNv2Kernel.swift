// Modulated deformable convolution (DCNv2) — forward-only.
//
// PORT FROM: torchvision.ops.deform_conv2d (used in
//            BiRefNet's models/modules/deform_conv.py::DeformableConv2d).
//
// IMPLEMENTATION (decomposed for performance — see PERF_NOTES.md for the
// history of the previous all-in-one kernel and why it was slow):
//
// The DCN math factors as:
//
//     y[b, h, w, co] = Σ_k Σ_ci  W[co, k, ci] · (m[b,h,w,k] · sample(in, k, ci))
//                    = (gathered · W^T)[b, h, w, co]
//
// where `gathered[b, h, w, k, ci] = m[b,h,w,k] · bilinear_sample(in, dy_k, dx_k, ci)`
// only depends on the input + offset + mask (not the weight), and is shared
// across all Cout output channels.
//
// We compute it in two passes:
//
//   1. Custom Metal kernel: pure bilinear gather + modulator multiply, one
//      thread per `(b, h, w, k, ci)`. Output shape `(B, H·W, K·Cin)`.
//   2. MLX matmul against the flattened weight `(Cout, K·Cin)`. This hits the
//      MPSGraph fp16 fast path, which is ~10× faster than a hand-rolled
//      multiply-accumulate in a custom kernel.
//
// Inputs / outputs are NHWC (MLX Conv2d layout). Stride and dilation are
// hard-coded to 1 to match every BiRefNet call site.

import Foundation
import MLX
import MLXFast

// MARK: - Bilinear gather kernel

/// Source of the bilinear-gather kernel.
///
/// Template params:
/// - `T`     — element type (set to the input dtype). When fp16, the kernel
///             compiles with `half` reads and accumulators so we actually hit
///             Apple GPU's 2× fp16 throughput. The previous version cast
///             everything to `float` internally and left fp16 perf on the
///             floor (decoder share didn't shrink at all when switching
///             from fp32 to fp16 — see PERF_NOTES.md).
/// - `KH`, `KW`, `PAD` — kernel size and padding constants.
///
/// Each thread handles ONE element of the output `(B, H, W, K, Cin)` tensor,
/// where `K = KH·KW`. The thread:
///
/// 1. Decodes `(b, h, w, k, ci)` from its flat index.
/// 2. Looks up `(dy, dx) = offset[b, h, w, 2k:2k+2]` and `m = mask[b, h, w, k]`.
/// 3. Bilinear-samples `input[b, ?, ?, ci]` at the deformed position.
/// 4. Writes `m · sample` to `output[b, h, w, k, ci]`.
///
/// The mask multiplication is fused into the gather so the matmul that
/// follows can be a plain GEMM without an extra elementwise step.
///
/// **Precision note:** the bilinear weights and sample sum are kept in
/// `T` to extract fp16 throughput. The 4-term sum of `T`-precision corner
/// reads is numerically safe even at fp16 (each term is `c · x` with
/// `c ∈ [0, 1]`).
private let bilinearGatherSource = """
    uint elem = thread_position_in_grid.x;

    int B    = input_shape[0];
    int Hi   = input_shape[1];
    int Wi   = input_shape[2];
    int Cin  = input_shape[3];

    int K = KH * KW;
    int total = B * Hi * Wi * K * Cin;
    if ((int)elem >= total) return;

    int ci   =  (int)elem        % Cin;
    int k    = ((int)elem / Cin) % K;
    int wo   = ((int)elem / (Cin * K)) % Wi;
    int ho   = ((int)elem / (Cin * K * Wi)) % Hi;
    int b    =  (int)elem / (Cin * K * Wi * Hi);

    int ky = k / KW;
    int kx = k % KW;

    // Sample location (input-space, before clipping). Coordinate arithmetic
    // stays float — small fixed cost, avoids fp16 range issues at large H/W.
    int offsetBase = ((b * Hi + ho) * Wi + wo) * (2 * K) + 2 * k;
    int maskBase   = ((b * Hi + ho) * Wi + wo) * K + k;
    float dy = (float) offset[offsetBase + 0];
    float dx = (float) offset[offsetBase + 1];
    T     m  = mask[maskBase];

    float sy = float(ho - PAD + ky) + dy;
    float sx = float(wo - PAD + kx) + dx;

    int y0 = (int) floor(sy);
    int x0 = (int) floor(sx);
    int y1 = y0 + 1;
    int x1 = x0 + 1;

    T wy1 = T(sy - float(y0));
    T wx1 = T(sx - float(x0));
    T wy0 = T(1.0) - wy1;
    T wx0 = T(1.0) - wx1;

    T c00 = wy0 * wx0;
    T c01 = wy0 * wx1;
    T c10 = wy1 * wx0;
    T c11 = wy1 * wx1;

    bool inY0 = (y0 >= 0 && y0 < Hi);
    bool inY1 = (y1 >= 0 && y1 < Hi);
    bool inX0 = (x0 >= 0 && x0 < Wi);
    bool inX1 = (x1 >= 0 && x1 < Wi);

    int rowStride   = Wi * Cin;
    int batchStride = Hi * rowStride;
    int base = b * batchStride;

    T s = T(0);
    if (inY0 && inX0) s += c00 * input[base + y0 * rowStride + x0 * Cin + ci];
    if (inY0 && inX1) s += c01 * input[base + y0 * rowStride + x1 * Cin + ci];
    if (inY1 && inX0) s += c10 * input[base + y1 * rowStride + x0 * Cin + ci];
    if (inY1 && inX1) s += c11 * input[base + y1 * rowStride + x1 * Cin + ci];

    output[elem] = s * m;
    """

nonisolated(unsafe) private let bilinearGatherKernel: MLXFast.MLXFastKernel = {
    MLXFast.metalKernel(
        name: "dcnv2_bilinear_gather",
        inputNames: ["input", "offset", "mask"],
        outputNames: ["output"],
        source: bilinearGatherSource
    )
}()

// MARK: - Fused kernel (experiment)

/// Fully fused DCN forward: bilinear gather + per-pixel dot-product against
/// all `Cout` weights, all in one kernel, with gathered values living in
/// threadgroup memory instead of round-tripping through global memory.
///
/// Layout: one threadgroup per spatial output pixel `(b, h, w)`. The
/// threadgroup cooperatively gathers the `K * Cin` deformed samples into
/// threadgroup memory (so each bilinear sample is computed once, not 256
/// times once per output channel). Each thread then computes
/// `OUT_PER_THREAD` consecutive `Cout` output channels by reading those
/// `K · Cin` values and the corresponding weight row from global memory.
///
/// Template params: `T` (element type), `KH`, `KW`, `PAD`, `KCIN` (= K·Cin),
/// `COUT`, and `TG_SIZE` (threads per group). `OUT_PER_THREAD = COUT / TG_SIZE`.
///
/// **Caveats**: without `simdgroup_matrix`, the matmul throughput is below
/// MLX's tuned matmul, so this is competitive only when the matmul cost
/// saved from skipping the global gather roundtrip outweighs the slower
/// per-thread MAC loop. See `useFusedKernel` for the dispatch heuristic.
private let dcnFusedSource = """
    threadgroup T tg_gathered[KCIN];

    int B    = input_shape[0];
    int Hi   = input_shape[1];
    int Wi   = input_shape[2];
    int Cin  = input_shape[3];

    int wo  = (int) threadgroup_position_in_grid.x;
    int ho  = (int) threadgroup_position_in_grid.y;
    int b   = (int) threadgroup_position_in_grid.z;
    int tid = (int) thread_position_in_threadgroup.x;

    // ── Phase 1: cooperative bilinear gather into threadgroup memory ────
    // tg_gathered[k * Cin + ci] = mask[k] * bilinear(input, dy_k, dx_k, ci)
    int K = KH * KW;
    int offsetBase = ((b * Hi + ho) * Wi + wo) * (2 * K);
    int maskBase   = ((b * Hi + ho) * Wi + wo) * K;

    for (int idx = tid; idx < KCIN; idx += TG_SIZE) {
        int ci = idx % Cin;
        int k  = idx / Cin;
        int ky = k / KW;
        int kx = k % KW;

        float dy = (float) offset[offsetBase + 2 * k + 0];
        float dx = (float) offset[offsetBase + 2 * k + 1];
        T     m  = mask[maskBase + k];

        float sy = float(ho - PAD + ky) + dy;
        float sx = float(wo - PAD + kx) + dx;

        int y0 = (int) floor(sy);
        int x0 = (int) floor(sx);
        int y1 = y0 + 1;
        int x1 = x0 + 1;

        T wy1 = T(sy - float(y0));
        T wx1 = T(sx - float(x0));
        T wy0 = T(1.0) - wy1;
        T wx0 = T(1.0) - wx1;

        T c00 = wy0 * wx0;
        T c01 = wy0 * wx1;
        T c10 = wy1 * wx0;
        T c11 = wy1 * wx1;

        bool inY0 = (y0 >= 0 && y0 < Hi);
        bool inY1 = (y1 >= 0 && y1 < Hi);
        bool inX0 = (x0 >= 0 && x0 < Wi);
        bool inX1 = (x1 >= 0 && x1 < Wi);

        int rowStride   = Wi * Cin;
        int batchStride = Hi * rowStride;
        int base = b * batchStride;

        T s = T(0);
        if (inY0 && inX0) s += c00 * input[base + y0 * rowStride + x0 * Cin + ci];
        if (inY0 && inX1) s += c01 * input[base + y0 * rowStride + x1 * Cin + ci];
        if (inY1 && inX0) s += c10 * input[base + y1 * rowStride + x0 * Cin + ci];
        if (inY1 && inX1) s += c11 * input[base + y1 * rowStride + x1 * Cin + ci];

        tg_gathered[idx] = s * m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: per-thread dot-product against Cout weight rows ────────
    // Each thread handles OUT_PER_THREAD consecutive Cout channels.
    const int OUT_PER_THREAD = COUT / TG_SIZE;
    int co_base = tid * OUT_PER_THREAD;
    int outBase = ((b * Hi + ho) * Wi + wo) * COUT + co_base;

    for (int oc = 0; oc < OUT_PER_THREAD; ++oc) {
        int co = co_base + oc;
        int wBase = co * KCIN;
        T acc = T(0);
        for (int j = 0; j < KCIN; ++j) {
            acc += tg_gathered[j] * weight[wBase + j];
        }
        output[outBase + oc] = acc;
    }
    """

nonisolated(unsafe) private let dcnFusedKernel: MLXFast.MLXFastKernel = {
    MLXFast.metalKernel(
        name: "dcnv2_fused",
        inputNames: ["input", "offset", "mask", "weight"],
        outputNames: ["output"],
        source: dcnFusedSource
    )
}()

/// Dispatch heuristic for the fused kernel.
///
/// **Currently always returns `false`**: an unfused gather + MLX matmul is
/// faster end-to-end than the fused kernel below, because MLX matmul hits
/// ~27 TFLOPS fp16 (MPSGraph tile cores) on the largest DCN shape, while
/// the per-thread MAC loop in our fused kernel tops out around 5 TFLOPS.
/// The gather → matmul global-memory roundtrip we save (~3 ms / largest
/// call) doesn't make up for the slower inner loop.
///
/// The kernel source + dispatch are kept in the file as a starting point
/// for a future tiled `simdgroup_matrix` rewrite — see `PERF_NOTES.md`.
/// Flip this to `true` (and pick a real heuristic) once the kernel uses
/// matrix cores.
@inlinable
func useFusedKernel(B: Int, H: Int, W: Int, K: Int, Cin: Int, Cout: Int, dtypeBytes: Int) -> Bool {
    return false
}

// MARK: - Public DCN forward

/// Forward modulated-deformable-conv. Inputs are NHWC float arrays.
///
/// - Parameters:
///   - input:    `(B, H, W, Cin)`
///   - offset:   `(B, H, W, 2 · kH · kW)`
///   - mask:     `(B, H, W,      kH · kW)`  — already `2 · sigmoid(modulator)`
///   - weight:   `(Cout, kH, kW, Cin)`        — same layout as `MLXNN.Conv2d`
///   - kernelSize: `(kH, kW)`
///   - padding:  symmetric padding around the spatial dims.
/// - Returns: `(B, H, W, Cout)` with the same dtype as `input`.
public func deformConv2dForward(
    input: MLXArray,
    offset: MLXArray,
    mask: MLXArray,
    weight: MLXArray,
    kernelSize: (Int, Int),
    padding: Int
) -> MLXArray {
    precondition(input.ndim == 4)
    precondition(offset.ndim == 4)
    precondition(mask.ndim == 4)
    precondition(weight.ndim == 4)

    let B = input.dim(0)
    let H = input.dim(1)
    let W = input.dim(2)
    let Cin = input.dim(3)
    let Cout = weight.dim(0)
    let kH = kernelSize.0
    let kW = kernelSize.1
    let K = kH * kW
    let dtypeBytes = (input.dtype == .float32) ? 4 : 2

    // Heuristic dispatch: large-shape DCNs go through the fused kernel
    // (avoids gather → matmul global-memory roundtrip); small-shape DCNs
    // stay on the gather + MLX matmul path (better matmul throughput).
    if useFusedKernel(B: B, H: H, W: W, K: K, Cin: Cin, Cout: Cout,
                      dtypeBytes: dtypeBytes) {
        let tgSize: Int = (Cout % 256 == 0) ? 256 : 128
        // MLX grid is total threads, threadgroups_per_grid = grid / threadGroup.
        // We want one threadgroup per (b, h, w) spatial pixel; multiply the X
        // dimension by `tgSize` so threadgroups_per_grid.x == W.
        let grid = (W * tgSize, H, B)
        let group = (tgSize, 1, 1)
        let outs = dcnFusedKernel(
            [input, offset, mask, weight.reshaped([Cout, K * Cin])],
            template: [
                ("T", input.dtype),
                ("KH", kH),
                ("KW", kW),
                ("PAD", padding),
                ("KCIN", K * Cin),
                ("COUT", Cout),
                ("TG_SIZE", tgSize),
            ],
            grid: grid,
            threadGroup: group,
            outputShapes: [[B, H, W, Cout]],
            outputDTypes: [input.dtype]
        )
        return outs[0]
    }

    // ── Step 1: bilinear gather + modulator fuse ────────────────────────
    // Output: (B, H, W, K, Cin), flat-indexed as one buffer.
    let gatherTotal = B * H * W * K * Cin
    let threadsPerGroup = min(256, gatherTotal)
    let gridSize = ((gatherTotal + threadsPerGroup - 1) / threadsPerGroup) * threadsPerGroup

    let gatherOut = bilinearGatherKernel(
        [input, offset, mask],
        template: [
            ("T", input.dtype),
            ("KH", kH),
            ("KW", kW),
            ("PAD", padding),
        ],
        grid: (gridSize, 1, 1),
        threadGroup: (threadsPerGroup, 1, 1),
        outputShapes: [[B, H, W, K, Cin]],
        outputDTypes: [input.dtype]
    )[0]

    // Reshape to (B·H·W, K·Cin) for the GEMM.
    let gatheredMat = gatherOut.reshaped([B * H * W, K * Cin])

    // ── Step 2: matmul (MPSGraph-backed fp16 fast path) ─────────────────
    // weight: (Cout, kH, kW, Cin) → (Cout, K · Cin), then transpose for
    //   `gathered · W^T` form.
    let weightFlat = weight.reshaped([Cout, K * Cin]).transposed(1, 0)
    let outFlat = gatheredMat.matmul(weightFlat)   // (B·H·W, Cout)

    return outFlat.reshaped([B, H, W, Cout])
}
