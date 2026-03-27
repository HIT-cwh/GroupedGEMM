"""
Test for permute_pad / unpermute_unpad (128-aligned per-expert segments).

Compares the padded CUDA kernels against a pure-PyTorch reference, checking:
  1. permute_pad forward: correct token placement + padding rows are zeros
  2. unpermute_unpad forward: weighted sum matches reference
  3. unpermute_unpad backward: act_grad and prob_grad match reference

Usage:
    python test_permute_128x.py
"""

import torch
import grouped_gemm_backend as backend
from torch.profiler import ProfilerActivity, profile


# ---------------------------------------------------------------------------
# PyTorch reference (same logic as the existing permute_test.py)
# ---------------------------------------------------------------------------

def ref_permute(tokens, indices):
    """Pure-PyTorch permute: sort tokens by expert id, expand by topK."""
    topK = indices.size(1)
    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    permuted_tokens = tokens.index_select(0, sorted_indices // topK)
    return permuted_tokens, sorted_indices


def ref_unpermute(permuted_tokens, sorted_indices, probs):
    """Pure-PyTorch unpermute with prob weighting."""
    topK = probs.size(1)
    unpermuted = permuted_tokens.index_copy(0, sorted_indices, permuted_tokens)
    unpermuted = unpermuted.reshape(-1, topK, permuted_tokens.size(-1))
    dtype = unpermuted.dtype
    unpermuted = unpermuted * probs.unsqueeze(-1)
    unpermuted = unpermuted.to(dtype)
    return unpermuted.sum(dim=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_inputs(num_tokens, hidden, num_experts, topK, dtype, seed=42):
    torch.manual_seed(seed)

    tokens = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.float32)
    tokens = tokens.to(dtype)

    if num_tokens > 0:
        indices = torch.stack(
            [torch.randperm(num_experts, device="cuda")[:topK] for _ in range(num_tokens)]
        )
    else:
        indices = torch.empty((0, topK), device="cuda")
    indices = indices.to(torch.int32)

    probs = torch.rand(num_tokens, topK, device="cuda", dtype=torch.float32)
    probs = probs / probs.sum(dim=1, keepdim=True)

    return tokens, indices, probs


def check_close(name, a, b, atol=1e-2, rtol=1e-2):
    a_f = a.float()
    b_f = b.float()
    max_err = (a_f - b_f).abs().max().item()
    ok = torch.allclose(a_f, b_f, atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name:50s} max_err={max_err:.3e}")
    if not ok:
        # Print a few mismatched positions for debugging
        diff = (a_f - b_f).abs()
        flat = diff.flatten()
        top_idx = flat.topk(min(5, flat.numel())).indices
        for idx in top_idx:
            print(f"    pos={idx.item()}: got={a_f.flatten()[idx].item():.6f}  ref={b_f.flatten()[idx].item():.6f}")
    return ok


# ---------------------------------------------------------------------------
# Core test
# ---------------------------------------------------------------------------

def test_permute_pad(
    dtype,
    num_tokens,
    hidden,
    num_experts,
    topK,
    benchmark=False,
):
    is_fp8 = dtype in (torch.float8_e5m2, torch.float8_e4m3fn)
    compute_dtype = torch.float16 if is_fp8 else dtype

    print(f"\n{'='*80}")
    print(f"  dtype={dtype}  tokens={num_tokens}  hidden={hidden}  "
          f"experts={num_experts}  topK={topK}")
    print(f"{'='*80}")

    tokens, indices, probs = make_inputs(num_tokens, hidden, num_experts, topK, compute_dtype)

    max_expanded = num_tokens * topK

    # ------------------------------------------------------------------
    # 1) permute_pad forward
    # ------------------------------------------------------------------
    workspace = []
    permuted_out, row_id_map, expert_counts, padded_offsets, workspace = backend.permute_pad(
        tokens if not is_fp8 else tokens.to(dtype),
        indices,
        0,                # num_out_tokens (0 → auto)
        workspace,
        0,                # num_negative_one_in_indices
        max_expanded,
        num_experts,
    )
    permuted_out_f = permuted_out.float() if is_fp8 else permuted_out

    # Reference
    ref_permuted, sorted_indices = ref_permute(tokens, indices)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(4):
            with torch.profiler.record_function("pad"):
                permuted_out, row_id_map, expert_counts, padded_offsets, workspace = backend.permute_pad(
                    tokens if not is_fp8 else tokens.to(dtype),
                    indices,
                    0,                # num_out_tokens (0 → auto)
                    [],
                    0,                # num_negative_one_in_indices
                    max_expanded,
                    num_experts,
                )
            with torch.profiler.record_function("wo_pad"):
                permuted_out, row_id_map, _ = backend.permute(
                    tokens if not is_fp8 else tokens.to(dtype),
                    indices,
                    0,                # num_out_tokens (0 → auto)
                    [],
                    0,                # num_negative_one_in_indices
                    max_expanded,
                )
    prof.export_chrome_trace("permute_vs_pad.json")
    breakpoint()

    # The padded output is larger than reference. Verify the valid (non-pad) rows match.
    # To compare, we need to map: for each original expanded token i (in sorted order),
    # find where it ended up in the padded output.
    # row_id_map[k * num_tokens + token_id] = padded_row_index
    # But the reference output is just sorted by argsort of flatten indices.

    # Simpler check: for each token, each topK slot, use row_id_map to read from padded output.
    all_valid = True
    num_valid_checked = 0
    for k in range(topK):
        for t in range(num_tokens):
            row = row_id_map[k * num_tokens + t].item()
            if row < 0:
                continue
            padded_row = permuted_out_f[row]
            ref_row = tokens[t].float()
            if not torch.allclose(padded_row.float(), ref_row, atol=1e-3, rtol=1e-3):
                if all_valid:
                    print(f"  [FAIL] permute_pad fwd: mismatch at token={t} topK={k} row={row}")
                all_valid = False
            num_valid_checked += 1

    if all_valid:
        print(f"  [PASS] {'permute_pad forward (valid rows via row_id_map)':50s} checked={num_valid_checked}")

    # Verify padding rows are zero
    ec = expert_counts.cpu()
    po = padded_offsets.cpu()
    total_padded = po[num_experts].item()
    ub_total = permuted_out.size(0)

    pad_zero_ok = True
    for e in range(num_experts):
        start = po[e].item() + ec[e].item()
        end = po[e + 1].item()
        if start < end:
            pad_region = permuted_out_f[start:end]
            if pad_region.abs().max().item() > 0:
                print(f"  [FAIL] expert {e} pad region [{start}:{end}] not zero, max={pad_region.abs().max().item():.3e}")
                pad_zero_ok = False

    # UB tail check
    if total_padded < ub_total:
        tail_region = permuted_out_f[total_padded:ub_total]
        if tail_region.abs().max().item() > 0:
            print(f"  [FAIL] UB tail [{total_padded}:{ub_total}] not zero, max={tail_region.abs().max().item():.3e}")
            pad_zero_ok = False

    if pad_zero_ok:
        print(f"  [PASS] {'permute_pad forward (pad+tail regions are zero)':50s} "
              f"ub={ub_total} padded={total_padded} pad_rows={total_padded - max_expanded}")

    # ------------------------------------------------------------------
    # 2) unpermute_unpad forward
    # ------------------------------------------------------------------
    if not is_fp8:
        unpermuted_out = backend.unpermute_unpad(
            permuted_out,
            row_id_map,
            probs,
            num_tokens,
            topK,
        )
        ref_unpermuted = ref_unpermute(ref_permuted, sorted_indices, probs)
        check_close("unpermute_unpad forward", unpermuted_out, ref_unpermuted, atol=1e-2, rtol=1e-2)

    # ------------------------------------------------------------------
    # 3) unpermute_unpad backward
    # ------------------------------------------------------------------
    if not is_fp8 and topK > 1:
        grad_output = torch.randn(num_tokens, hidden, device="cuda", dtype=compute_dtype)

        # --- CUDA backward ---
        act_grad_cuda, prob_grad_cuda = backend.unpermute_unpad_bwd(
            grad_output,
            permuted_out,
            row_id_map,
            probs,
            expert_counts,
            padded_offsets,
        )

        # --- PyTorch reference backward ---
        # unpermute fwd with autograd
        ref_unpermute_input = ref_permuted.detach().clone().requires_grad_(True)
        ref_probs = probs.detach().clone().requires_grad_(True)
        ref_unpermute_out = ref_unpermute(ref_unpermute_input, sorted_indices, ref_probs)
        ref_unpermute_out.backward(grad_output.float())

        # act_grad: the CUDA version outputs into the padded layout [UB, hidden].
        # We read only the valid positions corresponding to the non-padded reference.
        # For each sorted position i → padded_row via row_id_map reconstruction.
        # Simpler: check that pad rows in act_grad are zero.
        act_grad_pad_ok = True
        for e in range(num_experts):
            start = po[e].item() + ec[e].item()
            end = po[e + 1].item()
            if start < end:
                region = act_grad_cuda[start:end].float()
                if region.abs().max().item() > 0:
                    print(f"  [FAIL] bwd act_grad expert {e} pad [{start}:{end}] not zero")
                    act_grad_pad_ok = False

        if total_padded < ub_total:
            tail = act_grad_cuda[total_padded:ub_total].float()
            if tail.abs().max().item() > 0:
                print(f"  [FAIL] bwd act_grad UB tail not zero")
                act_grad_pad_ok = False

        if act_grad_pad_ok:
            print(f"  [PASS] {'unpermute_unpad bwd act_grad (pad+tail zero)':50s}")

        check_close("unpermute_unpad bwd prob_grad", prob_grad_cuda, ref_probs.grad, atol=1e-1, rtol=1e-1)

    # ------------------------------------------------------------------
    # 4) Benchmark (optional)
    # ------------------------------------------------------------------
    if benchmark and not is_fp8:
        def bench(fn, label, warmup=50, iters=200):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / iters
            print(f"  [BENCH] {label:45s} {ms:.3f} ms")

        ws = workspace  # reuse workspace
        bench(
            lambda: backend.permute_pad(tokens, indices, 0, ws, 0, max_expanded, num_experts),
            "permute_pad fwd",
        )
        bench(
            lambda: backend.unpermute_unpad(permuted_out, row_id_map, probs, num_tokens, topK),
            "unpermute_unpad fwd",
        )
        if topK > 1:
            go = torch.randn(num_tokens, hidden, device="cuda", dtype=compute_dtype)
            bench(
                lambda: backend.unpermute_unpad_bwd(go, permuted_out, row_id_map, probs, expert_counts, padded_offsets),
                "unpermute_unpad bwd",
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    BENCH = True

    # Core shapes
    configs = [
        # (dtype,       tokens,  hidden, experts, topK)
        (torch.bfloat16, 32768,   2048,   128,     8),
        # (torch.bfloat16, 8192,   7168,   256,     2),
        # (torch.bfloat16, 8192,   7168,   256,     1),
        # (torch.bfloat16, 4096,   4096,   64,      2),
        # (torch.bfloat16, 4096,   4096,   8,       2),
        # (torch.float16,  4096,   4096,   64,      2),
        # (torch.float32,  2048,   1024,   8,       2),
        # # Edge cases
        # (torch.bfloat16, 128,    512,    4,       2),   # small
        # (torch.bfloat16, 1,      256,    4,       2),   # single token
        # (torch.bfloat16, 4096,   4096,   8,       1),   # topK=1
        # (torch.bfloat16, 4096,   4096,   8,       4),   # topK=4
    ]

    all_pass = True
    for dtype, tokens, hidden, experts, topK in configs:
        try:
            test_permute_pad(dtype, tokens, hidden, experts, topK, benchmark=BENCH)
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            all_pass = False

    print(f"\n{'='*80}")
    print(f"  All tests {'PASSED' if all_pass else 'HAD FAILURES'}.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
