import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_N': 32}, num_warps=8),
        triton.Config({'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_N': 64}, num_warps=8),
        triton.Config({'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 128}, num_warps=8),
    ],
    key=['TopK', 'Num_Heads'],
)
@triton.jit
def _dsa_kernel(
    Q_Nope,
    Q_Pe,  # [T, H, 512], [T, H, 64]
    K_C_All,
    K_Pe_All,  # [P*S, 512], [P*S, 64]
    Indices,  # [T, 2048]
    Output,
    LSE,  # [T, H, 512], [T, H]
    sm_scale,  # float
    stride_qt,
    stride_qh,
    stride_qd,
    stride_qpe_t,
    stride_qpe_h,
    stride_qpe_d,
    stride_kc_all_t,
    stride_kc_all_d,
    stride_kpe_all_t,
    stride_kpe_all_d,
    stride_ind_t,
    stride_ind_k,
    stride_out_t,
    stride_out_h,
    stride_out_d,
    stride_lse_t,
    stride_lse_h,
    Num_Heads: tl.constexpr,
    Head_Dim_C: tl.constexpr,
    Head_Dim_P: tl.constexpr,
    TopK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    # Map pid to (t, h)
    t_idx = pid // Num_Heads
    h_idx = pid % Num_Heads

    # Q pointers
    q_nope_ptr = Q_Nope + t_idx * stride_qt + h_idx * stride_qh
    q_pe_ptr = Q_Pe + t_idx * stride_qpe_t + h_idx * stride_qpe_h

    # Output pointers
    out_ptr = Output + t_idx * stride_out_t + h_idx * stride_out_h
    lse_ptr = LSE + t_idx * stride_lse_t + h_idx * stride_lse_h

    # Offsets
    offs_dc = tl.arange(0, Head_Dim_C)
    offs_dp = tl.arange(0, Head_Dim_P)

    # Load Q and cast to float32 for precision
    q_nope = tl.load(q_nope_ptr + offs_dc).to(tl.float32)
    q_pe = tl.load(q_pe_ptr + offs_dp).to(tl.float32)

    # Initialize accumulators in float32
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([Head_Dim_C], dtype=tl.float32)

    # Pre-compute log2(e) for base-2 conversion
    log2_e = 1.44269504
    scale_val = sm_scale * log2_e

    # Indices pointer base
    idx_base_ptr = Indices + t_idx * stride_ind_t

    # Loop over blocks of K
    for start_k in range(0, TopK, BLOCK_N):
        offs_k = start_k + tl.arange(0, BLOCK_N)
        idx_ptr = idx_base_ptr + offs_k * stride_ind_k

        # Load indices
        indices = tl.load(idx_ptr)

        # Mask for valid indices (-1 is invalid)
        valid_mask = indices != -1

        # Safe indices for gathering K (clamp to 0 to avoid out-of-bounds, though masked)
        safe_indices = tl.where(valid_mask, indices, 0)

        # Gather Kc [BLOCK_N, 512]
        kc_ptrs = (
            K_C_All
            + safe_indices[:, None] * stride_kc_all_t
            + offs_dc[None, :] * stride_kc_all_d
        )
        kc = tl.load(kc_ptrs, mask=valid_mask[:, None], other=0.0).to(tl.float32)

        # Gather Kp [BLOCK_N, 64]
        kpe_ptrs = (
            K_Pe_All
            + safe_indices[:, None] * stride_kpe_all_t
            + offs_dp[None, :] * stride_kpe_all_d
        )
        kp = tl.load(kpe_ptrs, mask=valid_mask[:, None], other=0.0).to(tl.float32)

        # Compute scores (GEMV: q @ K.T)
        # Using element-wise mul + sum for vector-matrix product
        score_c = tl.sum(q_nope[None, :] * kc, axis=1)
        score_p = tl.sum(q_pe[None, :] * kp, axis=1)

        score = (score_c + score_p) * scale_val

        # Mask invalid scores before softmax
        score = tl.where(valid_mask, score, -float("inf"))

        # Online Softmax update (base-2)
        m_curr = tl.max(score, axis=0)
        m_new = tl.maximum(m_i, m_curr)

        # Robust subtraction to avoid NaN (inf - inf)
        offset = tl.where(m_new == -float('inf'), 0.0, m_new)
        
        p = tl.exp2(score - offset)
        alpha = tl.exp2(m_i - offset)
        
        # Update accumulators
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * kc, axis=0)

        m_i = m_new

    # Finalize
    if l_i > 0.0:
        # lse = m_new + log2(l_i)
        lse_val = m_i + tl.log2(l_i)
        tl.store(lse_ptr, lse_val.to(tl.float32))

        out_val = acc / l_i
        # Store in the output format
        tl.store(out_ptr + offs_dc, out_val.to(Output.dtype.element_ty))
    else:
        tl.store(lse_ptr, -float("inf"))
        tl.store(out_ptr + offs_dc, tl.zeros([Head_Dim_C], dtype=tl.float32))


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    # Validations
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    _, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]

    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert topk == 2048
    assert page_size == 64

    # Flatten paged KV cache to token-level
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv)
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe)

    # Kernel config
    grid = (num_tokens * num_qo_heads,)

    _dsa_kernel[grid](  # pyright: ignore[reportIndexIssue]
        q_nope,
        q_pe,
        Kc_all,
        Kp_all,
        sparse_indices,
        output,
        lse,
        sm_scale,
        q_nope.stride(0),
        q_nope.stride(1),
        q_nope.stride(2),
        q_pe.stride(0),
        q_pe.stride(1),
        q_pe.stride(2),
        Kc_all.stride(0),
        Kc_all.stride(1),
        Kp_all.stride(0),
        Kp_all.stride(1),
        sparse_indices.stride(0),
        sparse_indices.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        Num_Heads=num_qo_heads,
        Head_Dim_C=head_dim_ckv,
        Head_Dim_P=head_dim_kpe,
        TopK=topk,
        # BLOCK_N is handled by autotuner
    )


def kernel(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse)
