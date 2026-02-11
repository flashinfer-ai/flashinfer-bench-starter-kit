import torch
import triton
import triton.language as tl


@triton.jit
def dsa_sparse_attention_kernel(
    q_nope_ptr,
    q_pe_ptr,
    ckv_cache_ptr,
    kpe_cache_ptr,
    sparse_indices_ptr,
    sm_scale,
    output_ptr,
    lse_ptr,
    num_tokens,
    num_pages,
    page_size,
    topk,
    NUM_QO_HEADS: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    HEAD_DIM_KPE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= num_tokens:
        return
    
    NEG_INF = -1e38
    INV_LN2 = 1.4426950408889634  # 1 / ln(2)
    
    for h in range(NUM_QO_HEADS):
        qn = tl.load(q_nope_ptr + pid * NUM_QO_HEADS * HEAD_DIM_CKV + h * HEAD_DIM_CKV + tl.arange(0, HEAD_DIM_CKV)).to(tl.float32)
        qp = tl.load(q_pe_ptr + pid * NUM_QO_HEADS * HEAD_DIM_KPE + h * HEAD_DIM_KPE + tl.arange(0, HEAD_DIM_KPE)).to(tl.float32)
        
        acc = tl.zeros([HEAD_DIM_CKV], dtype=tl.float32)
        m = NEG_INF
        d = 0.0
        
        indices_base = sparse_indices_ptr + pid * topk
        
        for k in range(topk):
            idx = tl.load(indices_base + k)
            
            is_valid = idx != -1
            
            if is_valid:
                page_idx = idx // page_size
                offset = idx - page_idx * page_size
                
                kc = tl.load(ckv_cache_ptr + page_idx * page_size * HEAD_DIM_CKV + offset * HEAD_DIM_CKV + tl.arange(0, HEAD_DIM_CKV)).to(tl.float32)
                kp = tl.load(kpe_cache_ptr + page_idx * page_size * HEAD_DIM_KPE + offset * HEAD_DIM_KPE + tl.arange(0, HEAD_DIM_KPE)).to(tl.float32)
                
                attn = (tl.sum(qn * kc) + tl.sum(qp * kp)) * sm_scale
                
                new_m = tl.maximum(m, attn)
                alpha = tl.exp(m - new_m)
                beta = tl.exp(attn - new_m)
                acc = acc * alpha + beta * kc
                d = d * alpha + beta
                m = new_m
        
        if d > 0:
            acc = acc / d
            lse_final = (m + tl.log(d)) * INV_LN2
        else:
            acc = tl.zeros([HEAD_DIM_CKV], dtype=tl.float32)
            lse_final = NEG_INF
        
        out_ptr = output_ptr + pid * NUM_QO_HEADS * HEAD_DIM_CKV + h * HEAD_DIM_CKV
        tl.store(out_ptr + tl.arange(0, HEAD_DIM_CKV), acc.to(tl.bfloat16))
        
        lse_ptr_h = lse_ptr + pid * NUM_QO_HEADS + h
        tl.store(lse_ptr_h, lse_final)


def kernel(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sm_scale: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    _, _, head_dim_kpe = q_pe.shape
    num_pages, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]
    
    q_nope = q_nope.contiguous()
    q_pe = q_pe.contiguous()
    ckv_cache = ckv_cache.contiguous()
    kpe_cache = kpe_cache.contiguous()
    sparse_indices = sparse_indices.contiguous()
    output = output.contiguous()
    lse = lse.contiguous()
    
    grid = (num_tokens,)
    
    dsa_sparse_attention_kernel[grid](
        q_nope_ptr=q_nope,
        q_pe_ptr=q_pe,
        ckv_cache_ptr=ckv_cache,
        kpe_cache_ptr=kpe_cache,
        sparse_indices_ptr=sparse_indices,
        sm_scale=sm_scale.item() if hasattr(sm_scale, 'item') else sm_scale,
        output_ptr=output,
        lse_ptr=lse,
        num_tokens=num_tokens,
        num_pages=num_pages,
        page_size=page_size,
        topk=topk,
        NUM_QO_HEADS=num_qo_heads,
        HEAD_DIM_CKV=head_dim_ckv,
        HEAD_DIM_KPE=head_dim_kpe,
    )

