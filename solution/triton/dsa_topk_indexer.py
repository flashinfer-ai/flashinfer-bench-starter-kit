import torch


def dequant_fp8_kv_cache(k_index_cache_fp8):
    """Dequantize FP8 KV cache from deep_gemm format."""
    # View as uint8 for correct byte interpretation
    k_index_cache_fp8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, num_heads, head_dim_sf = k_index_cache_fp8.shape
    head_dim = head_dim_sf - 4  # 128

    # Go back to flat format to reverse the packing
    kv_flat = k_index_cache_fp8.view(num_pages, page_size * head_dim_sf)

    # FP8 part: first page_size * head_dim bytes
    fp8_bytes = kv_flat[:, : page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(
        torch.float8_e4m3fn
    )
    fp8_float = fp8_tensor.to(torch.float32)

    # Scale part: last page_size * 4 bytes -> page_size float32 values
    scale_bytes = kv_flat[:, page_size * head_dim :].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(
        torch.float32
    )  # [num_pages, page_size, 1]

    return fp8_float * scale


@torch.no_grad()
def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    num_pages, page_size, _, _ = k_index_cache_fp8.shape
    topk = 2048

    # Check constants
    assert num_index_heads == 64
    assert index_head_dim == 128
    assert page_size == 64

    # Dequantize inputs
    q = q_index_fp8.to(torch.float32)  # [batch, heads, head_dim]
    K_all = dequant_fp8_kv_cache(k_index_cache_fp8)  # [num_pages, page_size, head_dim]

    topk_indices.fill_(-1)

    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())

        if seq_len == 0:
            continue

        # Get pages for this sequence
        num_pages_for_seq = (seq_len + page_size - 1) // page_size
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)

        # Gather K from pages
        K_paged = K_all[page_indices]  # [num_pages_for_seq, page_size, head_dim]
        K = K_paged.reshape(-1, index_head_dim)[:seq_len]  # [seq_len, head_dim]

        # Query for this batch element
        q_b = q[b]  # [num_heads, head_dim]

        # Compute attention scores
        scores = q_b @ K.T  # [num_heads, seq_len]

        # Apply ReLU (deep_gemm uses ReLU activation)
        scores_relu = torch.relu(scores)  # [num_heads, seq_len]

        # Apply learned weights and sum across heads
        w = weights[b]  # [num_heads]
        weighted_scores = scores_relu * w[:, None]  # [num_heads, seq_len]
        final_scores = weighted_scores.sum(dim=0)  # [seq_len]

        # Select top-K
        actual_topk = min(topk, seq_len)
        _, topk_idx = torch.topk(final_scores, actual_topk)

        page_idx_per_token = topk_idx // page_size
        offset_per_token = topk_idx % page_size
        global_page_idx = page_indices[page_idx_per_token]
        topk_tokens = global_page_idx * page_size + offset_per_token

        topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)


def kernel(
    q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices
):
    run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices)
