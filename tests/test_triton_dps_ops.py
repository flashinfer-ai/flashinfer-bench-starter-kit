import importlib.util
import math
from pathlib import Path

import torch


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPARSE_PATH = PROJECT_ROOT / "solution" / "triton" / "dsa_sparse_attention.py"
TOPK_PATH = PROJECT_ROOT / "solution" / "triton" / "dsa_topk_indexer.py"


def test_sparse_attention_run_uses_dps_buffers():
    mod = _load_module("dsa_sparse_attention", SPARSE_PATH)

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: Running on CPU, Triton kernel might fail if it expects GPU pointers")

    num_tokens = 2
    num_heads = 16
    head_dim_ckv = 512
    head_dim_kpe = 64
    page_size = 64
    topk = 2048

    q_nope = torch.randn(num_tokens, num_heads, head_dim_ckv, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(num_tokens, num_heads, head_dim_kpe, dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn(1, page_size, head_dim_ckv, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn(1, page_size, head_dim_kpe, dtype=torch.bfloat16, device=device)
    sparse_indices = torch.full((num_tokens, topk), -1, dtype=torch.int32, device=device)
    sparse_indices[0, :3] = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    sm_scale = 1.0 / math.sqrt(192.0)

    output = torch.full((num_tokens, num_heads, head_dim_ckv), 7, dtype=torch.bfloat16, device=device)
    lse = torch.full((num_tokens, num_heads), 123.0, dtype=torch.float32, device=device)

    result = mod.run(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        sm_scale,
        output,
        lse,
    )

    assert result is None

    valid_idx = sparse_indices[0, :3].long()
    kc = ckv_cache.reshape(-1, head_dim_ckv).float()[valid_idx]
    kp = kpe_cache.reshape(-1, head_dim_kpe).float()[valid_idx]
    qn = q_nope[0].float()
    qp = q_pe[0].float()
    logits_scaled = ((qn @ kc.T) + (qp @ kp.T)) * sm_scale
    expected_lse0 = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
    expected_out0 = (torch.softmax(logits_scaled, dim=-1) @ kc).to(torch.bfloat16)

    assert torch.allclose(lse[0], expected_lse0, rtol=1e-4, atol=1e-4)
    assert torch.allclose(output[0], expected_out0, rtol=5e-2, atol=5e-2)
    assert torch.all(output[1] == 0)
    assert torch.isneginf(lse[1]).all()


def test_topk_indexer_run_uses_dps_buffer():
    mod = _load_module("dsa_topk_indexer", TOPK_PATH)

    torch.manual_seed(0)
    batch_size = 2
    num_heads = 64
    head_dim = 128
    num_pages = 3
    page_size = 64
    topk = 2048

    q_index_fp8 = torch.randn(batch_size, num_heads, head_dim, dtype=torch.float32)
    k_index_cache_fp8 = torch.zeros(num_pages, page_size, 1, 132, dtype=torch.int8)
    weights = torch.randn(batch_size, num_heads, dtype=torch.float32)
    seq_lens = torch.tensor([20, 0], dtype=torch.int32)
    block_table = torch.tensor([[1, 2, 0, 0], [0, 0, 0, 0]], dtype=torch.int32)
    topk_indices = torch.full((batch_size, topk), -99, dtype=torch.int32)

    result = mod.run(
        q_index_fp8,
        k_index_cache_fp8,
        weights,
        seq_lens,
        block_table,
        topk_indices,
    )

    assert result is None

    assert (topk_indices[0, :20] >= 0).all()
    assert (topk_indices[0, :20] < num_pages * page_size).all()
    assert (topk_indices[0, 20:] == -1).all()
    assert (topk_indices[1] == -1).all()


if __name__ == "__main__":
    print("Running test_sparse_attention_run_uses_dps_buffers...")
    test_sparse_attention_run_uses_dps_buffers()
    print("test_sparse_attention_run_uses_dps_buffers passed!")

    print("Running test_topk_indexer_run_uses_dps_buffer...")
    test_topk_indexer_run_uses_dps_buffer()
    print("test_topk_indexer_run_uses_dps_buffer passed!")
