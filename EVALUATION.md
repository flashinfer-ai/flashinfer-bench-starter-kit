# Evaluation

This document describes how we evaluate submissions for the [MLSys 2026 NVIDIA Track: FlashInfer AI Kernel Generation Contest](http://mlsys26.flashinfer.ai/).

## Environment

| Field | Value |
|---|---|
| Docker image | `flashinfer/flashinfer-ci-cu132:20260401-2c675fb` |
| Hardware | Bare-metal NVIDIA B200 (sm_100a) |
| GPU clocks | Locked to max (`nvidia-smi -ac 3996,1965`) |
| CUDA | 13.2 |
| Python | 3.12 |
| PyTorch | 2.12.0+cu132 |
| Triton | 3.6.0 |

Packages inside the container:
- FlashInfer (latest main, built from source)
- FlashInfer-Bench (latest main, built from source)
- `cupti-python` for accurate GPU timing
- `deep-gemm`
- `helion`
- `mlc-ai-tirx-cu130` (TVM)
- `nvidia-cutlass-dsl` (CuTe DSL)
- Contest dataset from https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest

## Evaluation Pipeline

### Collect Submissions

We scan each registered team's GitHub repo for git tags:

- For **multiple tags targeting the same definition**, only the latest tag is evaluated.
- Tags targeting **different definitions** (e.g., GDN decode + GDN prefill) are all evaluated.
- Private repos are cloned via `flashinfer-bot`. Make sure you've granted read access (Repo → Settings → Collaborators → Add `flashinfer-bot`).

For each qualifying tag we checkout the tag and read `config.toml` to determine the track and build configuration.

### Run Evaluation

Each track is evaluated **in parallel** on B200 with locked GPU clocks. Each solution runs in an **isolated subprocess**

Per-track commands:

```bash
# MoE
flashinfer-bench run \
  --local ./contest-dataset \
  --definitions moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 \
  --save-results --use-isolated-runner --log-level INFO --resume --timeout 300 \
  --atol 1 --rtol 0.3 --required-matched-ratio 0.9

# DSA Attention
flashinfer-bench run \
  --local ./contest-dataset \
  --definitions dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 \
  --save-results --use-isolated-runner --log-level INFO --resume --timeout 300

# DSA Indexer
flashinfer-bench run \
  --local ./contest-dataset \
  --definitions dsa_topk_indexer_fp8_h64_d128_topk2048_ps64 \
  --save-results --use-isolated-runner --log-level INFO --resume --timeout 300

# GDN Decode
flashinfer-bench run \
  --local ./contest-dataset \
  --definitions gdn_decode_qk4_v8_d128_k_last \
  --save-results --use-isolated-runner --log-level INFO --resume --timeout 300

# GDN Prefill
flashinfer-bench run \
  --local ./contest-dataset \
  --definitions gdn_prefill_qk4_v8_d128_k_last \
  --save-results --use-isolated-runner --log-level INFO --resume --timeout 300 \
  --warmup-runs 1 --iterations 5 --num-trials 3
```


## FlashInfer Baselines

The contest dataset includes FlashInfer baseline solutions under `solutions/baseline/` for reference.

| Track | Solution Name | Definition |
|---|---|---|
| MoE | `flashinfer_wrapper_9sdjf3` | `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048` |
| DSA Attention | `flashinfer_wrapper_5af199` | `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64` |
| DSA Indexer | `flashinfer_deepgemm_wrapper_2ba145` | `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64` |
| GDN Decode | `flashinfer_wrapper_9b7f1e` | `gdn_decode_qk4_v8_d128_k_last` |
| GDN Prefill | `flashinfer_wrapper_123ca6` | `gdn_prefill_qk4_v8_d128_k_last` |

To run a baseline locally (e.g., GDN Decode):

```bash
flashinfer-bench run \
  --local /path/to/mlsys26-contest \
  --definitions gdn_decode_qk4_v8_d128_k_last \
  --solutions flashinfer_wrapper_9b7f1e \
  --use-isolated-runner --timeout 300
```

## Schedule

Bi-weekly evaluations are provided to help participants track their progress. These results are **not** counted toward the final evaluation — only the final submission at the kernel submission deadline will be scored.

Make sure your latest submission tag is pushed before each evaluation date.

| Date | Event |
|---|---|
| Feb 15, 2026 | Registration deadline |
| Feb 27, 2026 | First bi-weekly evaluation |
| Mar 13, 2026 | Second bi-weekly evaluation |
| Mar 27, 2026 | Third bi-weekly evaluation |
| Apr 12, 2026 | Fourth bi-weekly evaluation |
| Apr 18, 2026 | Extra validation round (compilation/correctness check only, no ranking) |
| **Apr 24, 2026** | **Final kernel submission deadline (11:59 PM AoE)** |
