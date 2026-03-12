# Frequently Asked Questions

Last updated: March 12, 2026

---

## Submission & Evaluation

**Q: How do I submit my solution?**

Create a starter-kit repo (one per track), push a git tag (`git tag submission-v1 && git push origin submission-v1`), and if your repo is private, grant read access to **flashinfer-bot** (Repo → Settings → Collaborators → Add people). We evaluate the most recent tag in each repo. When a track has multiple definitions (e.g., GDN decode + prefill), place each submission in a top-level subfolder named after the definition name.

**Q: I used "Use this template" instead of forking. Is that okay?**

Yes. Both template and fork are fine, as long as your repo follows the starter-kit structure.

**Q: How do I share my repo URL with organizers?**

Reply in the Discord thread with your repo URL(s) and team name, email `mlsys26-contest-contact@nvidia.com`, or DM an organizer on Discord.

**Q: Is there a public leaderboard?**

No. We run **bi-weekly evaluations** on bare-metal B200 GPUs and notify teams individually with their performance numbers and ranking via email.

**Q: How are workloads scored?**

The final score for a definition is the **arithmetic mean** of speedups across all its workloads.

**Q: What does "speedup" mean exactly?**

Speedup is measured relative to the **definition reference** (a simple Python reference implementation), not the optimized FlashInfer baseline. The reference is intentionally kept simple to define correctness.

**Q: For Track C (GDN), how are decode and prefill weighted?**

Decode and prefill are separate definitions, scored and ranked independently.

**Q: What is the maximum team size?**

Maximum **5 members** per team.

---

## Official Evaluation Environment

**Q: What CUDA / Triton / PyTorch versions are used in official evaluation?**

The specific versions will be announced later. The official environment will include torch, triton, tilelang, CuTe-DSL, CuTile, and other packages. We will open a link for teams to request additional libraries.

**Q: Is the final evaluation done on Modal?**

No. The final evaluation runs on **bare-metal B200 GPUs** with locked clock frequencies. Scores on Modal are for development reference only.

**Q: Can I use `torch.utils.cpp_extension` to compile my CUDA solution?**

Yes. The flashinfer-bench TorchBuilder uses `torch.utils.cpp_extension` under the hood. You can also call it directly in a Python submission.

**Q: Can I pass custom compile flags for CUDA C++ submissions?**

The builders currently do not support custom compile flags. As a workaround, submit a Python solution and compile the CUDA kernel yourself within the code (using `torch.utils.cpp_extension.load()` or `tvm_ffi.cpp.load()`). We will consider adding compile flag support in a future update.

**Q: Can I install additional Python packages?**

The `BuildSpec` has a `dependencies` field, but builder-side support is still being finalized. For Python packages, we will use the packages in our official evaluation environment (versions to be announced). We will open a link for teams to request additional libraries.

---

## CuTe-DSL / CuTile

**Q: Can I use CuTe-DSL or CuTile?**

Yes. The competition supports multiple languages including CUDA, Triton, CuTe-DSL, CuTile, Tilelang, and more. All of these will be available in the official evaluation environment.

---

## GPU Resources & Profiling

**Q: Can I use NCU (Nsight Compute) on Modal?**

NCU is not currently available on Modal. We are still working with Modal to find a solution.

**Q: Does compute-sanitizer work on Modal?**

Same situation — still working with Modal to find a solution.

**Q: I haven't received my Modal credits / B200 access. What should I do?**

We are currently running out of credits and looking into alternative solutions. Please email `mlsys26-contest-contact@nvidia.com` to check your status.

**Q: Is Modal's B200 sm100 or sm100a?**

Modal B200 instances are **sm100**.

---

## Known Issues

**Q: GDN Prefill reference implementation fails correctness — output explodes on longer sequences.**

This is a known issue. The reference implementation produces numerically unstable output on longer sequences. We are fixing the workload dataset and will release the corrected version soon.

**Q: `flashinfer.fused_moe.trtllm_fp8_block_scale_moe` produces INCORRECT_NUMERICAL with very large errors.**

This is a known issue. We will fix the benchmark system and workload data. In the meantime, you can start from the definition's reference implementation.

**Q: My FP8 kernel shows very high max_relative_error but is still marked correct. Is the final evaluation stricter?**

The evaluation uses relaxed tolerance settings for FP8 kernels, allowing a certain percentage of elements to not match exactly. A high max error while still being marked correct is expected behavior as long as the majority of elements are within tolerance. The final evaluation will use the same criteria.

**Q: DSA reference upcasts to fp32 — impossible to match exact correctness with fp8 MMA.**

The reference uses fp32 to ensure numerical correctness of the baseline. The evaluation tolerance settings already account for fp8 precision loss — exact matching is not required.

**Q: Are there numerical issues with DSA (sparse attention)?**

We are aware of and investigating DSA correctness concerns. The updated benchmark system and workloads will address these.

**Q: Benchmark runtimes seem inflated (issue #195).**

Fixed in the latest version of flashinfer-bench ([PR #196](https://github.com/flashinfer-ai/flashinfer-bench/pull/196)). Timing now uses CUPTI activity tracing for hardware-level precision. Please update to the latest version.

---

## Other

**Q: What is `binding.py` for? Isn't `PYBIND11_MODULE` enough?**

`binding.py` is for TVM FFI bindings. `PYBIND11_MODULE` is the PyTorch extension approach, which also works. Both backends (TVM FFI and Torch) are supported.

**Q: DSA currently only has decode shapes. Will there be prefill?**

Yes, both decode and prefill shapes will be available.

**Q: Track C — HuggingFace dataset uses qk4_v8 but the website uses qk16_v32. Which to target?**

Please target the specifications on the contest website [mlsys26.flashinfer.ai](http://mlsys26.flashinfer.ai). The qk4_v8 in the HuggingFace dataset is an earlier version and may be updated.

**Q: Is FlashInfer available for sm120 / Blackwell Pro 6000?**

The competition targets B200 (sm100) only. sm120 support is outside the scope of this competition.

**Q: My kernel produces no trace (len trace = 0) when running on Modal.**

If the kernel fails to run or does not pass correctness checks, no trace will be generated. Check the log file for error messages (use the `--log-file` parameter).

**Q: I'm getting "Failed to fetch" errors when uploading to Modal.**

This is an intermittent network issue on the Modal platform. Please retry.

**Q: When will implementation baselines (GDN, DSA, etc.) be released?**

Implementation baselines for all kernels will be provided in a subsequent update.

**Q: What's the difference between the agent track and the agent-assisted track?**

The **agent track** requires submitting the agent itself — it must fully reproduce the kernel end-to-end. The **agent-assisted track** allows experts and agents to collaborate; you submit the kernel code. Note: in the agent track, your agent's prompts and database must not contain large portions of the final solution (we will verify manually).
