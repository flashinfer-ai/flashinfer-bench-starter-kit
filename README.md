# [FlashInfer AI Kernel Generation Contest @ MLSys 2026](http://mlsys26.flashinfer.ai/)

Create high-performance GPU kernels for state-of-the-art LLM architectures on NVIDIA Blackwell GPUs with humans and/or AI agents.

---

<p align="center">
  <a href="https://www.nvidia.com"><img src="images/nvidia-logo.svg" alt="NVIDIA" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://modal.com"><img src="images/modal-logo.png" alt="Modal" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://mlsys.org"><img src="images/mlsys-logo.svg" alt="MLSys" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/flashinfer-ai/flashinfer"><img src="images/flashinfer-logo.png" alt="FlashInfer" height="50"/></a>
</p>

---

## Quick Setup

### Installation

```bash
conda create -n fi-bench python=3.12
conda activate fi-bench
pip install flashinfer-bench modal
```

### Download the TraceSet

Clone the competition dataset from HuggingFace:

```bash
git lfs install
git clone https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace
```

This contains definitions, workloads, and reference solutions needed for benchmarking.

Set the environment variable to point to the dataset:

```bash
export FIB_DATASET_PATH=/path/to/flashinfer-trace
```

### Modal Setup (for cloud GPU access)

1. Authenticate with Modal:

```bash
modal setup
```

2. Create a Modal Volume and upload the dataset (one-time setup):

```bash
modal volume create flashinfer-trace
modal volume put flashinfer-trace /path/to/flashinfer-trace
```

This uploads the dataset to a persistent Modal Volume for fast access during benchmarks.

## Running Benchmarks

This starter kit includes two example scripts to help you benchmark your kernel solutions.

**Note:** Edit the script to specify your solution path and configure benchmark parameters (warmup runs, iterations, trials).

### Cloud Evaluation (`run_modal.py`)

Run benchmarks on NVIDIA B200 GPUs in the cloud via Modal.

**Requirements:**

- Modal account (authenticated via `modal setup`)
- Dataset uploaded to Modal Volume (see Modal Setup section above)

**Usage:**

```bash
modal run run_modal.py
```

### Local Evaluation (`run_local.py`)

Run benchmarks on your local GPU. This script loads a solution from a JSON file and evaluates it against all workloads in the dataset using your local CUDA-capable GPU.

**Requirements:**

- Local CUDA-capable GPU
- `FIB_DATASET_PATH` environment variable set to your local dataset path

**Usage:**

```bash
python run_local.py
```

## Additional Resources

The `flashinfer_bench.agents` module provides tools to facilitate kernel development.

### Solution Handling

```python
from flashinfer_bench import BuildSpec
from flashinfer_bench.agents import pack_solution_from_files, extract_solution_to_files

# Pack source files into a Solution object
spec = BuildSpec(
    language="triton",  # or "cuda"
    target_hardware=["cuda"],
    entry_point="my_kernel",
)
solution = pack_solution_from_files(
    path="./my_solution_dir",
    spec=spec,
    name="my_solution_v1",
    definition="rmsnorm",
    author="your_name",
)

# Extract a Solution to files in a working directory
extract_solution_to_files(solution, "./output_dir")
```

### Running Sanitizers

```python
from flashinfer_bench.agents import flashinfer_bench_run_sanitizer

output = flashinfer_bench_run_sanitizer(
    solution=solution,
    workload=workload,
    sanitizer_types=["memcheck", "racecheck", "synccheck", "initcheck"],
    timeout=300,
)
print(output)
```

### NCU Profiling

```python
from flashinfer_bench.agents import flashinfer_bench_run_ncu

output = flashinfer_bench_run_ncu(
    solution=solution,
    workload=workload,
    set="detailed",
    page="details",
    timeout=120,
)
print(output)
```

### List Available Tools and Descriptions

```python
from flashinfer_bench.agents import get_all_tool_schemas

schemas = get_all_tool_schemas()
# Returns list of OpenAI-compatible function schemas
```

## Additional Notes

### Kernel Signature Requirements

When implementing kernels using Destination Passing Style (DPS), ensure you specify the kernel signature type in your `BuildSpec` and adjust the build configuration accordingly.

**Important:** Avoid using variadic input arguments in your kernel signatures, as they will fail the builder validation check.

### CUDA Kernel Bindings

For CUDA kernel implementations, we recommend using [TVM FFI](https://tvm.apache.org/ffi/) for Python bindings. The `flashinfer_bench.agents` module provides TVM FFI agent instruction prompts to assist with development.
