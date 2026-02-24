# [FlashInfer AI 算子生成竞赛 @ MLSys 2026](http://mlsys26.flashinfer.ai/)

与人类和/或 AI 智能体一起，在 NVIDIA Blackwell GPU 上为最先进的 LLM 架构创建高性能 GPU 算子（Kernels）。

---

<p align="center">
  <a href="https://www.nvidia.com"><img src="images/nvidia-logo.svg" alt="NVIDIA" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://modal.com"><img src="images/modal-logo.png" alt="Modal" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://mlsys.org"><img src="images/mlsys-logo.svg" alt="MLSys" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/flashinfer-ai/flashinfer"><img src="images/flashinfer-logo.png" alt="FlashInfer" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/flashinfer-ai/flashinfer-bench"><img src="images/fib_logo.png" alt="FlashInfer-Bench" height="50"/></a>
</p>

---

[FlashInfer-Bench](https://github.com/flashinfer-ai/flashinfer-bench) 是我们用于评估您 AI 生成算子的官方框架。

## 更新

* 2026.02.05: 定义和工作负载（Workloads）的完整数据集已在 [HuggingFace](https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest) 发布。

## 竞赛赛道

本次竞赛包含三个赛道，每个赛道都针对一个关键的 LLM 操作：

| 赛道 | 描述 |
|-------|-------------|
| **fused_moe** | 融合专家混合（Fused Mixture-of-Experts）算子，用于高效的专家路由和计算 |
| **sparse_attention** | 用于长上下文推理的稀疏注意力（Sparse Attention）机制 |
| **gated_delta_net** | 用于高效状态更新的门控增量网络（Gated Delta Network）操作 |

对于您想要参加的**每个赛道，请分别 Fork 一次此模板**（每个赛道使用独立的仓库）。

## 快速入门

### 1. Fork 此模板

点击 "Use this template" 或 Fork 本仓库来创建您的解决方案仓库。

### 2. 安装依赖

```bash
conda create -n fi-bench python=3.12
conda activate fi-bench
pip install flashinfer-bench modal
```

### 3. 下载 TraceSet

我们提供 [FlashInfer-Trace 格式](https://bench.flashinfer.ai/docs/flashinfer-trace) 的算子定义和工作负载。从 HuggingFace 克隆竞赛数据集：

```bash
git lfs install
git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
```

设置环境变量：

```bash
export FIB_DATASET_PATH=/path/to/flashinfer-trace
```

### 4. 配置您的解决方案

编辑 `config.toml` 以设置您的赛道和团队信息：

```toml
[solution]
name = "my-team-solution-v1"      # 解决方案名称
definition = "fused_moe"          # 赛道: fused_moe | sparse_attention | gated_delta_net
author = "team-name"              # 团队/作者名称

[build]
language = "triton"               # triton | cuda
entry_point = "kernel"            # 算子函数名称
```

### 5. 实现您的算子

**对于 Triton:**
编辑 `solution/triton/kernel.py` 进行实现。

**对于 CUDA:**
编辑 `solution/cuda/kernel.cu` 和 `solution/cuda/binding.py` 进行实现。

## 开发工作流

### 打包您的解决方案

根据您的源文件生成 `solution.json`：

```bash
python scripts/pack_solution.py
```

### 运行本地基准测试

在您的本地 GPU 上测试解决方案：

```bash
python scripts/run_local.py
```

要求：本地具有 CUDA 能力的 GPU 以及 `FIB_DATASET_PATH` 环境变量。

### 运行云端基准测试 (Modal)

通过 Modal 在 NVIDIA B200 GPU 上测试您的解决方案：

**一次性设置:**

```bash
modal setup
modal volume create flashinfer-trace
modal volume put flashinfer-trace /path/to/flashinfer-trace
```

**运行基准测试:**

```bash
modal run scripts/run_modal.py
```

## 提交

提交您的解决方案进行评估：

1. 确保您的实现已完成并经过测试
2. 运行 `python scripts/pack_solution.py` 生成 `solution.json`
3. 提交并推送您的更改
4. 为评估标记您的提交（例如，`git tag submission-v1`）

## 项目结构

```
flashinfer-bench-starter-kit/
├── README.md                    # 此文件
├── config.toml                  # 赛道配置（编辑此文件）
├── solution/                    # 解决方案源文件
│   ├── triton/                  # Triton 实现
│   │   └── kernel.py           # 您的 Triton 算子
│   └── cuda/                    # CUDA 实现
│       ├── kernel.cu           # 您的 CUDA 算子
│       └── binding.py          # TVM FFI 绑定
├── scripts/                     # 工具脚本
│   ├── run_local.py            # 本地基准测试运行器
│   ├── run_modal.py            # Modal 云端基准测试运行器
│   └── pack_solution.py        # 将源文件打包为 solution.json
└── images/                      # 赞助商 Logo
```

## 更多资源

### FlashInfer Trace 查看器

FlashInfer Trace 由多个 JSON 对象（定义、工作负载、解决方案和 Trace）组成，其中可能包含大型代码块。为了方便地可视化和检查这些对象，您可以使用 [FlashInfer Trace Viewer](https://bench.flashinfer.ai/viewer)。只需将任何 FlashInfer Trace JSON 粘贴到查看器中，即可获得友好的结构化内容视图。

### 解决方案处理 API

```python
from flashinfer_bench import BuildSpec
from flashinfer_bench.agents import pack_solution_from_files, extract_solution_to_files

# 将源文件打包为 Solution 对象
spec = BuildSpec(
    language="triton",  # 或 "cuda"
    target_hardware=["cuda"],
    entry_point="my_kernel",
)
solution = pack_solution_from_files(
    path="./my_solution_dir",
    spec=spec,
    name="my_solution_v1",
    definition="fused_moe",
    author="your_name",
)

# 将 Solution 提取到工作目录中的文件
extract_solution_to_files(solution, "./output_dir")
```

### 运行 Sanitizer

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

### NCU 性能剖析 (NCU Profiling)

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

### 列出可用工具

```python
from flashinfer_bench.agents import get_all_tool_schemas

schemas = get_all_tool_schemas()
# 返回与 OpenAI 兼容的函数 schema 列表
```

## 注意事项

### 目标传递风格 (Destination Passing Style, DPS)

FlashInfer-Bench 默认使用目标传递风格 (DPS)，其中输入和输出都作为函数参数传递。DPS 避免了测量张量分配开销，从而产生更准确的性能数据。我们建议尽可能使用 DPS，因为它能产生更好的基准测试结果。

**重要提示：** 避免在算子签名中使用变长输入参数，因为它们将无法通过构建器的验证检查。

如果您的算子使用值返回风格（即返回输出张量而不是写入预分配的张量），请在解决方案的 `spec` 中将 `destination_passing_style` 设置为 `false`：

```json
{
  "name": "my_solution",
  "definition": "gdn_decode_qk4_v8_d128_k_last",
  "author": "my_name",
  "spec": {
    "language": "triton",
    "target_hardware": ["cuda"],
    "entry_point": "kernel.py::my_kernel",
    "dependencies": [],
    "destination_passing_style": false
  },
  "sources": [...]
}
```

**当 DPS 不匹配时的常见错误：**

```
Destination-passing style callable: expected xx parameters, but got xx
```

这可能由于两个原因发生：(1) 您的算子函数签名的参数数量错误，或者 (2) 您的算子使用值返回风格，但解决方案的 `destination_passing_style` 仍默认为 `true`。对于后者，请通过将 `destination_passing_style` 设置为 `false` 来修复。

### CUDA 算子绑定

对于 CUDA 算子实现，我们建议使用 [TVM FFI](https://tvm.apache.org/ffi/) 进行 Python 绑定。`flashinfer_bench.agents` 模块提供 TVM FFI 智能体指令提示以协助开发。

您可以在解决方案的 `spec` 中设置 `binding` 字段来指定 C++ 绑定类型。如果未指定，默认为 `"tvm-ffi"`。支持的值有：`"tvm-ffi"`, `"torch"`。

```json
{
  "name": "my_cuda_solution",
  "definition": "gdn_decode_qk4_v8_d128_k_last",
  "author": "my_name",
  "spec": {
    "language": "cuda",
    "target_hardware": ["cuda"],
    "entry_point": "kernel.cu::my_kernel",
    "dependencies": [],
    "binding": "torch"
  },
  "sources": [...]
}
```
