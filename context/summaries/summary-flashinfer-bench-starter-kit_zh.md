# 项目摘要：FlashInfer-Bench 入门套件 (FlashInfer-Bench Starter Kit)

## 目的

**FlashInfer-Bench 入门套件** 是一个模板仓库，专为 **FlashInfer AI 算子生成竞赛 @ MLSys 2026** 的参赛者设计。其目的是为开发、测试和提交针对最先进 LLM 架构的高性能 GPU 算子（Kernels）（特别针对 NVIDIA Blackwell GPU）提供一个结构化环境。它支持人工编写和 AI 生成的算子。

## 概念

*   **竞赛赛道 (Competition Tracks)**：三个特定的优化目标：
    *   `fused_moe`：融合专家混合（Fused Mixture-of-Experts）路由和计算。
    *   `sparse_attention`：针对长上下文的稀疏注意力（Sparse Attention）机制。
    *   `gated_delta_net`：门控增量网络（Gated Delta Network）操作。
*   **FlashInfer-Trace**：一种用于定义算子定义和工作负载（Workloads）的格式。
*   **TraceSet**：竞赛提供的数据集，包含要优化的特定算子定义和工作负载。
*   **解决方案 (Solution)**：包含算子实现（源代码）及元数据/配置的包。
*   **目标传递风格 (Destination Passing Style, DPS)**：一种编程风格，其中输出张量预先分配并作为参数传递给算子函数，以避免内存分配带来的测量噪声。这是默认且推荐的风格。
*   **Modal**：一个用于在高端 GPU（NVIDIA B200）上运行基准测试的云平台，这些 GPU 可能在本地无法使用。
*   **Triton / CUDA**：两种受支持的算子实现语言/框架。
*   **TVM FFI**：推荐的将 CUDA 算子绑定到 Python 的机制。

## 使用方法

### 前提条件
*   Python 3.12
*   Conda（推荐）
*   NVIDIA GPU（用于本地测试）
*   Modal 账户（用于云端基准测试）

### 安装
1.  **Fork 仓库**：从该模板为您选择的赛道创建一个新仓库。
2.  **环境配置**：
    ```bash
    conda create -n fi-bench python=3.12
    conda activate fi-bench
    pip install flashinfer-bench modal
    ```
3.  **下载数据集**：
    ```bash
    git lfs install
    git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
    export FIB_DATASET_PATH=/path/to/mlsys26-contest
    ```

### 基本工作流
1.  **配置**：编辑 `config.toml` 以指定赛道、团队名称和构建设置（语言、入口点）。
2.  **实现**：在 `solution/triton/kernel.py`（针对 Triton）或 `solution/cuda/kernel.cu`（针对 CUDA）中编写算子代码。
3.  **打包**：生成提交文件 `solution.json`。
    ```bash
    python scripts/pack_solution.py
    ```
4.  **基准测试**：
    *   本地：`python scripts/run_local.py`
    *   云端 (Modal)：`modal run scripts/run_modal.py`

## 内部机制

该项目结构是一个标准的基于 Python 的开发工具包，具有用于源代码和实用程序脚本的特定目录。核心逻辑围绕 `flashinfer-bench` 库展开，该库负责算子的打包、构建和基准测试。

### 架构
*   **配置**：`config.toml` 是解决方案元数据的唯一事实来源。
*   **源代码**：`solution/` 目录隔离了用户的实现。
*   **构建系统**：`scripts/` 中的脚本使用 `flashinfer_bench` 将源代码打包成标准化的 `solution.json` 格式，以便竞赛系统进行评估。
*   **评估**：基准测试脚本加载解决方案，编译算子（如有必要），并针对 `FIB_DATASET_PATH` 指向的 `TraceSet` 中定义的工作负载运行它。

### 目录树

```
flashinfer-bench-starter-kit/
├── README.md                    # 主要文档和指南
├── config.toml                  # 赛道和团队信息的配置文件
├── solution/                    # 用户实现目录
│   ├── triton/                  # Triton 实现文件
│   │   └── kernel.py           # Triton 算子的入口点
│   └── cuda/                    # CUDA 实现文件
│       ├── kernel.cu           # CUDA 算子源代码
│       └── binding.py          # TVM FFI Python 绑定
├── scripts/                     # 工作流实用脚本
│   ├── run_local.py            # 在本地 GPU 上运行基准测试的脚本
│   ├── run_modal.py            # 在 Modal 云端运行基准测试的脚本
│   └── pack_solution.py        # 将解决方案打包为 JSON 的脚本
└── images/                      # README 中使用的图像
```

## 参考资料

*   [**README.md**](../../README.md)：此入门套件的主要文档文件。
*   [**FlashInfer-Bench 仓库**](https://github.com/flashinfer-ai/flashinfer-bench)：官方框架仓库。
*   [**FlashInfer Trace 文档**](https://bench.flashinfer.ai/docs/flashinfer-trace)：有关定义和工作负载所用数据格式的详细信息。
*   [**FlashInfer Trace 查看器**](https://bench.flashinfer.ai/viewer)：用于可视化 Trace JSON 对象的工具。
*   [**竞赛数据集 (HuggingFace)**](https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest)：竞赛的官方数据集。
