# DDA5001 Final Project - Part II: LLM Finetuning

本项目是香港中文大学（深圳）数据科学学院 DDA5001 课程最终项目的第二部分，专注于对 **Qwen1.5-0.5B-Chat** 模型进行数学问题求解能力的指令微调（Instruction-Tuning）。

项目提供了一个完整的端到端（End-to-End）流程，涵盖了**数据准备、模型训练、模型推理**和**自动化评估**等关键环节，旨在为学生提供一个清晰、完整的 LLM 微调实践案例。

## 目录

- [项目目标](#项目目标)
- [项目特色](#项目特色)
- [项目结构](#项目结构)
- [工作流](#工作流)
  - [推荐：自动化实验流程](#推荐自动化实验流程)
  - [手动分步执行](#手动分步执行)
- [核心模块详解](#核心模块详解)
- [使用建议](#使用建议)
- [常见问题](#常见问题)
- [许可证](#许可证)

## 项目目标

本项目旨在通过在 `MATH-500` 数据集上微调 **Qwen1.5-0.5B-Chat** 模型，显著提升其解决数学问题的能力。学生需要完成以下核心任务：

1.  **数据处理**：对 `MATH-500` 数据集应用 Chat Template 并进行 Tokenization，将问题-答案对转换为模型可训练的格式，并正确标记 prompt 部分以在训练中忽略其损失。
2.  **模型训练**：使用不同的优化器（**SGD**、**AdamW**、**LoRA**）进行模型微调，并探索不同超参数（学习率、Epoch、LoRA 秩等）对训练效果的影响。
3.  **模型评估**：在 `MATH-500` 测试集上，量化评估微调后模型相较于基础模型的性能提升，并支持多种复杂数学答案格式的准确匹配。

## 项目特色

- **完整的 MLOps 流程**：从数据处理到模型评估，提供了一套完整的脚本，清晰地展示了 LLM 微调的典型工作流。
- **多种优化方法支持**：
  - **AdamW**：传统的全参数微调优化器。
  - **SGD**：标准随机梯度下降优化器。
  - **LoRA (高效参数微调)**：采用 PEFT 技术，仅训练少量适配器参数，大幅降低计算和存储资源消耗。
- **强大的答案评估器**：内置了功能强大的数学答案验证器 (`src/verifier/`)，其逻辑参考了 Hendrycks' MATH、ToRA、PRM800K 等业界标杆，能够进行符号和数值层面的精确匹配，支持分数、百分比、区间、矩阵等多种格式。
- **高效推理**：默认集成 **Flash Attention 2** 优化，显著加速模型生成速度，并支持批量推理。
- **自动化实验管理**：提供 `run_experiments.py` 脚本，可一键运行完整的实验流程，自动循环测试多个超参数组合，并清晰地管理日志与结果。
- **详细的日志与可视化**：训练过程会记录详细的 JSON 日志（包括所有超参数、训练时间、GPU 内存使用等），并能自动生成训练/验证损失曲线图，便于分析和比较不同训练策略。
- **跨平台兼容**：解决了在 Windows 等非 Posix 系统上由 `signal` 模块引发的兼容性问题，确保代码在不同操作系统上都能顺利运行。

## 项目结构

```
.
├── data/                     # 存放由 prepare.py 生成的 .pkl 数据文件
├── result/                   # 存放推理和评估结果
│   └── AdamW_exp/            # 按实验组织的文件夹（例如 AdamW 优化器）
│       └── lr_1e-05_epochs_1/  # 具体的超参数组合
│           ├── answers.jsonl          # 1. 模型生成的答案
│           ├── scored_answers.jsonl   # 2. 评估后的答案（含分数）
│           ├── execution.log          # 3. 完整的运行日志
│           └── loss_curve.png         # 4. 训练损失曲线图
├── src/                      # 核心源代码
│   ├── verifier/             # 数学答案验证模块
│   │   ├── grader.py         # 核心评分逻辑
│   │   └── math_normalize.py # 答案规范化处理
│   ├── prepare.py            # 1. 数据准备脚本
│   ├── finetune.py           # 2. 模型微调脚本
│   ├── rollout.py            # 3. 模型推理（生成答案）脚本
│   ├── evaluate.py           # 4. 自动化评估脚本
│   └── run_experiments.py    # (推荐) 自动化实验运行脚本
├── train_log/                # 存放训练输出（模型权重和日志）
│   └── AdamW_exp/            # 按实验组织的文件夹
│       └── lr_1e-05_epochs_1/
│           ├── training_logs.json  # 详细的训练日志
│           └── ...                 # 模型文件 (Checkpoint)
└── requirements.txt          # 项目依赖
```

## 工作流

### 推荐：自动化实验流程

我们强烈推荐使用 `run_experiments.py` 脚本来自动化整个工作流。该脚本会依次执行**数据准备、模型微调、推理**和**评估**，并能循环测试多种超参数，将每次实验的结果整齐地保存在独立的目录中。

```bash
# 启动自动化实验流程
python src/run_experiments.py
```

- **工作原理**: 该脚本会智能地调用 `prepare.py`, `finetune.py`, `rollout.py`, 和 `evaluate.py`，并自动管理所有文件路径和日志记录。
- **超参数配置**: 你可以在 `run_experiments.py` 脚本内部轻松修改要测试的优化器、学习率和 Epoch 范围。
- **日志模式**: 通过 `LOG_TO_FILE` 变量切换日志模式。`True` (默认) 会将所有输出保存到文件，`False` 则在控制台实时显示。

### 手动分步执行

如果你希望手动控制每个步骤，可以按照以下流程操作。

#### 1. 环境配置
首先，请确保你的环境满足要求。建议在虚拟环境中安装依赖。

```bash
# 安装所有必需的库
pip install -r requirements.txt
```

#### 2. 数据准备
运行 `prepare.py` 脚本。它会从 Hugging Face Hub 下载 `MATH-500` 数据集，应用 Chat Template，进行 Tokenize 和格式化，然后将处理好的数据保存到 `data/` 目录下。

```bash
python src/prepare.py
```
- **调试模式**: 添加 `--debug` 参数可在 5% 的数据子集上快速运行，以验证流程。

#### 3. 模型微调
运行 `finetune.py` 脚本启动模型训练。

```bash
# 示例：使用 LoRA 进行训练，学习率为 1e-4，训练 1 个 epoch，并绘制损失图
python src/finetune.py \
    --optimization_method lora \
    --learning_rate 1e-4 \
    --num_epochs 1 \
    --plot
```
- **核心参数**:
  - `--optimization_method`: 选择优化器，可选 `adam`, `sgd`, `lora`。
  - `--lora_rank`: 当使用 `lora` 时，设置 LoRA 的秩。
  - `--learning_rate`: 设置学习率。
  - `--num_epochs`: 设置训练轮次。
  - `--plot`: 训练结束后自动生成并保存损失曲线图。
- **输出**: 训练完成后，最佳的模型文件和 `training_logs.json` 将保存在 `train_log/` 目录下。

#### 4. 模型推理
训练完成后，运行 `rollout.py` 脚本，在测试集上生成答案。

```bash
# 示例：加载微调后的 LoRA 模型进行推理
python src/rollout.py \
    --lora_path "train_log/LoRA_exp/lr_1e-04_epochs_1" \
    --output_file "result/answers_lora.jsonl"
```
- **性能优化**: 该脚本已通过 `attn_implementation="flash_attention_2"` 进行了优化。
- **输出**: 推理结果将以 JSONL 格式保存在指定的输出文件中。

#### 5. 自动化评估
最后，运行 `evaluate.py` 脚本来评估模型的性能。

```bash
# 示例：对 LoRA 模型的答案进行评分
python src/evaluate.py \
    --input_file "result/answers_lora.jsonl" \
    --output_file "result/scored_answers_lora.jsonl"
```
- **工作原理**: 该脚本会读取 `answers.jsonl`，调用 `verifier` 模块对每条答案进行评分。
- **输出**: 包含分数的评估结果将保存在输出文件中，并在控制台打印最终的准确率。

## 核心模块详解

- **`src/run_experiments.py`**: **（自动化核心）** 串联所有步骤，自动管理文件路径、循环测试超参数，并将每次实验的结果清晰地组织在独立目录中。**强烈推荐使用此脚本。**

- **`src/prepare.py`**: **（数据处理）** 负责将原始问答对转换成模型可以理解的 ChatML 格式，并对 prompt 和 response 进行恰当的 Tokenize 和标签化（labeling），确保 prompt 部分的损失在训练中被忽略。

- **`src/finetune.py`**: **（模型训练）** 核心训练脚本，集成了 `torch`、`transformers` 和 `peft`，实现了完整的训练、验证、日志记录和模型保存逻辑。

- **`src/rollout.py`**: **（模型推理）** 加载基础模型或微调后的模型进行批量推理。已通过 **Flash Attention 2** 进行性能优化。

- **`src/evaluate.py`**: **（性能评估）** 连接推理输出和评分模块，实现了自动化的端到端评估，并输出最终准确率。

- **`src/verifier/`**: **（答案验证器）** 项目的亮点之一。提供了强大的数学答案评分能力，其逻辑借鉴了多个业界领先的开源项目，能够准确判断各种形式（数值、符号、矩阵等）的数学答案是否正确。

## 使用建议

1.  **快速体验 (Debug 模式)**: 使用 `run_experiments.py` 中的 `DEBUG = True` 选项，可以在 5% 的数据子集上快速跑通整个流程，验证环境和代码是否正常。

2.  **对比不同优化器**: 在 `run_experiments.py` 中，可以方便地配置 `optimizers_to_run` 列表，例如 `["adam", "lora"]`，并为它们设置不同的学习率范围，以系统地比较其效果。

3.  **系统的超参数搜索**:
    - 在 `run_experiments.py` 中修改 `learning_rates` 和 `num_epochs_list` 来定义搜索空间。
    - 设置 `LOG_TO_FILE = True` 以避免控制台信息泛滥。
    - 实验结束后，分析 `result/` 目录中各个子文件夹的 `scored_answers.jsonl` 和 `execution.log` 来比较性能。

4.  **性能与资源平衡**:
    - **显存不足时**：优先使用 **LoRA** 微调，它能以极低的显存占用达到接近全量微调的效果。
    - **加速训练**: 在显存允许的情况下，适当增大 `batch_size` 和 `grad_accumulation_steps`。

## 常见问题

**Q: 如何在 GPU 上加速训练?**
**A**: 脚本会自动检测并使用 GPU。请确保已安装 CUDA 兼容的 PyTorch。代码已默认启用 bfloat16 混合精度训练以加速并节省显存。

**Q: LoRA 的秩（rank）应该设置为多少?**
**A**: 通常设置在 8-32 之间。更高的秩意味着更多的可训练参数，可能带来更好的性能，但也会增加显存占用。建议从 8 或 16 开始尝试。

**Q: 如何评估基础模型（未微调）的性能?**
**A**: 运行 `rollout.py` 时不提供 `--lora_path` 参数即可。
   ```bash
   python src/rollout.py --output_file "result/answers_base.jsonl"
   python src/evaluate.py --input_file "result/answers_base.jsonl" --output_file "result/scored_answers_base.jsonl"
   ```

**Q: 训练过程中如何查看实时进度?**
**A**: 将 `run_experiments.py` 中的 `LOG_TO_FILE` 设置为 `False`，所有日志（包括 `transformers` 的进度条）将直接打印在控制台。

## 许可证

该项目的设计和代码实现参考了多个优秀的开源项目，特此感谢：
- [Hendrycks' MATH](https://github.com/hendrycks/math)
- [ToRA (Tool-integrated Reasoning Agent)](https://github.com/microsoft/ToRA)
- [PRM800K (Process Reward Models)](https://github.com/openai/prm800k)

本项目仅供学术研究和教学使用。具体许可信息见相应模块的文件头注释。