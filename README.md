# DDA5001 Final Project - Part II: LLM Finetuning

本项目是香港中文大学（深圳）数据科学学院 DDA5001 课程最终项目的第二部分，专注于对 Qwen-3-0.6B 模型进行数学问题求解能力的指令微调（Instruction-Tuning）。项目提供了一个完整的端到端（End-to-End）流程，涵盖了数据准备、模型训练、模型推理和自动化评估等关键环节。

## 项目目标

本项目旨在通过在 `Math500` 数据集上微调 Qwen-3-0.6B-Base 模型，来提升其解决数学问题的能力。学生需要完成以下核心任务：
1.  **数据处理**：对 `Math500` 数据集应用 Chat Template 并进行 Tokenization。
2.  **模型训练**：使用不同的优化器（SGD, AdamW, LoRA）进行模型微调，并探索不同超参数对训练效果的影响。
3.  **模型评估**：在 `Math500` 测试集上评估微调后模型与基础模型的性能差异。

## 项目特色

- **完整的 MLOps 流程**: 从数据处理到模型评估，提供了一套完整的脚本，清晰地展示了 LLM 微调的典型工作流。
- **高效微调**: 采用 PEFT (Parameter-Efficient Fine-Tuning) 中的 LoRA (Low-Rank Adaptation) 技术，显著降低了训练所需的计算资源。
- **强大的答案评估器**: 内置了一个复杂的数学答案验证器 (`verifier`)，能够进行符号和数值层面的精确匹配，支持分数、百分比、区间、矩阵等多种格式。
- **跨平台兼容**: 解决了在 Windows 等非 Posix 系统上由 `signal` 模块引发的兼容性问题，确保代码在不同操作系统上都能顺利运行。
- **详细的日志与可视化**: 训练过程会记录详细的日志（包括超参数），并能生成训练/验证损失曲线图，便于分析和比较不同训练策略的效果。

## 项目结构

```
.
├── data/                     # 存放由 prepare.py 生成的 .pkl 数据文件
├── result/                   # 存放推理和评估结果
│   ├── answer.jsonl          # rollout.py 生成的模型答案
│   └── scored_answer.jsonl   # evaluate.py 生成的评分结果
├── src/                      # 核心源代码
│   ├── verifier/             # 数学答案验证模块
│   │   ├── __init__.py
│   │   ├── grader.py
│   │   └── math_normalize.py
│   ├── prepare.py            # 数据准备脚本
│   ├── finetune.py           # 模型微调脚本
│   ├── rollout.py            # 模型推理（生成答案）脚本
│   └── evaluate.py           # 自动化评估脚本
├── train_log/                # 存放训练输出
│   └── out-instruction-tuning/ # 微调后的 LoRA 适配器和训练日志
└── requirements.txt          # 项目依赖
```

## 工作流

### 1. 环境配置

首先，请确保你的环境满足要求。建议在虚拟环境中安装依赖。

```bash
# 安装所有必需的库
pip install -r requirements.txt
```

### 2. 数据准备

运行 `prepare.py` 脚本，它会从 Hugging Face Hub 下载 `ricdomolm/MATH-500` 数据集，应用 Chat Template，进行 Tokenize 和格式化，然后将处理好的数据保存到 `data/` 目录下。

```bash
python src/prepare.py
```
- **调试模式**: 添加 `--debug` 参数可以在 5% 的数据子集上快速运行，以验证流程。

### 3. 模型微调

运行 `finetune.py` 脚本来启动模型训练。该脚本会加载预处理好的数据，并使用指定的优化器（默认为 AdamW，可通过参数选择 LoRA 或 SGD）对 Qwen-3-0.6B 模型进行微调。

```bash
# 使用 LoRA 进行训练并绘制损失图
python src/finetune.py --optimization_method lora --plot
```
- **核心参数**:
  - `--optimization_method`: 选择优化器，可选 `adam`, `sgd`, `lora`。
  - `--lora_rank`: 当使用 `lora` 时，设置 LoRA 的秩，默认为 8。
  - `--learning_rate`: 设置学习率，默认为 2e-5。
  - `--num_epochs`: 设置训练轮次，默认为 1。
  - `--plot`: 训练结束后自动生成并显示损失曲线图，并保存到 `result/loss_curve.png`。
- **输出**: 训练完成后，最佳的模型适配器（或完整模型）和 `training_logs.json` 日志文件将保存在 `train_log/out-instruction-tuning/` 目录下。

### 4. 模型推理

训练完成后，运行 `rollout.py` 脚本，它会加载基础模型和微调好的 LoRA 适配器，在 MATH-500 测试集上生成答案。

```bash
python src/rollout.py
```
- **工作原理**: 该脚本将基础模型与 LoRA 权重合并，然后在测试集上进行批量推理。
- **输出**: 推理结果（包含问题、模型答案、标准答案）将以 JSONL 格式保存在 `result/answer.jsonl` 文件中。

### 5. 自动化评估

最后，运行 `evaluate.py` 脚本来评估模型的性能。

```bash
python src/evaluate.py
```
- **工作原理**: 该脚本会读取 `result/answer.jsonl`，并调用 `verifier` 模块对每一条答案进行评分。
- **输出**: 包含分数和标准化答案的详细评估结果将保存在 `result/scored_answer.jsonl` 中，并在控制台打印最终的准确率。

## 核心模块详解

- **`src/prepare.py`**: 负责数据处理。它将原始的问答对转换成 Qwen 模型可以理解的 ChatML 格式，并对 prompt 和 response 进行恰当的 Tokenize 和标签化（labeling），其中 prompt 部分的损失在训练中会被忽略。
- **`src/finetune.py`**: 核心训练脚本。它集成了 `torch`、`transformers` 和 `peft`，实现了完整的训练、验证和保存逻辑。
- **`src/rollout.py`**: 推理脚本。演示了如何加载 LoRA 适配器并进行批量推理，生成用于评估的答案。
- **`src/evaluate.py`**: 评估脚本。它连接了推理输出和评分模块，实现了自动化的端到端评估。
- **`src/verifier/`**: 项目的亮点之一。该模块提供了强大的数学答案评分能力，其逻辑借鉴了多个业界领先的开源项目（如 Hendrycks' MATH、ToRA、PRM800K），能够准确判断各种形式的数学答案是否正确。
