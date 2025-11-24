# DDA5001 Final Project - Part II: LLM Finetuning



本项目是香港中文大学（深圳）数据科学学院 DDA5001 课程最终项目的第二部分，专注于对 Qwen-3-0.6B 模型进行数学问题求解能力的指令微调（Instruction-Tuning）。项目提供了一个完整的端到端（End-to-End）流程，涵盖了数据准备、模型训练、模型推理和自动化评估等关键环节。本项目是香港中文大学（深圳）数据科学学院 DDA5001 课程最终项目的第二部分，专注于对 Qwen-3-0.6B 模型进行数学问题求解能力的指令微调（Instruction-Tuning）。项目提供了一个完整的端到端（End-to-End）流程，涵盖了数据准备、模型训练、模型推理和自动化评估等关键环节。



## 项目目标## 项目目标



本项目旨在通过在 `MATH-500` 数据集上微调 Qwen-3-0.6B-Base 模型，来提升其解决数学问题的能力。学生需要完成以下核心任务：本项目旨在通过在 `Math500` 数据集上微调 Qwen-3-0.6B-Base 模型，来提升其解决数学问题的能力。学生需要完成以下核心任务：

1.  **数据处理**：对 `MATH-500` 数据集应用 Chat Template 并进行 Tokenization，将问题-答案对转换为 ChatML 格式，并正确标记 prompt 部分以在训练中忽略其损失。1.  **数据处理**：对 `Math500` 数据集应用 Chat Template 并进行 Tokenization。

2.  **模型训练**：使用不同的优化器（SGD、AdamW、LoRA）进行模型微调，并探索不同超参数（学习率、Epoch、LoRA秩等）对训练效果的影响。2.  **模型训练**：使用不同的优化器（SGD, AdamW, LoRA）进行模型微调，并探索不同超参数对训练效果的影响。

3.  **模型评估**：在 `MATH-500` 测试集上评估微调后模型与基础模型的性能差异，支持多种数学答案格式的准确匹配。3.  **模型评估**：在 `Math500` 测试集上评估微调后模型与基础模型的性能差异。



## 项目特色## 项目特色



- **完整的 MLOps 流程**: 从数据处理到模型评估，提供了一套完整的脚本，清晰地展示了 LLM 微调的典型工作流，包括详细的日志记录和错误处理。- **完整的 MLOps 流程**: 从数据处理到模型评估，提供了一套完整的脚本，清晰地展示了 LLM 微调的典型工作流。

- **三种优化方法支持**: - **高效微调**: 采用 PEFT (Parameter-Efficient Fine-Tuning) 中的 LoRA (Low-Rank Adaptation) 技术，显著降低了训练所需的计算资源。

  - **AdamW** - 传统的全参数微调- **强大的答案评估器**: 内置了一个复杂的数学答案验证器 (`verifier`)，能够进行符号和数值层面的精确匹配，支持分数、百分比、区间、矩阵等多种格式。

  - **SGD** - 标准随机梯度下降优化器- **跨平台兼容**: 解决了在 Windows 等非 Posix 系统上由 `signal` 模块引发的兼容性问题，确保代码在不同操作系统上都能顺利运行。

  - **LoRA** - 高效参数微调，通过低秩适配器大幅减少计算和存储资源消耗- **详细的日志与可视化**: 训练过程会记录详细的日志（包括超参数），并能生成训练/验证损失曲线图，便于分析和比较不同训练策略的效果。

- **高效推理**: 集成 Flash Attention 2 优化，加速模型生成速度，支持批量推理处理。

- **强大的答案评估器**: 内置了功能完备的数学答案验证器 (`src/verifier/grader.py`)，参考 Hendrycks' MATH、ToRA、PRM800K 等业界标杆项目，能够进行符号和数值层面的精确匹配，支持分数、百分比、区间、矩阵、指数、三角函数等多种数学表达式格式。## 项目结构

- **自动化实验管理**: `run_experiments.py` 脚本支持循环测试多个超参数组合，自动管理文件路径和日志，一键运行完整的实验流程。

- **详细的日志与可视化**: 训练过程记录详细的 JSON 日志（包括所有超参数、训练时间、GPU 内存使用等），并能自动生成训练/验证损失曲线图，便于分析和比较不同训练策略的效果。```

.

## 项目结构├── data/                     # 存放由 prepare.py 生成的 .pkl 数据文件

├── result/                   # 存放推理和评估结果

```│   └── AdamW_lr_exp/         # 按实验组织的文件夹

.│       └── lr_1e-06/

├── data/                     # 存放由 prepare.py 生成的 .pkl 数据文件│           ├── answers.jsonl

├── result/                   # 存放推理和评估结果│           ├── execution.log

│   ├── AdamW_exp/            # 按实验组织的文件夹（AdamW 优化器）│           ├── loss_curve.png

│   │   └── lr_1e-06_epochs_1/│           └── scored_answers.jsonl

│   │       ├── answers.jsonl          # 模型生成的答案├── src/                      # 核心源代码

│   │       ├── execution.log          # 执行日志│   ├── verifier/             # 数学答案验证模块

│   │       ├── loss_curve.png         # 训练损失曲线图│   │   ├── __init__.py

│   │       └── scored_answers.jsonl   # 评估后的答案（包含分数）│   │   ├── grader.py

│   ├── SGD_exp/              # 按实验组织的文件夹（SGD 优化器）│   │   └── math_normalize.py

│   └── LoRA_exp/             # 按实验组织的文件夹（LoRA 优化器）│   ├── prepare.py            # 数据准备脚本

├── src/                      # 核心源代码│   ├── finetune.py           # 模型微调脚本

│   ├── verifier/             # 数学答案验证模块│   ├── rollout.py            # 模型推理（生成答案）脚本

│   │   ├── __init__.py│   ├── evaluate.py           # 自动化评估脚本

│   │   ├── grader.py         # 核心评分逻辑（支持多种数学格式）│   ├── run_experiments.py    # 自动化实验运行脚本

│   │   └── math_normalize.py # 答案规范化处理│   └── vllm_rollout.py       # (可选) 使用 vLLM 的高速推理脚本

│   ├── prepare.py            # 数据准备脚本├── train_log/                # 存放训练输出

│   ├── finetune.py           # 模型微调脚本│   └── AdamW_lr_exp/         # 按实验组织的文件夹

│   ├── rollout.py            # 模型推理（生成答案）脚本│       └── lr_1e-06/

│   ├── evaluate.py           # 自动化评估脚本│           ├── ... (模型文件)

│   ├── run_experiments.py    # 自动化实验运行脚本（推荐使用）│           └── training_logs.json

│   ├── run_experiments_lora.py # LoRA 专用实验脚本└── requirements.txt          # 项目依赖

│   └── vllm_rollout.py       # (可选) 使用 vLLM 的高速推理脚本```

├── train_log/                # 存放训练输出（模型权重和日志）

│   ├── AdamW_exp/            # 按实验组织的文件夹## 工作流

│   │   └── lr_1e-06_epochs_1/

│   │       ├── training_logs.json  # 训练日志（包括参数、时间、内存等）### 推荐：自动化实验流程

│   │       ├── adapter_config.json # (LoRA 才有) LoRA 配置文件

│   │       └── ...                 # 模型文件我们强烈推荐使用 `run_experiments.py` 脚本来自动化整个工作流。该脚本会依次执行数据准备、模型微调、推理和评估，并能循环测试多种超参数（如学习率），将每次实验的结果（日志、模型、答案、评估分数）整齐地保存在独立的目录中。

│   ├── SGD_exp/

│   └── LoRA_exp/```bash

├── requirements.txt          # 项目依赖# 启动自动化实验流程

└── README.md                 # 本文件python src/run_experiments.py

``````

- **工作原理**: 该脚本会调用 `finetune.py`, `rollout.py`, 和 `evaluate.py`，并自动管理文件路径和日志记录。

## 工作流- **日志模式**: 你可以在脚本内通过 `LOG_TO_FILE` 变量切换日志模式。`True` 会将所有输出保存到文件，`False` 则在控制台实时显示。



### 推荐：自动化实验流程### 手动分步执行



我们强烈推荐使用 `run_experiments.py` 脚本来自动化整个工作流。该脚本会依次执行数据准备、模型微调、推理和评估，并能循环测试多种超参数（如学习率），将每次实验的结果（日志、模型、答案、评估分数）整齐地保存在独立的目录中。如果你希望手动控制每个步骤，可以按照以下流程操作。



```bash#### 1. 环境配置

# 启动自动化实验流程（SGD 优化器）

python src/run_experiments.py首先，请确保你的环境满足要求。建议在虚拟环境中安装依赖。

```

```bash

- **工作原理**: 该脚本会调用 `finetune.py`、`rollout.py` 和 `evaluate.py`，并自动管理文件路径和日志记录。# 安装所有必需的库

- **日志模式**: 你可以在脚本内通过 `LOG_TO_FILE` 变量切换日志模式。`True` 会将所有输出保存到文件，`False` 则在控制台实时显示。pip install -r requirements.txt

- **超参数范围**: 脚本默认测试以下超参数组合：```

  - 学习率: `[1e-6, 5e-6, 1e-5, 5e-5, 1e-4]`

  - Epoch 数: `[1, 2, 3]`#### 2. 数据准备

  - 总共 15 个实验组合

运行 `prepare.py` 脚本，它会从 Hugging Face Hub 下载 `ricdomolm/MATH-500` 数据集，应用 Chat Template，进行 Tokenize 和格式化，然后将处理好的数据保存到 `data/` 目录下。

### 手动分步执行

```bash

如果你希望手动控制每个步骤，可以按照以下流程操作。python src/prepare.py

```

#### 1. 环境配置- **调试模式**: 添加 `--debug` 参数可以在 5% 的数据子集上快速运行，以验证流程。



首先，请确保你的环境满足要求。建议在虚拟环境中安装依赖。#### 3. 模型微调



```bash运行 `finetune.py` 脚本来启动模型训练。该脚本会加载预处理好的数据，并使用指定的优化器（默认为 AdamW，可通过参数选择 LoRA 或 SGD）对模型进行微调。

# 安装所有必需的库

pip install -r requirements.txt```bash

```# 使用 LoRA 进行训练并绘制损失图

python src/finetune.py --optimization_method lora --plot

**依赖项包括**:```

- `torch`: 深度学习框架- **核心参数**:

- `transformers`: Hugging Face 模型库  - `--optimization_method`: 选择优化器，可选 `adam`, `sgd`, `lora`。

- `peft`: 参数高效微调库（用于 LoRA）  - `--lora_rank`: 当使用 `lora` 时，设置 LoRA 的秩，默认为 8。

- `datasets`: 数据集加载库  - `--learning_rate`: 设置学习率，默认为 2e-5。

- `sympy`: 符号数学库（用于答案评估）  - `--num_epochs`: 设置训练轮次，默认为 1。

- `numpy`, `matplotlib`: 数值计算和可视化  - `--plot`: 训练结束后自动生成并保存损失曲线图。

- **输出**: 训练完成后，最佳的模型适配器（或完整模型）和 `training_logs.json` 日志文件（包含训练时间）将保存在指定的输出目录中。

#### 2. 数据准备

#### 4. 模型推理

运行 `prepare.py` 脚本，它会从 Hugging Face Hub 下载 `ricdomolm/MATH-500` 数据集，应用 Chat Template，进行 Tokenize 和格式化，然后将处理好的数据保存到 `data/` 目录下。

训练完成后，运行 `rollout.py` 脚本，它会加载基础模型和微调好的模型，在 MATH-500 测试集上生成答案。

```bash

python src/prepare.py```bash

```# 需要指定模型路径和输出文件

python src/rollout.py --lora_path "path/to/your/model" --output_file "result/answers.jsonl"

**主要功能**:```

- 从 Hugging Face 下载 MATH-500 数据集- **性能优化**: 该脚本已通过 `attn_implementation="flash_attention_2"` 进行了优化，以加速推理过程。

- 转换为 ChatML 格式（含数学指令）- **输出**: 推理结果（包含问题、模型答案、标准答案）将以 JSONL 格式保存在指定的输出文件中。

- 应用分词器处理

- 正确标记 prompt 部分以在训练中忽略其损失#### 5. 自动化评估

- 按 90% 训练/10% 验证比例分割数据

- 保存为 pickle 文件最后，运行 `evaluate.py` 脚本来评估模型的性能。



**可选参数**:```bash

- `--debug`: 在 5% 的数据子集上快速运行，以验证流程# 需要指定输入和输出文件

- `--model_name_or_path`: 指定分词器模型路径python src/evaluate.py --input_file "result/answers.jsonl" --output_file "result/scored_answers.jsonl"

- `--max_length`: 最大序列长度（默认 512）```

- `--num_proc`: 并行处理的 CPU 核心数- **工作原理**: 该脚本会读取 `answers.jsonl`，并调用 `verifier` 模块对每一条答案进行评分。

- **输出**: 包含分数和标准化答案的详细评估结果将保存在指定的输出文件中，并在控制台打印最终的准确率。

#### 3. 模型微调

## 核心模块详解

运行 `finetune.py` 脚本来启动模型训练。该脚本会加载预处理好的数据，并使用指定的优化器对模型进行微调。

- **`src/run_experiments.py`**: **（推荐使用）** 自动化实验的核心脚本。它串联了 `finetune`、`rollout` 和 `evaluate` 的所有步骤，能够自动管理文件路径、循环测试不同超参数，并将每次实验的结果清晰地组织在独立的目录中。

```bash- **`src/prepare.py`**: 负责数据处理。它将原始的问答对转换成模型可以理解的 ChatML 格式，并对 prompt 和 response 进行恰当的 Tokenize 和标签化（labeling），其中 prompt 部分的损失在训练中会被忽略。

# 使用 LoRA 进行训练并绘制损失图- **`src/finetune.py`**: 核心训练脚本。它集成了 `torch`、`transformers` 和 `peft`，实现了完整的训练、验证和保存逻辑。现在它还会记录总训练时间。

python src/finetune.py --optimization_method lora --plot- **`src/rollout.py`**: 推理脚本。演示了如何加载微调后的模型并进行批量推理。该脚本已通过 **Flash Attention 2** 进行了性能优化。

- **`src/evaluate.py`**: 评估脚本。它连接了推理输出和评分模块，实现了自动化的端到端评估。

# 使用 AdamW 进行训练- **`src/verifier/`**: 项目的亮点之一。该模块提供了强大的数学答案评分能力，其逻辑借鉴了多个业界领先的开源项目（如 Hendrycks' MATH、ToRA、PRM800K），能够准确判断各种形式的数学答案是否正确。

python src/finetune.py --optimization_method adam --learning_rate 2e-5

# 使用 SGD 进行训练
python src/finetune.py --optimization_method sgd --learning_rate 1e-4
```

**核心参数**:
- `--optimization_method`: 选择优化器，可选 `adam`、`sgd`、`lora`（默认: adam）
- `--lora_rank`: 当使用 `lora` 时，设置 LoRA 的秩（默认: 8）
- `--learning_rate`: 设置学习率（默认: 2e-5）
- `--num_epochs`: 设置训练轮次（默认: 1）
- `--batch_size`: 批处理大小（默认: 4）
- `--grad_accumulation_steps`: 梯度累积步数（默认: 16）
- `--plot`: 训练结束后自动生成并保存损失曲线图

**输出**:
- 最佳的模型适配器（或完整模型）
- `training_logs.json` 日志文件（包含：
  - 所有训练参数
  - 每个步骤的训练损失
  - 验证损失曲线
  - 总训练时间
  - 峰值 GPU 内存使用量）
- `loss_curve.png` 损失曲线图（如果指定 `--plot`）

#### 4. 模型推理

训练完成后，运行 `rollout.py` 脚本，它会加载基础模型和微调好的模型，在 MATH-500 测试集上生成答案。

```bash
# 使用微调后的模型进行推理
python src/rollout.py --lora_path "path/to/your/model" --output_file "result/answers.jsonl"

# 使用基础模型进行推理
python src/rollout.py --output_file "result/answers_base.jsonl"
```

**参数说明**:
- `--model`: 基础模型路径（默认: Qwen-3-0.6B-Base）
- `--lora_path`: 微调后模型的路径（支持 LoRA 适配器或完整模型）
- `--output_file`: 输出 JSONL 文件路径

**特性**:
- 自动检测 LoRA 适配器并进行合并
- 通过 Flash Attention 2 优化推理速度
- 批量处理（默认批大小: 32）
- 生成参数：温度=1.0，top_p=0.95，最大生成长度=512 tokens

**输出**: 推理结果（包含问题、模型答案、标准答案）将以 JSONL 格式保存，每行一条记录，格式如下：
```json
{
  "id": 0,
  "prompt": "问题文本",
  "answer": "模型生成的答案",
  "gold": "标准答案"
}
```

#### 5. 自动化评估

最后，运行 `evaluate.py` 脚本来评估模型的性能。

```bash
# 对推理结果进行评估
python src/evaluate.py --input_file "result/answers.jsonl" --output_file "result/scored_answers.jsonl"
```

**参数说明**:
- `--input_file`: 推理结果文件（JSONL 格式）
- `--output_file`: 评估结果输出文件（JSONL 格式）

**工作原理**: 
- 读取 JSONL 文件中的每条记录
- 调用 `src/verifier/grader.py` 对每个答案进行评分
- 支持的答案格式：
  - 数字（整数、小数、百分比）
  - 分数表达式
  - 区间表示
  - 矩阵和向量
  - 指数表达式
  - 三角函数
  - LaTeX 数学表达式

**输出**: 
- 包含分数和标准化答案的详细评估结果保存在指定的输出文件中
- 每条记录新增字段：
  - `score`: 0 或 1（答案是否正确）
  - `extracted_pred`: 提取的预测答案
  - `error`: 处理过程中的错误信息（如有）
- 控制台输出最终的准确率百分比

## 核心模块详解

### `src/prepare.py` - 数据准备
- **功能**: 将 MATH-500 数据集转换为模型可训练的格式
- **关键处理**:
  1. 从 Hugging Face 下载 MATH-500 数据集
  2. 将每条问答对转换为 ChatML 格式
  3. 添加数学指令：`"请逐步推理，并将最终答案放在 \\boxed{} 中"`
  4. 应用分词器的 Chat Template
  5. 标记 prompt 部分的 token（损失计算中忽略）
  6. 进行数据分割（90% 训练，10% 验证）
  7. 保存为 pickle 文件
- **输出**: `data/train.pkl` 和 `data/val.pkl`

### `src/finetune.py` - 模型微调
- **功能**: 实现完整的 LLM 指令微调流程
- **支持的优化方法**:
  - **AdamW**: 全参数微调，使用 AdamW 优化器
  - **SGD**: 全参数微调，使用随机梯度下降
  - **LoRA**: 参数高效微调，只训练低秩适配器
- **关键特性**:
  1. 自动检测 GPU 并使用 bfloat16 混合精度
  2. 梯度累积支持
  3. 验证集评估和最佳模型保存
  4. 详细的 JSON 日志记录
  5. 自动生成损失曲线图
- **输出**: 
  - `train_log/` 下的模型检查点
  - `training_logs.json` 包含完整的训练统计

### `src/rollout.py` - 模型推理
- **功能**: 在测试集上生成模型答案
- **特性**:
  - 支持加载 LoRA 适配器并自动合并
  - Flash Attention 2 加速
  - 批量推理处理
  - 可配置的生成参数
- **输出**: JSONL 格式的推理结果

### `src/evaluate.py` - 自动化评估
- **功能**: 评估模型答案的准确性
- **调用**: `src/verifier` 模块进行答案评分
- **输出**: 
  - 评分后的 JSONL 文件
  - 控制台的准确率统计

### `src/verifier/` - 数学答案验证
- **grader.py**: 核心评分逻辑
  - 参考 Hendrycks' MATH、ToRA、PRM800K 等项目
  - 支持符号级别的数学等价性检验
  - 使用 SymPy 进行符号计算和简化
  - 处理多种数学表达式格式
- **math_normalize.py**: 答案规范化
  - 处理百分比、货币符号
  - 处理不同进制表示
  - 处理 π 相关表达式

### `src/run_experiments.py` - 自动化实验运行 **（推荐使用）**
- **功能**: 一键运行多个超参数组合的完整实验流程
- **工作流程**:
  1. 遍历所有超参数组合
  2. 为每个组合创建独立的输出目录
  3. 依次执行 finetune → rollout → evaluate
  4. 将每次实验的日志、模型、答案、分数保存到独立的目录
  5. 支持日志模式切换（文件或控制台）
- **优势**:
  - 自动化管理复杂的实验流程
  - 清晰的目录组织结构
  - 便于比较不同超参数的效果
  - 支持从失败的地方恢复

## 使用建议

1. **快速体验**: 使用 `--debug` 参数在小数据集上测试完整流程
   ```bash
   python src/prepare.py --debug
   python src/finetune.py --num_epochs 1 --plot
   python src/rollout.py --output_file "result/answers_debug.jsonl"
   python src/evaluate.py --input_file "result/answers_debug.jsonl" --output_file "result/scored_debug.jsonl"
   ```

2. **对比不同优化器**: 使用相同的超参数对比 SGD、AdamW 和 LoRA
   ```bash
   python src/finetune.py --optimization_method adam --learning_rate 1e-4 --num_epochs 1
   python src/finetune.py --optimization_method sgd --learning_rate 1e-4 --num_epochs 1
   python src/finetune.py --optimization_method lora --lora_rank 8 --learning_rate 1e-6 --num_epochs 1
   ```

3. **系统的超参数搜索**: 使用 `run_experiments.py` 进行大规模超参数搜索
   - 修改脚本中的 `learning_rates` 和 `num_epochs_list` 参数
   - 设置 `LOG_TO_FILE = True` 以减少控制台输出
   - 运行后分析 `result/` 目录中各个实验的 `scored_answers.jsonl` 文件

4. **性能优化**:
   - 使用 LoRA 而非全参数微调可显著降低显存使用
   - 增大 `batch_size` 和 `grad_accumulation_steps` 可加速训练（需要充足的显存）
   - Flash Attention 2 已在 `rollout.py` 中默认启用

## 常见问题

**Q: 如何在 GPU 上加速训练?**
A: 确保安装了 CUDA 兼容的 PyTorch，显存充足。增大 `batch_size` 和 `grad_accumulation_steps`，启用混合精度（代码已默认启用）。

**Q: LoRA 的秩应该设置为多少?**
A: 通常设置在 8-32 之间。更高的秩 = 更多的可训练参数 = 更好的性能但更多的显存占用。建议从 8 开始尝试。

**Q: 如何评估基础模型的性能?**
A: 运行 `python src/rollout.py --output_file "result/answers_base.jsonl"`，然后评估该文件即可。

**Q: 训练过程中如何查看实时进度?**
A: 将 `run_experiments.py` 中的 `LOG_TO_FILE` 设置为 `False`，这样会在控制台显示详细的进度条。

## 许可证

该项目参考了多个开源项目的代码和想法，特别是：
- [Hendrycks' MATH](https://github.com/hendrycks/math)
- [ToRA](https://github.com/microsoft/ToRA)
- [PRM800K](https://github.com/openai/prm800k)
- [VERL](https://github.com/volcengine/verl)

具体许可信息见相应模块的文件头注释。

## 联系信息

如有问题或建议，欢迎提出 Issue 或 Pull Request。
