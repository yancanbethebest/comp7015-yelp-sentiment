# COMP7015 — Yelp Review Sentiment Analysis

A comparative study of deep learning approaches for binary sentiment classification using the **Yelp Review Polarity Dataset**.

本项目为 COMP7015 小组课程项目，研究随机词向量、预训练词向量和 BERT 在 Yelp 评论情感二分类任务中的表现。

本阶段主要完成了**全量数据清洗、固定数据划分，以及抽样数据上的模型初步训练**。正式模型训练、超参数优化和最终评估由后续负责成员继续完成。

| 阶段 | 内容 | 当前状态 |
|---|---|---|
| Data Preparation | 全量数据清洗、质量检查、固定划分与数据交付 | 已完成 |
| Preliminary Training | 在抽样数据上验证三种模型的训练、保存和加载流程 | 已完成 |
| Formal Training & Optimization | 在全量清洗数据上训练、调参并比较模型 | 待对应成员完成 |
| Final Evaluation & Reporting | 统一测试评估、错误分析、报告与 PPT | 待完成 |

当前发布的模型权重属于 **Preliminary Checkpoints**，其验证结果来自历史抽样实验。

---

## 1. Project Overview

### 1.1 Task Definition

输入一条 Yelp 用户评论，预测其情感类别：

| Label | Sentiment | Description |
|---|---|---|
| `0` | Negative | 负面评论 |
| `1` | Positive | 正面评论 |

### 1.2 Research Objectives

1. **Embedding Comparison:** 比较随机初始化词向量与 GloVe 预训练词向量对 BiLSTM 的影响。
2. **Architecture Comparison:** 比较循环神经网络与预训练 BERT 的分类表现。
3. **Hyperparameter Optimization:** 研究学习率、Batch Size、Dropout 和隐藏单元数量等参数的影响。
4. **Computational Analysis:** 比较分类效果、训练时间、显存需求与实现复杂度。
5. **Error Analysis:** 分析模型在否定表达、长评论和混合情感等样本上的错误。

GRU、网络深度调整等可作为后续扩展，当前已交付的模型为 Random BiLSTM、GloVe BiLSTM 和 BERT。


---

## 2. Dataset & Data Preprocessing

### 2.1 Official Dataset

**Source:** [Yelp Review Polarity — Hugging Face](https://huggingface.co/datasets/fancyzhx/yelp_polarity)

| Official Split | Samples |
|---|---:|
| Training | 560,000 |
| Test | 38,000 |
| **Total** | **598,000** |

官方数据集没有单独的验证集。训练集和验证集均从官方训练数据中划分，官方测试数据独立保留。

### 2.2 Full Cleaned Dataset

已对全部官方数据执行统一清洗，共保留 **597,977 条合格评论**：

| Cleaning Operation | Removed Rows |
|---|---:|
| 同标签完全重复记录 | 14 |
| 相同文本但标签冲突的记录 | 7 |
| 官方训练集与测试集之间的重叠记录 | 2 |
| **Total Removed** | **23** |

清洗后，官方训练池包含 **559,977 条**，官方测试池包含 **38,000 条**。

使用随机种子 `42`，按标签将全部合格官方训练记录分层划分为 90% 训练集与 10% 验证集：

| Current File | Samples | Negative | Positive |
|---|---:|---:|---:|
| [`processed_data/train.csv`](processed_data/train.csv) | 503,979 | 251,986 | 251,993 |
| [`processed_data/val.csv`](processed_data/val.csv) | 55,998 | 27,999 | 27,999 |
| [`processed_data/test.csv`](processed_data/test.csv) | 38,000 | 19,000 | 19,000 |
| **Total** | **597,977** | **298,985** | **298,992** |

`processed_data/` 是当前正式交付的**全量清洗数据目录**。类别比例接近 50% / 50%；清洗后的训练集没有通过复制或额外删除样本强制凑成完全平衡。

CSV 字段为：

```text
sample_id,text,label
```

`sample_id` 包含官方来源分区和原始行号，便于追溯。

### 2.3 Historical Subset Used for Preliminary Training

此前为验证模型和训练流程，从清洗后的数据池中构建了 **25,000 条类别平衡的抽样数据**：

| Split | Samples | Negative | Positive |
|---|---:|---:|---:|
| Training | 16,000 | 8,000 | 8,000 |
| Validation | 4,000 | 2,000 | 2,000 |
| Test | 5,000 | 2,500 | 2,500 |
| **Total** | **25,000** | **12,500** | **12,500** |

该历史划分保存在 [`experiments/sample_25k/`](experiments/sample_25k/)。

**当前三组 checkpoint 均在 16,000 条训练样本上完成初步训练，使用 4,000 条验证样本选择最佳权重。它们尚未在全量清洗数据上重新训练。**

| Experiment Stage | Training | Validation | Test |
|---|---:|---:|---:|
| Published preliminary experiments | 16,000 | 4,000 | 5,000 |
| Full cleaned dataset for formal experiments | 503,979 | 55,998 | 38,000 |

两阶段的指标应分别报告，不能将已有抽样实验成绩写成全量训练结果。

### 2.4 Data Cleaning Pipeline

共享清洗流程包括：

1. 检查缺失文本、空文本和非法标签。
2. 规范空白字符，解码 HTML 实体并规范明确的 URL。
3. 检查清洗后的完全重复文本。
4. 隔离相同文本对应不同标签的冲突记录。
5. 排除官方训练集与测试集之间的文本重叠。
6. 标记极短文本并统计文本长度，不因长度自动删除。
7. 固定随机种子，完成分层划分、统计与校验和记录。

清洗保留否定词、标点、数字、大小写和 emoji。共享清洗阶段不统一删除停用词，也不提前执行模型专属的分词、词表构建或 padding。

相关文件：

- [Cleaning Rules](Yelp_Polarity_Cleaning_Rules.md)
- [Preprocessing Code](scripts/preprocessing.py)
- [Cleaning Summary](reports/cleaning_summary.csv)
- [Flagged Records](reports/flagged_records.csv)
- [Data Manifest & SHA-256](reports/manifest.json)

### 2.5 Model-Specific Preprocessing

| Model | Text Processing | Representation |
|---|---|---|
| Random BiLSTM | 小写化、分词、训练集词表、整数编码、动态 padding | 随机初始化且可训练的词向量 |
| GloVe BiLSTM | 与 Random BiLSTM 相同 | GloVe 初始化，并参与微调 |
| BERT | 预训练 tokenizer、截断与动态 padding | 上下文相关表示 |

LSTM 词表只由对应实验的训练集构建，验证集和测试集不参与拟合。

正式训练应基于全量训练集重新建立适用的词表与缓存，并保持各模型使用相同的样本划分。

---

## 3. Preliminary Models

### 3.1 Random Embedding BiLSTM

**Directory:** [`models/lstm_random/`](models/lstm_random/)

- Architecture: Bidirectional LSTM
- Embedding: Random initialization, trainable
- Training Strategy: Supervised training from scratch
- Preliminary Training Samples: 16,000

用于建立不依赖外部预训练词向量的循环网络基线。

### 3.2 GloVe Embedding BiLSTM

**Directory:** [`models/lstm_glove/`](models/lstm_glove/)

- Architecture: Bidirectional LSTM
- Embedding: GloVe 6B, 100 dimensions
- Embedding Strategy: Fine-tuned during training
- Preliminary Training Samples: 16,000

使用预训练 GloVe 词向量初始化 embedding layer，比较预训练词向量对分类效果的影响。

### 3.3 BERT Sequence Classifier

**Directory:** [`models/bert/`](models/bert/)

- Model: `google-bert/bert-base-uncased`
- Architecture: BERT with a binary classification head
- Training Strategy: Fine-tuning
- Preliminary Training Samples: 16,000

使用已有预训练 BERT 进行情感分类微调。

这里的 **Preliminary Training** 指项目中的初步试跑阶段：Random BiLSTM 从头训练，GloVe BiLSTM 使用预训练词向量，BERT 使用预训练语言模型进行任务微调。

---

## 4. Preliminary Experimental Results

### 4.1 Validation Performance

以下结果使用相同的历史划分：训练 16,000 条、验证 4,000 条，随机种子为 `42`。

| Model | Actual Epochs | Best Epoch | Validation Accuracy | Validation Macro-F1 |
|---|---:|---:|---:|---:|
| Random BiLSTM | 71 | 69 | 89.25% | 89.25% |
| GloVe BiLSTM | 35 | 10 | 90.12% | 90.12% |
| BERT | 10 | 2 | **95.20%** | **95.20%** |

**Checkpoint Selection:** Best Validation Macro-F1

两组 LSTM 原计划最多训练 100 轮，运行过程中手动提前停止。实际完成轮数和最佳轮次分别记录在配置及指标文件中。

BERT 的 `best_model.pt` 由原始 Hugging Face checkpoint 转换得到，转换前后权重张量一致。

### 4.2 Preliminary Observations

- GloVe BiLSTM 的验证 Accuracy 比 Random BiLSTM 高 **0.87 个百分点**。
- BERT 的验证 Accuracy 比 GloVe BiLSTM 高 **5.08 个百分点**。
- 两种 LSTM 的训练准确率接近 100%，而验证表现明显较低，训练曲线显示过拟合。
- 更多训练轮数没有持续带来验证性能提升，GloVe 的最佳结果出现在第 10 轮。

[查看 LSTM 训练曲线、混淆矩阵及完整记录](experiments/lstm_100/)

### 4.3 Interpretation & Limitations

这些是**单个种子、特定配置下的初步验证结果**，用于检查训练流程并提供后续实验参考。

目前尚未完成全量数据上的正式训练、系统超参数研究、多种子稳定性分析和统一最终测试。不同训练预算下的初步结果，也不足以单独判断模型的计算效率。

---

## 5. Project Structure

```text
COMP7015-Yelp-Sentiment/
├── processed_data/              # Full cleaned dataset: 597,977 reviews
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── models/
│   ├── lstm_random/
│   ├── lstm_glove/
│   └── bert/
├── experiments/
│   ├── sample_25k/              # Historical split used by published models
│   └── lstm_100/                # Preliminary LSTM curves and metrics
├── reports/                     # Cleaning statistics, checksums and reports
├── scripts/                     # Cleaning, plotting and prediction tools
├── requirements.txt
├── requirements-training.txt
├── Yelp_Polarity_Cleaning_Rules.md
├── NEXT_STEPS.md
└── README.md
```

每个模型目录包含：

| File | Description |
|---|---|
| `best_model.pt` | 初步实验中验证 Macro-F1 最佳的 checkpoint |
| `model.py` | 模型架构及加载函数 |
| `train.py` | 训练实现 |
| `config.yaml` | 训练参数与已发布实验说明 |
| `metrics.json` | 初步训练及验证指标 |
| `README.md` | 模型使用说明 |

BERT 目录另包含 tokenizer 文件。

---

## 6. Installation & Environment Setup

模型权重和全量清洗 CSV 使用 **Git LFS** 管理。

```bash
git clone https://github.com/yancanbethebest/comp7015-yelp-sentiment.git
cd comp7015-yelp-sentiment

git lfs install
git lfs pull

python -m pip install -r requirements-training.txt
```

若下载的 `.pt` 或 `.csv` 文件只有几行文字，说明取得的是 LFS 指针，需要运行 `git lfs pull` 获取真实文件。

重新运行数据清洗时，还需安装：

```bash
python -m pip install -r requirements.txt
python scripts/preprocessing.py
```

建议在独立 Python 环境中运行。GPU 用于后续正式模型训练。

---

## 7. Training & Reproducibility

以下命令供相应模型负责人使用。

### 7.1 Historical Subset Training

若要使用历史抽样重新运行初步实验，应显式指定数据路径：

```bash
python models/lstm_random/train.py \
  --data experiments/sample_25k

python models/lstm_glove/train.py \
  --data experiments/sample_25k \
  --embedding-file glove.6B.100d.txt

python models/bert/train.py \
  --data experiments/sample_25k \
  --train-limit 16000 \
  --val-limit 4000
```

每个脚本默认读取自身目录的 `config.yaml`，命令行参数可覆盖配置。

LSTM 配置保留原计划 100 轮上限。历史实验为手动停止，因此重新运行不会自动在第 71 或第 35 轮停止。

### 7.2 Full-Dataset Training

正式训练使用当前全量清洗划分：

```bash
python models/lstm_random/train.py \
  --data processed_data \
  --output runs/full_training/lstm_random

python models/lstm_glove/train.py \
  --data processed_data \
  --embedding-file glove.6B.100d.txt \
  --output runs/full_training/lstm_glove

python models/bert/train.py \
  --data processed_data \
  --train-limit 503979 \
  --val-limit 55998 \
  --output runs/full_training/bert
```

**BERT 必须同时确认 `--train-limit` 和 `--val-limit`。** 现有配置保留了初步实验的样本数量限制，仅更换数据目录不会自动使用全部训练和验证样本。

正式负责人应在运行前确定训练预算和超参数。这些命令是入口示例，不代表已经执行或完成正式训练。

GloVe 原始词向量文件只在重新训练时需要，加载已发布 checkpoint 不需要该文件。

### 7.3 Checkpoint Loading and Continued Training

各模型的 `model.py` 提供 `load_checkpoint()`，用法见对应模型目录的 README。

当前 checkpoint 不含优化器状态，训练入口也未提供自动恢复历史训练的功能。

若希望基于初步 LSTM 权重继续训练，必须保持词表及 token ID 映射兼容。重新建立全量词表后，不能直接假设旧 embedding 权重仍可按原索引使用；应重新初始化或显式完成词向量映射。

### 7.4 Reproducibility Records

每组正式实验应保存：

- 数据版本、样本数及校验和；
- 随机种子、模型结构和初始化来源；
- 学习率、Batch Size、Dropout 等训练配置；
- 每轮指标、最佳轮次与 checkpoint；
- 软件版本、GPU 型号、训练耗时及显存使用。

固定随机种子有助于复现，但不同硬件与软件环境仍可能产生数值差异。

---

## 8. Hyperparameter Optimization

正式模型负责人需要在固定的全量划分上开展参数对照。

| Model | Suggested Parameters |
|---|---|
| Random BiLSTM | Learning Rate、Batch Size、Dropout、Hidden Size |
| GloVe BiLSTM | Embedding Frozen/Fine-tuned、Learning Rate、Batch Size、Dropout |
| BERT | Learning Rate、Batch Size、Sequence Length、Fine-tuning Strategy |

参数选择仅依据验证集。建议对最终候选配置使用多个训练种子重复实验，报告均值与标准差。

改变隐藏维度、层数等结构参数时，需相应修改模型实现。`config.yaml` 中增加一个字段，并不意味着训练代码已经支持该参数。

历史权重保留在：

- [Extended LSTM Checkpoints](https://github.com/yancanbethebest/comp7015-yelp-sentiment/releases/tag/lstm-extended-2026-10-31)
- [10-Epoch Experimental Checkpoints](https://github.com/yancanbethebest/comp7015-yelp-sentiment/releases/tag/checkpoints-2026-10-30)

---

## 9. Formal Evaluation Protocol

### 9.1 Metrics

正式评估应报告：

| Metric | Reporting Requirement |
|---|---|
| Accuracy | 整体分类准确率 |
| Precision | 每类及 Macro Precision |
| Recall | 每类及 Macro Recall |
| F1-score | 每类 F1 和 Macro-F1 |
| Confusion Matrix | 正负类别预测分布 |

### 9.2 Validation and Test Usage

验证集用于选择超参数、训练轮次和最终 checkpoint。

在模型配置锁定后，使用统一的 **38,000 条测试数据**进行最终评估，保存逐样本预测，支持混淆矩阵和错误分析。

历史试跑曾查看过部分官方测试子集结果，因此不能声称测试集在整个项目中从未被访问。后续应避免根据测试结果继续调参，并在正式报告中说明这一限制。

### 9.3 Final Comparison

最终比较应包含分类指标、训练配置、数据规模、训练时间、显存需求和错误类型。

历史抽样实验与正式全量实验分别展示。计算成本比较应尽量统一硬件条件，并说明各模型的训练预算。

---

## 10. Team Responsibilities & Next Steps

| Member | Responsibility | Deliverables |
|---|---|---|
| 成员 1 | 项目统筹、全量数据清洗与划分、初步训练流程验证 | 数据预处理代码、数据交付、报告 Data 部分 |
| 成员 2 | 随机 Embedding 的 LSTM/GRU 正式基础模型 | 模型代码、正式训练结果、报告 Model 部分 |
| 成员 3 | GloVe/Word2Vec 与基础模型对比、超参数优化 | 实验代码、调参结果、报告 Experiments 部分 |
| 成员 4 | BERT/DistilBERT 正式微调及性能与计算成本比较 | BERT 代码、对比结果、报告 Advanced Model 部分 |
| 成员 5 | 统一评估、混淆矩阵、报告与 PPT 整合 | 评估代码、最终报告、展示 PPT |

**当前数据清洗和初步试跑已完成，正式训练由成员 2–4 继续开展。**

详细任务与交付要求见 [NEXT_STEPS.md](NEXT_STEPS.md)。

---

## 11. Additional Resources

| Path | Description |
|---|---|
| [`processed_data/`](processed_data/) | 全量清洗数据及固定划分 |
| [`experiments/sample_25k/`](experiments/sample_25k/) | 已发布初步模型使用的历史抽样 |
| [`experiments/lstm_100/`](experiments/lstm_100/) | LSTM 初步训练曲线、混淆矩阵与日志 |
| [`reports/`](reports/) | 清洗统计、校验和及历史实验报告 |
| [`scripts/`](scripts/) | 清洗、绘图和预测工具 |
| [Cleaning Rules](Yelp_Polarity_Cleaning_Rules.md) | 数据清洗规范 |
| [Next Steps](NEXT_STEPS.md) | 正式训练与协作指南 |

`experiments/lstm_100/` 中的 `100` 表示原计划训练上限，实际完成轮数为 71 和 35。

---

## 12. Project Notes

- 正式数据交付共 **597,977 条**，位于 `processed_data/`。
- 当前三组模型使用历史 **25,000 条抽样划分**完成初步训练。
- 当前指标均为初步验证结果，尚无统一的正式全量训练结果。
- 模型权重和全量清洗 CSV 通过 Git LFS 下载。
- 正式训练使用统一数据划分，词表和参数选择仅依据训练集与验证集。
- 后续模型结果应明确记录数据版本、配置和初始化来源。
- 原始数据与新的运行输出目录不上传仓库。
- 不提交服务器密码、访问令牌或其他敏感信息。

---

**Course:** COMP7015  
**Project:** Yelp Review Polarity Sentiment Classification  
**Dataset:** [Yelp Review Polarity — Hugging Face](https://huggingface.co/datasets/fancyzhx/yelp_polarity)  
**Task:** Natural Language Processing / Binary Sentiment Classification
