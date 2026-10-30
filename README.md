# COMP7015 — Yelp Review Sentiment Analysis

A comparative study of deep learning approaches for binary sentiment classification using the **Yelp Review Polarity Dataset**.

本项目为 COMP7015 自然语言处理小组项目，旨在比较不同文本表示方法与深度学习模型在 Yelp 用户评论情感分类任务中的表现。

项目使用统一的数据清洗、分层抽样和模型评估流程，构建 Random Embedding BiLSTM、GloVe BiLSTM 和 BERT 三种基线模型，并支持后续超参数优化、模型对比与错误分析。

---

## 1. Project Overview

### Task: Binary Sentiment Classification

给定一条 Yelp 用户评论，预测其情感类别：

| Label | Sentiment | Description |
|---|---|---|
| `0` | Negative | 负面评论 |
| `1` | Positive | 正面评论 |

### Research Objectives

本项目主要研究以下问题：

1. **Embedding Comparison:** 比较随机初始化词向量与预训练 GloVe 词向量在 BiLSTM 模型中的表现。
2. **Architecture Comparison:** 分析循环神经网络与预训练 Transformer 模型之间的性能差异。
3. **Hyperparameter Optimization:** 研究学习率、Batch Size、Dropout 等超参数对模型表现的影响。
4. **Model Evaluation:** 从分类性能、训练效率、模型稳定性和错误类型等方面进行综合分析。

---

## 2. Dataset & Data Preprocessing

### 2.1 Official source and full cleaned dataset

Data source: [Yelp Review Polarity](https://huggingface.co/datasets/fancyzhx/yelp_polarity). The official dataset contains **560,000 training reviews and 38,000 test reviews**, for 598,000 rows in total. Member 1 cleaned the complete official dataset. After removing 14 same-label duplicates, 7 conflicting-label rows and 2 train/test overlaps, **597,977 eligible reviews** remain: 559,977 from official train and all 38,000 from official test.

Using seed `42`, we split all eligible official training rows into stratified train/validation sets (90%/10%). The official test split remains separate.

| Current file | Source | Total | Negative (0) | Positive (1) |
|---|---|---:|---:|---:|
| [`processed_data/train.csv`](processed_data/train.csv) | Official train | 503,979 | 251,986 | 251,993 |
| [`processed_data/val.csv`](processed_data/val.csv) | Official train | 55,998 | 27,999 | 27,999 |
| [`processed_data/test.csv`](processed_data/test.csv) | Official test | 38,000 | 19,000 | 19,000 |
| **Total** | | **597,977** | **298,985** | **298,992** |

The full CSV files are stored with Git LFS. After cloning, run `git lfs pull`. Cleaning statistics and SHA-256 checksums are in [`reports/`](reports/) and [`reports/manifest.json`](reports/manifest.json).

### 2.2 Data used for the published preliminary models

**The checkpoints and validation scores in Sections 3–4 were produced earlier using a fixed 25,000-row sample, not the full cleaned dataset above.** That historical sample is preserved at [`experiments/sample_25k/`](experiments/sample_25k/): 16,000 train, 4,000 validation and 5,000 test. Its train/validation rows came from official train, and its test rows came from official test. Each split was balanced by label. The current full-data CSV files were published after those model trials; no full-data retraining has been performed yet.

| Experiment stage | Train | Validation | Test | Use |
|---|---:|---:|---:|---|
| Published preliminary checkpoints | 16,000 | 4,000 | 5,000 | Reproduce existing trial results |
| Current full cleaned split | 503,979 | 55,998 | 38,000 | Formal training by the assigned model members |

Results from the two stages must be reported separately. The formal training responsibilities are listed in [`NEXT_STEPS.md`](NEXT_STEPS.md).

### 2.3 Cleaning and model-specific preprocessing

The shared pipeline checks missing text and invalid labels, normalizes whitespace, decodes HTML entities, replaces explicit URLs, removes exact duplicates, isolates conflicting labels and excludes official train/test overlaps. It keeps negation, punctuation, numbers and case. Very short reviews are flagged rather than deleted. See [cleaning rules](Yelp_Polarity_Cleaning_Rules.md), [`scripts/preprocessing.py`](scripts/preprocessing.py) and [cleaning summary](reports/cleaning_summary.csv).

For the LSTM models, the training script lowercases and tokenizes text, builds the vocabulary **only from its training split**, maps tokens to integers and applies dynamic padding. The random and GloVe LSTMs share this pipeline; GloVe initializes the embedding layer. BERT uses its pretrained tokenizer and fine-tunes the sequence classifier. Formal models must fit their vocabulary or other data-derived settings on the full training split only.

---

## 3. Model Architectures

本项目实现三种主要模型，用于比较不同文本表示方法和网络架构的性能。

### 3.1 Random Embedding BiLSTM

**Directory:** [`models/lstm_random/`](models/lstm_random/)

- Architecture: Bidirectional LSTM
- Embedding: Random Initialization
- Training Strategy: Training from Scratch
- Objective: Establish a recurrent neural network baseline without external pre-trained embeddings.

该模型通过监督训练同时学习词向量表示与情感分类能力，作为后续模型比较的基础。

### 3.2 GloVe Embedding BiLSTM

**Directory:** [`models/lstm_glove/`](models/lstm_glove/)

- Architecture: Bidirectional LSTM
- Embedding: Pre-trained GloVe
- Embedding Strategy: Trainable Embeddings
- Objective: Evaluate the contribution of pre-trained word embeddings.

该模型使用预训练 GloVe 词向量初始化 Embedding Layer，并在情感分类训练过程中对词向量进行微调。

### 3.3 Pre-trained BERT

**Directory:** [`models/bert/`](models/bert/)

- Architecture: BERT-based Sequence Classification
- Initialization: Pre-trained BERT Checkpoint
- Training Strategy: Fine-tuning
- Objective: Evaluate contextual language representations for sentiment classification.

BERT 利用预训练语言模型的上下文表示能力，通过下游情感分类任务进行监督微调。

---

## 4. Baseline Experimental Results

### 4.1 Validation Performance

以下为三组基线模型在验证集上的最佳实验结果。

| Model | Actual Epochs | Best Epoch | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| Random BiLSTM | 71 | 69 | 89.25% | 89.25% |
| GloVe BiLSTM | 35 | 10 | 90.12% | 90.12% |
| BERT | 10 | 2 | **95.20%** | **95.20%** |

**Checkpoint Selection Metric:** Validation Macro-F1

所有发布的最佳权重均按照验证集 Macro-F1 选择，而不是直接使用最后一轮的权重。

### 4.2 Preliminary Findings

**GloVe vs. Random Embedding**

GloVe BiLSTM 的验证集 Accuracy 比 Random BiLSTM 高 0.87 个百分点。

这表明在当前实验条件下，预训练词向量对情感分类任务具有一定帮助。

**BERT vs. GloVe BiLSTM**

BERT 的验证集 Accuracy 比 GloVe BiLSTM 高 5.08 个百分点。

在当前三组基线实验中，BERT 获得最高的验证集 Accuracy 与 Macro-F1。

**Training Behavior**

- Random BiLSTM: 最佳结果出现在第 69 轮。
- GloVe BiLSTM: 最佳结果出现在第 10 轮。
- BERT: 最佳结果出现在第 2 轮。

这些结果说明不同架构的最佳训练轮次存在差异，但不能仅根据最佳轮次判断模型是否过拟合。

### 4.3 Training Notes

- 两组 LSTM 的原计划训练上限均为 100 epochs。
- Random BiLSTM 实际完成 71 epochs。
- GloVe BiLSTM 实际完成 35 epochs。
- 两组 LSTM 均根据实际运行安排手动提前停止。
- BERT 完成 10 epochs，最佳 checkpoint 来自第 2 轮。
- BERT 的 `.pt` 文件由原始 Hugging Face checkpoint 转换得到，转换过程未重新训练，模型张量保持一致。

详细训练记录、配置及指标保存在对应模型目录和实验报告中。

**Important:** 本章节结果均为验证集指标，并非最终测试集结果。当前三组基线模型尚未完成统一最终测试。

---

## 5. Project Structure

```text
COMP7015-Yelp-Sentiment/
│
├── models/
│   ├── lstm_random/             # Random Embedding BiLSTM
│   ├── lstm_glove/              # GloVe BiLSTM
│   └── bert/                    # Fine-tuned BERT
│
├── processed_data/              # Cleaned and split datasets
│
├── experiments/
│   └── lstm_100/                # Extended LSTM experiments
│
├── reports/                     # Audit logs and experiment reports
│
├── scripts/                     # Preprocessing, prediction and plotting
│
├── requirements-training.txt    # Training dependencies
├── Yelp_Polarity_Cleaning_Rules.md
├── NEXT_STEPS.md
└── README.md
```

### Standard Model Directory

每个模型目录包含以下文件：

```text
model_name/
├── best_model.pt
├── model.py
├── train.py
├── config.yaml
├── metrics.json
└── README.md
```

| File | Description |
|---|---|
| `best_model.pt` | Best model checkpoint selected by validation Macro-F1 |
| `model.py` | Model architecture and checkpoint loading utilities |
| `train.py` | Model training implementation |
| `config.yaml` | Model configuration and training hyperparameters |
| `metrics.json` | Evaluation results and experiment metrics |
| `README.md` | Model-specific usage instructions |

BERT 模型目录还包含对应的 Tokenizer 文件。

---

## 6. Installation & Environment Setup

### 6.1 Clone Repository

模型权重采用 **Git LFS** 管理。

建议使用 Git 克隆仓库，而不是直接下载 GitHub ZIP 压缩包。

```bash
git clone https://github.com/yancanbethebest/comp7015-yelp-sentiment.git
cd comp7015-yelp-sentiment

git lfs install
git lfs pull
```

### 6.2 Install Dependencies

```bash
python -m pip install -r requirements-training.txt
```

建议在独立 Python 虚拟环境中运行项目。

### 6.3 Git LFS Notes

如果下载后的 `best_model.pt` 只有几行文本，说明该文件可能是 Git LFS 指针，而不是真正的模型权重。

运行：

```bash
git lfs pull
```

即可获取仓库中受 Git LFS 管理的实际权重文件。

---

## 7. Training & Reproducibility

### 7.1 Run Baseline Models

所有训练命令均从仓库根目录执行。

**Random BiLSTM**

```bash
python models/lstm_random/train.py
```

**GloVe BiLSTM**

```bash
python models/lstm_glove/train.py --embedding-file glove.6B.100d.txt
```

**BERT**

```bash
python models/bert/train.py
```

每个训练脚本默认读取其模型目录下的 `config.yaml`。

支持的命令行参数可以覆盖配置文件中的对应参数，具体参数以训练脚本和模型 README 为准。

GloVe 原始词向量文件仅在重新训练相关模型时需要。

### 7.2 Training Output

新的训练结果默认输出至：

```text
runs/
├── lstm_random/
├── lstm_glove/
└── bert/
```

重新运行训练脚本不会覆盖 `models/` 目录中已发布的最佳模型权重。

### 7.3 Reproducibility

本项目采用以下措施提高可复现性：

- Fixed Random Seed: `42`
- Fixed Training / Validation / Test Splits
- Consistent Binary Label Definition
- Shared Evaluation Metrics
- Validation Macro-F1-based Checkpoint Selection
- Saved Training Configurations and Experiment Logs

固定随机种子有助于减少随机波动，但不同硬件、软件版本和非确定性计算操作仍可能造成实验结果差异。

---

## 8. Model Checkpoints & Hyperparameter Tuning

### 8.1 Model Loading

每个模型目录中的 `model.py` 提供 `load_checkpoint()` 函数。

具体调用方式与依赖请参考各模型目录中的 README。

已发布的 `best_model.pt` 主要用于加载最佳模型参数。

**注意：** 当前 checkpoint 不包含优化器状态，因此不能用于精确恢复先前中断的训练过程。

### 8.2 Hyperparameter Optimization

后续实验可以基于已有模型架构、训练脚本和配置文件开展超参数优化。

| Hyperparameter | Candidate Values | Purpose |
|---|---|---|
| Learning Rate | 1e-3 / 1e-4 / 1e-5 | Optimization behavior |
| Batch Size | 16 / 32 / 64 | Training stability and efficiency |
| Dropout | 0.2 / 0.3 / 0.5 | Regularization |
| Hidden Size | 64 / 128 / 256 | Model representation capacity |
| Number of Layers | 1 / 2 / 3 | Network depth |
| Max Sequence Length | Model-dependent | Input context coverage |

以上参数仅作为候选示例，不代表所有模型都适合使用相同的搜索范围。

超参数实验应根据对应架构分别设计。

### 8.3 Important Notes on Fine-tuning

**Training Hyperparameters**

学习率、Batch Size 等训练参数可以在兼容模型结构的情况下进行调整，并可选择重新训练或继续训练。

**Architectural Hyperparameters**

Hidden Size、网络层数等结构参数可能改变权重张量的形状，通常需要重新构建模型并重新训练相关参数。

**Checkpoint Limitation**

`.pt` 文件提供已有模型参数，但完整的超参数优化仍依赖：

- Model Architecture
- Training Code
- Configuration Files
- Training Dataset
- Evaluation Pipeline

建议所有实验均独立保存配置、验证指标、训练记录及模型权重。

### 8.4 Historical Checkpoints

历史 Release 保留作为实验过程记录：

- [Extended LSTM Checkpoints](https://github.com/yancanbethebest/comp7015-yelp-sentiment/releases/tag/lstm-extended-2026-10-31)
- [10-Epoch Experimental Checkpoints](https://github.com/yancanbethebest/comp7015-yelp-sentiment/releases/tag/checkpoints-2026-10-30)

当前正式模型以各模型目录中的 `best_model.pt` 为准。

---

## 9. Evaluation Protocol

本项目采用以下评估指标：

| Metric | Description |
|---|---|
| Accuracy | Overall classification accuracy |
| Precision | Proportion of predicted positives that are correct |
| Recall | Proportion of actual positives correctly identified |
| Macro-F1 | Unweighted mean of per-class F1 scores |
| Confusion Matrix | Distribution of correct and incorrect predictions |

### 9.1 Validation Set

验证集用于：

- 模型训练过程监控
- 超参数选择
- 最佳 checkpoint 选择
- 模型配置比较

### 9.2 Test Set

测试集用于模型配置最终确定后的泛化性能评估。

原则上不应使用测试集进行反复调参或模型选择。

### 9.3 Evaluation Limitations

当前发布的三组模型仅报告统一的验证集指标，尚未完成最终统一测试。

由于历史试跑中曾查看部分测试结果，因此不能将现有测试集描述为完全未被访问。

后续最终报告应明确披露这一情况，并避免进一步依据测试表现修改模型配置。

---

## 10. Additional Resources

| Path | Description |
|---|---|
| [`processed_data/`](processed_data/) | Fixed cleaned datasets |
| [`experiments/lstm_100/`](experiments/lstm_100/) | Extended LSTM training logs, plots and validation results |
| [`reports/`](reports/) | Data cleaning audit, checksums and experimental reports |
| [`scripts/`](scripts/) | Data preparation, visualization and prediction utilities |
| [Cleaning Rules](Yelp_Polarity_Cleaning_Rules.md) | Detailed data cleaning specification |
| [Next Steps](NEXT_STEPS.md) | Model optimization, evaluation and collaboration guidelines |

`experiments/lstm_100/` 中的 `100` 表示原计划训练轮数，而非实际完成轮数。

早期五轮训练结果仅作为历史试跑记录，不属于当前正式交付的基线模型。

---

## 11. Future Work

后续研究将围绕模型优化、系统比较与最终评估展开。

### Hyperparameter Optimization

系统研究不同模型的学习率、Batch Size、Dropout、网络结构及输入长度等参数。

### Architecture Comparison

进一步比较 BiLSTM、GRU 与 BERT 架构的分类性能、训练效率和计算开销。

### Embedding Analysis

对比随机初始化词向量与预训练 GloVe 词向量，并分析冻结和微调策略对分类结果的影响。

### Robustness & Error Analysis

开展多随机种子实验，并分析错误分类样本中的语言特征及可能的模型局限性。

### Final Model Evaluation

在最终模型配置确定后，统一计算测试集指标，完成模型对比与实验总结。

具体任务安排、协作方式及实验交付要求见 [NEXT_STEPS.md](NEXT_STEPS.md)。

---

## 12. Project Notes

- 本项目使用抽样子集开展实验，不代表完整 Yelp Review Polarity 数据集上的模型表现。
- 三组基线模型使用相同的数据划分和随机种子 `42`。
- 模型权重使用 Git LFS 管理。
- 原始数据与新的运行输出目录不上传到仓库。
- 不在仓库中保存服务器地址、密码、访问令牌等敏感信息。
- 历史训练记录与实验报告保留用于实验过程追踪。
- 所有最终性能结论应基于统一的评估流程。

---

**Course:** COMP7015

**Project:** Yelp Review Polarity Sentiment Classification

**Dataset:** [Hugging Face — Yelp Review Polarity](https://huggingface.co/datasets/fancyzhx/yelp_polarity)

**Task:** Natural Language Processing / Binary Sentiment Classification
