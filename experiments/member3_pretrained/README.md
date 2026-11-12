# 成员三 · 预训练词向量对比与调参

> 成员三负责：加载 GloVe/Word2Vec 预训练词向量，与随机初始化基线对比，并系统调参。
> 本目录汇总全部实验结果。代码位于 `models/lstm_pretrained/`，报告见 `reports/member3_experiments.md`。
> 全量数据（train 503,979 / val 55,998）为与组内统一口径的最终结果；20k 子集仅用于快速调参和过拟合分析，作为补充。

## 目录结构

| 目录 | 内容 |
|---|---|
| full/ | 全量数据最终结果（GPU T4） |
| error_analysis/ | 全量最佳模型的错误分析与阈值校准 |
| random/ | 20k 随机词向量基线（补充） |
| glove_frozen/ | 20k GloVe 冻结（补充） |
| glove_finetuned/ | 20k GloVe 微调默认参数（补充） |
| best/ | 20k 调参后最佳组合（补充） |
| tuning/ | 8 组调参 + 正则消融 + 汇总表（补充） |

## 全量数据结果（seed 42，GPU T4，最终）

- 数据：processed_data 全量，train 503,979 / val 55,998，max_len=256
- 模型：BiLSTM，embedding_dim=100，复用成员二 `RecurrentClassifier`
- 训练：AdamW，50 轮预算 + 早停
- 测试集：未评估，留到成员五统一评估

### 总体指标

| 模型 | 最佳轮次 | Accuracy | Macro-P | Macro-R | Macro-F1 | 可训练参数量 |
|---|---:|---:|---:|---:|---:|---:|
| random_bilstm（成员二，随机初始化） | 3 | 95.33% | 95.34% | 95.33% | 95.33% | 3,236,034 |
| random_bigru（成员二，随机初始化） | 2 | 95.65% | 95.66% | 95.65% | 95.65% | 3,177,154 |
| glove_frozen（GloVe 冻结） | 7 | 95.62% | 95.62% | 95.62% | 95.62% | 236,034 |
| glove_finetuned（GloVe 微调，默认参数） | 2 | 95.77% | 95.78% | 95.77% | 95.77% | 3,236,034 |
| best（GloVe 微调 + 调参） | 2 | 95.73% | 95.74% | 95.73% | 95.73% | 3,734,210 |

### 逐类指标

| 模型 | 类别 | Precision | Recall | F1 | 支持数 |
|---|---|---|---:|---:|---:|
| random_bilstm | 负面(0) | 94.99% | 95.71% | 95.35% | 27,999 |
| random_bilstm | 正面(1) | 95.68% | 94.95% | 95.32% | 27,999 |
| random_bigru | 负面(0) | 94.96% | 96.42% | 95.69% | 27,999 |
| random_bigru | 正面(1) | 96.37% | 94.89% | 95.62% | 27,999 |
| glove_frozen | 负面(0) | 95.63% | 95.60% | 95.62% | 27,999 |
| glove_frozen | 正面(1) | 95.60% | 95.63% | 95.62% | 27,999 |
| glove_finetuned | 负面(0) | 94.94% | 96.69% | 95.81% | 27,999 |
| glove_finetuned | 正面(1) | 96.63% | 94.85% | 95.73% | 27,999 |
| best | 负面(0) | 95.18% | 96.35% | 95.76% | 27,999 |
| best | 正面(1) | 96.30% | 95.12% | 95.71% | 27,999 |

### 混淆矩阵

行=真实，列=预测，格式 `[[TN, FP], [FN, TP]]`：

| 模型 | 混淆矩阵 |
|---|---|
| random_bilstm | `[[26799, 1200], [1413, 26586]]` |
| random_bigru | `[[26997, 1002], [1432, 26567]]` |
| glove_frozen | `[[26768, 1231], [1224, 26775]]` |
| glove_finetuned | `[[27072, 927], [1442, 26557]]` |
| best | `[[26976, 1023], [1366, 26633]]` |

### 全量结论

1. GloVe 预训练词向量略优于随机初始化：默认微调 95.77% 比随机 BiLSTM 95.33% 高约 0.44 个百分点，也比随机 BiGRU 95.65% 高约 0.12 个百分点。
2. 默认微调（95.77%）最高，调参组合（95.73%）未带来额外提升，说明充分数据下该任务对超参数不敏感。
3. 冻结版参数量最少、分数几乎持平，是性价比最高的选择。
4. 微调版对正类 Precision 更高（96.63%）、对负类 Recall 更高（96.69%），正负类略有偏向；冻结版正负类几乎完全对称。

## 错误分析与阈值校准（全量最佳模型）

对全量最佳模型（glove_finetuned，95.77%）在验证集上做错误分析，见 error_analysis/error_analysis.md。

要点：

- 长评论（>200 词）错误率最高（5.67%），印证 max_len 截断的影响。
- 含否定词的评论错误率更高（4.42% vs 3.37%）。
- 低置信度预测错误率高达 44%，prob_positive 可作为可靠性参考。
- 阈值校准基本无收益（0.5 → 95.77%，最佳 0.35 → 95.79%），模型已较好校准。

## 20k 子集结果（补充：用于调参和过拟合分析）

20k 子集（train 20,000 / val 4,000，seed 42，max_len=128，CPU 本地）只用于快速筛选超参数和观察过拟合，不参与最终口径。

### 总体指标

| 模型 | 最佳轮次 | Accuracy | Macro-P | Macro-R | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| random（随机词向量） | 3 | 87.58% | 87.75% | 87.58% | 87.56% |
| glove_frozen 10 轮 | 10 | 88.10% | 88.16% | 88.10% | 88.09% |
| glove_frozen 20 轮 | 11 | 89.00% | 89.04% | 89.00% | 89.00% |
| glove_finetuned（默认参数） | 3 | 88.88% | 88.92% | 88.87% | 88.87% |
| best（微调 + 调参） | 2 | 89.75% | 89.79% | 89.75% | 89.75% |

### 逐类指标

| 模型 | 类别 | Precision | Recall | F1 | 支持数 |
|---|---|---|---:|---:|---:|
| random | 负面(0) | 85.20% | 90.95% | 87.98% | 2,000 |
| random | 正面(1) | 90.29% | 84.20% | 87.14% | 2,000 |
| glove_frozen 10 轮 | 负面(0) | 89.73% | 86.05% | 87.85% | 2,000 |
| glove_frozen 10 轮 | 正面(1) | 86.60% | 90.15% | 88.34% | 2,000 |
| glove_frozen 20 轮 | 负面(0) | 90.21% | 87.50% | 88.83% | 2,000 |
| glove_frozen 20 轮 | 正面(1) | 87.86% | 90.50% | 89.16% | 2,000 |
| glove_finetuned | 负面(0) | 90.31% | 87.10% | 88.67% | 2,000 |
| glove_finetuned | 正面(1) | 87.54% | 90.65% | 89.07% | 2,000 |
| best | 负面(0) | 88.48% | 91.40% | 89.92% | 2,000 |
| best | 正面(1) | 91.11% | 88.10% | 89.58% | 2,000 |

### 混淆矩阵

| 模型 | 混淆矩阵 |
|---|---|
| random | `[[1819, 181], [316, 1684]]` |
| glove_frozen 10 轮 | `[[1721, 279], [197, 1803]]` |
| glove_frozen 20 轮 | `[[1750, 250], [190, 1810]]` |
| glove_finetuned | `[[1742, 258], [187, 1813]]` |
| best | `[[1828, 172], [238, 1762]]` |

20k 调参汇总见 tuning/tuning_summary.md。

20k 与全量的对比：

- 20k 上预训练优势更明显：随机 87.56% → GloVe 微调 88.87%，约 +1.3 个百分点；全量上仅约 +0.1–0.4 个百分点。数据越充分，预训练词向量的额外收益越小。
- 20k 上微调过拟合明显（训练准确率可达 99%+），因此曾选择冻结；全量上过拟合变轻，默认微调反而略高。

## 复现命令

### 全量（GPU）

    python -m models.lstm_pretrained.train --embedding-source glove --embedding-file glove.txt --freeze-embedding --run-name member3_frozen_full_seed42 --epochs 50 --patience 5 --batch-size 64 --max-len 256

    python -m models.lstm_pretrained.train --embedding-source glove --embedding-file glove.txt --run-name member3_finetuned_full_seed42 --epochs 50 --patience 3 --batch-size 64 --max-len 256

    python -m models.lstm_pretrained.train --embedding-source glove --embedding-file glove.txt --run-name member3_best_finetuned_full_seed42 --epochs 50 --patience 3 --batch-size 32 --max-len 256 --lr 0.002 --hidden-dim 256

    python -m models.lstm_pretrained.error_analysis --checkpoint runs/member3_finetuned_full_seed42/best_model.pt --data processed_data --output experiments/member3_pretrained/error_analysis

### 20k（CPU 本地，补充）

    python -m models.lstm_pretrained.train --embedding-source glove --embedding-file <glove.txt> --freeze-embedding --run-name member3_frozen_20ep_seed42 --limit-train 20000 --limit-val 4000 --epochs 20 --batch-size 64 --max-len 128 --patience 5

## 局限

1. 单种子（seed 42），未做多种子方差分析。
2. max_len 会截断长评论（20k 用 128，全量用 256）。
3. GloVe 覆盖词表 83.57%，约 16% 的词仍为随机初始化。
4. 尚未做 Word2Vec 双源对比（题目允许 GloVe 或 Word2Vec 二选一）。