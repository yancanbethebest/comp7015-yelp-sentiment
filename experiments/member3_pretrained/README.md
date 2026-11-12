# 成员三 · 预训练词向量对比与调参

> 20k 与全量两个版本并存：20k 用于调参和过拟合分析，全量为最终结果。

## 20k 子集结果（seed 42，CPU 本地）

- 数据：processed_data 固定划分，分层抽取 train 20,000 / val 4,000（seed 42）
- 模型：BiLSTM，embedding_dim=100，max_len=128
- 训练：AdamW，带早停

| 模型 | 最佳轮次 | 验证 Accuracy | 验证 Macro-F1 |
|---|---:|---:|---:|
| random（随机词向量） | 3 | 87.58% | 87.56% |
| glove_frozen 10 轮 | 10 | 88.10% | 88.09% |
| glove_frozen 20 轮 | 11 | 89.00% | 89.00% |
| glove_finetuned（默认参数） | 3 | 88.88% | 88.87% |
| best（微调 + 调参） | 2 | 89.75% | 89.75% |

20k 调参见 tuning/tuning_summary.md。

## 全量数据结果（seed 42，GPU T4，最终）

- 数据：processed_data 全量，train 503,979 / val 55,998，max_len=256
- 训练：AdamW，50 轮预算 + 早停

| 模型 | 最佳轮次 | 验证 Accuracy | 验证 Macro-F1 |
|---|---:|---:|---:|
| glove_frozen（冻结） | 7 | 95.62% | 95.62% |
| glove_finetuned（默认参数） | 2 | 95.77% | 95.77% |
| best（微调 + 调参） | 2 | 95.73% | 95.73% |

参考：成员二随机基线（全量）BiLSTM 95.33% / BiGRU 95.65%。

结论：全量上预训练词向量仍略优于随机，但优势明显缩小（约 0.1–0.4 个百分点，而 20k 上约 1.3 个百分点）；默认微调最高，20k 上调出的组合在全量上没有带来额外提升。

## 全量数据复现（服务器 / GPU）

    # 1) GloVe 冻结
    python -m models.lstm_pretrained.train --embedding-source glove --embedding-file glove.txt --freeze-embedding --run-name member3_frozen_full_seed42 --epochs 50 --patience 5 --batch-size 64 --max-len 256

    # 2) GloVe 微调（默认参数）
    python -m models.lstm_pretrained.train --embedding-source glove --embedding-file glove.txt --run-name member3_finetuned_full_seed42 --epochs 50 --patience 3 --batch-size 64 --max-len 256

    # 3) 调参后的最佳组合
    python -m models.lstm_pretrained.train --embedding-source glove --embedding-file glove.txt --run-name member3_best_finetuned_full_seed42 --epochs 50 --patience 3 --batch-size 32 --max-len 256 --lr 0.002 --hidden-dim 256

注：不传 --limit-train / --limit-val 即使用全量数据。