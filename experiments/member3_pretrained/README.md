# 成员三 · 预训练词向量对比与调参

## 20k 子集结果（seed 42，CPU 本地）

- 数据：processed_data 固定划分，分层抽取 train 20,000 / val 4,000（seed 42）
- 模型：BiLSTM，embedding_dim=100，max_len=128
- 训练：AdamW，带早停

### 最终方案（20k）

采用 GloVe 冻结（20 轮预算 + 早停）：验证 Accuracy **89.00%**，Macro-F1 **89.00%**，最佳轮次 11。

| 模型 | 最佳轮次 | 验证 Accuracy | 验证 Macro-F1 |
|---|---:|---:|---:|
| random（随机词向量） | 3 | 87.58% | 87.56% |
| glove_frozen 10 轮 | 10 | 88.10% | 88.09% |
| glove_frozen 20 轮（最终） | 11 | 89.00% | 89.00% |
| glove_finetuned（默认参数） | 3 | 88.88% | 88.87% |
| best（微调 + 调参） | 2 | 89.75% | 89.75% |

调参见 tuning/tuning_summary.md。

20k 最终方案复现：

    python -m models.lstm_pretrained.train --embedding-source glove --embedding-file <glove.txt> --freeze-embedding --run-name member3_frozen_20ep_seed42 --limit-train 20000 --limit-val 4000 --epochs 20 --batch-size 64 --max-len 128 --lr 0.001 --dropout 0.3 --hidden-dim 128 --patience 5

## 全量数据复现（服务器 / GPU）

在 GPU 服务器上，对全量 processed_data（train 503,979 / val 55,998）运行：

    # 1) GloVe 冻结（默认参数，与成员二随机基线同结构）
    python -m models.lstm_pretrained.train --embedding-source glove --embedding-file glove.txt --freeze-embedding --run-name member3_frozen_full_seed42 --epochs 50 --patience 5 --batch-size 64 --max-len 256

    # 2) GloVe 微调（默认参数）
    python -m models.lstm_pretrained.train --embedding-source glove --embedding-file glove.txt --run-name member3_finetuned_full_seed42 --epochs 50 --patience 3 --batch-size 64 --max-len 256

    # 3) 调参后的最佳组合（50 轮 + 早停）
    python -m models.lstm_pretrained.train --embedding-source glove --embedding-file glove.txt --run-name member3_best_finetuned_full_seed42 --epochs 50 --patience 3 --batch-size 32 --max-len 256 --lr 0.002 --hidden-dim 256

注：不传 --limit-train / --limit-val 即使用全量数据；全量阶段 max_len=256 与成员二全量随机基线一致。