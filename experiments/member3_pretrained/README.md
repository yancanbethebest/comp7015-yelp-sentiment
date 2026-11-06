# 成员三 · 预训练词向量对比与调参（20k 子集，seed 42）

- 数据：processed_data 固定划分，分层抽取 train 20,000 / val 4,000（seed 42）
- 模型：BiLSTM，embedding_dim=100，max_len=128
- 训练：AdamW，带早停

## 最终方案

采用 **GloVe 冻结（20 轮预算 + 早停）**：

- 验证 Accuracy **89.00%**，Macro-F1 **89.00%**，最佳轮次 11
- 可训练参数 236,034，过拟合明显更轻

选择理由：分数与微调接近（微调最佳 89.75%），但冻结版背答案更慢、泛化更稳，更适合作为最终模型。

## 核心对比

| 模型 | 最佳轮次 | 验证 Accuracy | 验证 Macro-F1 |
|---|---:|---:|---:|
| random（随机词向量） | 3 | 87.58% | 87.56% |
| glove_frozen 10 轮 | 10 | 88.10% | 88.09% |
| glove_frozen 20 轮（最终） | 11 | 89.00% | 89.00% |
| glove_finetuned（默认参数） | 3 | 88.88% | 88.87% |
| best（微调 + 调参） | 2 | 89.75% | 89.75% |

## 调参

见 tuning/tuning_summary.md。

## 复现命令

    # 最终冻结方案
    python -m models.lstm_pretrained.train --embedding-source glove --embedding-file <glove.6B.100d.txt路径> --freeze-embedding --run-name member3_frozen_20ep_seed42 --limit-train 20000 --limit-val 4000 --epochs 20 --batch-size 64 --max-len 128 --lr 0.001 --dropout 0.3 --hidden-dim 128 --patience 5