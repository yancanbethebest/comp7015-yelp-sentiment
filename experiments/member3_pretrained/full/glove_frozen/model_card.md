# Model card — member3_frozen_full_seed42

- 架构 / architecture: Bidirectional **LSTM**
- Embedding source: **glove**（frozen）
- 预训练词表覆盖率 / coverage: **0.8886**
- Embedding dim: 100, hidden: 128 per direction
- Dropout: 0.3 / embedding 0.1
- 参数量 / parameters: **236,034**
- 词表 / vocabulary: 30,000 ids (PAD=0, UNK=1)
- 截断长度 / max_len: 256
- 数据 / data: train 503,979 / val 55,998 / test 38,000
- 优化 / optimisation: AdamW lr=0.001, batch=64
- 最佳轮次 / best epoch: **7**

## Validation performance (best epoch)

| Metric | Value |
|---|---:|
| Accuracy | 0.9562 |
| Macro Precision | 0.9562 |
| Macro Recall | 0.9562 |
| Macro-F1 | **0.9562** |

## Figures

![training curves](training_curves.png)
![validation confusion matrix](confusion_matrix.png)
