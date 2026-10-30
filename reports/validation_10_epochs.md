# 10-epoch validation comparison (trial)

Date: 2026-10-30 (Asia/Shanghai). This is a validation-stage experiment; the official test split was not evaluated.

## Shared setup

- Data: fixed cleaned Yelp Polarity train (16,000) and validation (4,000), balanced by label; seed 42.
- Hardware: NVIDIA GeForce RTX 4090 on AutoDL.
- Model selection: highest validation Macro-F1 among 10 epochs.
- Checkpoints and full JSON metrics remain on the server under `/root/autodl-tmp/yelp_trial/`. No training outputs were downloaded to the local computer.

| Model | Embedding / model | Best epoch | Validation Accuracy | Validation Macro-F1 |
|---|---|---:|---:|---:|
| BiLSTM random | 100-dimensional trainable random embedding | 10 | 87.98% | 87.97% |
| BiLSTM GloVe | 100-dimensional GloVe 6B embedding, fine-tuned | 6 | 89.60% | 89.60% |
| BERT | `google-bert/bert-base-uncased`, full fine-tuning | 2 | 95.20% | 95.20% |

GloVe covered 81.41% of the train-derived LSTM vocabulary. The GloVe source archive matched SHA-256 `22d27d5f0080da2a57f48e6691af7b18aa39e96a54ef6fad02d9456a117239e1`.

## Validation results by epoch

| Epoch | Random LSTM Acc / F1 | GloVe LSTM Acc / F1 | BERT Acc / F1 |
|---:|---:|---:|---:|
| 1 | 78.70 / 78.63 | 83.65 / 83.64 | 94.40 / 94.40 |
| 2 | 83.08 / 83.07 | 84.92 / 84.90 | **95.20 / 95.20** |
| 3 | 85.88 / 85.87 | 86.00 / 85.95 | 94.92 / 94.92 |
| 4 | 85.88 / 85.85 | 88.42 / 88.42 | 94.47 / 94.47 |
| 5 | 87.48 / 87.45 | 88.05 / 88.03 | 95.15 / 95.15 |
| 6 | 87.42 / 87.42 | **89.60 / 89.60** | 94.70 / 94.70 |
| 7 | 87.35 / 87.34 | 89.18 / 89.16 | 94.83 / 94.82 |
| 8 | 86.88 / 86.86 | 87.95 / 87.90 | 94.60 / 94.60 |
| 9 | 87.60 / 87.60 | 88.65 / 88.64 | 94.67 / 94.67 |
| 10 | **87.98 / 87.97** | 88.85 / 88.85 | 94.95 / 94.95 |

Values are percentages. BERT reached its best result after 2 epochs; further training did not improve validation Macro-F1. Do not choose configurations using the official test split.

## Training settings

- LSTM: bidirectional, embedding 100, hidden 128, dropout 0.3, AdamW learning rate 0.001, batch size 128, max length 256, lowercased word/punctuation tokenizer, train-only vocabulary.
- BERT: max length 256, batch size 16, AdamW learning rate 0.00002, weight decay 0.01. Pretrained tokenizer and weights; the binary classification head was initialized for this task.
- BERT 10-epoch training and validation took 819.6 seconds (about 13.7 minutes) as recorded by the script. LSTM runtimes were not recorded.
- Code: `scripts/train_lstm.py` and `scripts/train_bert.py` at commit `c491548`.

This comparison is exploratory. The assignment's hyperparameter study and final test evaluation remain to be done.
