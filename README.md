# COMP7015 Yelp Sentiment

本仓库提供 Yelp Review Polarity 的统一清洗数据，以及随机词向量 BiLSTM、GloVe 词向量 BiLSTM 和预训练 BERT 微调的实验代码与验证结果。标签：`0` 为负面，`1` 为正面。

## 从哪里找文件

| 路径 | 内容 |
|---|---|
| [`processed_data/`](processed_data/) | 清洗后的 `train.csv`（16,000）、`val.csv`（4,000）、`test.csv`（5,000）。所有模型使用相同划分。 |
| [`scripts/`](scripts/) | `preprocessing.py` 生成数据；`train_lstm.py` 训练两种 BiLSTM；`train_bert.py` 微调 BERT。 |
| [`reports/`](reports/) | 清洗统计、数据校验和、审计文件，以及[10 轮模型对比](reports/validation_10_epochs.md)。 |
| [`lstm_results/`](lstm_results/) | **早期 5 轮随机 embedding LSTM 试跑**的模型、指标、预测和日志；不属于下表的 10 轮对比。 |
| `raw/` | 官方原始 Parquet 数据，仅重新清洗时下载，不上传 GitHub。 |

根目录还有[清洗规则](Yelp_Polarity_Cleaning_Rules.md)、[清洗依赖](requirements.txt)和[训练依赖](requirements-training.txt)。

## 每个模型的代码与结果

以下三组实验均使用 16,000 条训练、4,000 条验证数据，随机种子 42，训练 **10 个 epoch**，按验证集 Macro-F1 选择最佳轮次。**官方测试集尚未用于这三组实验。**

| 模型 | 代码 | 10 轮最佳验证 Accuracy / Macro-F1 | 最佳轮次 | 完整结果与 checkpoint |
|---|---|---:|---:|---|
| 随机 embedding BiLSTM | [`scripts/train_lstm.py`](scripts/train_lstm.py) | 87.98% / **87.97%** | 10 | 服务器 `/root/autodl-tmp/yelp_trial/random_10/` |
| GloVe embedding BiLSTM | [`scripts/train_lstm.py`](scripts/train_lstm.py) | 89.60% / **89.60%** | 6 | 服务器 `/root/autodl-tmp/yelp_trial/glove_10/` |
| 预训练 BERT 微调 | [`scripts/train_bert.py`](scripts/train_bert.py) | 95.20% / **95.20%** | 2 | 服务器 `/root/autodl-tmp/yelp_trial/bert_10/` |

GitHub 上的[验证报告](reports/validation_10_epochs.md)列出全部 10 轮的 Accuracy、Macro-F1 和训练设置。服务器各结果目录有 `metrics.json` 与最佳模型；训练日志分别是服务器上的 `random_10.log`、`glove_10.log` 和 `bert_10.log`。**这三组 10 轮 checkpoint 没有上传到 GitHub，也没有下载到本地**。BERT 使用已预训练的 `google-bert/bert-base-uncased`，本实验做的是情感分类微调，并非从零预训练 BERT。

`lstm_results/` 是此前另一组 5 轮试跑，其测试结果不能当作上表三组的最终测试比较。正式模型比较还需要超参数实验，选定配置后再统一评估官方测试集。

## 数据说明

三份 CSV 均为 UTF-8，列为 `sample_id,text,label`。训练和验证来自官方 train，测试来自官方 test；正负类各半。样本 ID 可与 `reports/*_audit.csv` 对照原文及原始行号。清洗规则保留大小写、否定词、标点、数字和 emoji；模型自己的词表只用训练集拟合。数据来源及 SHA-256 见 [`reports/manifest.json`](reports/manifest.json)。

## 复现命令

重新清洗数据（需要约 1 GB 空间及 `curl`）：

```bash
python -m pip install -r requirements.txt
python scripts/preprocessing.py
```

在有 GPU 的环境中安装 PyTorch、Transformers 后运行训练：

```bash
python -m pip install -r requirements-training.txt
python scripts/train_lstm.py --epochs 10 --embedding-dim 100 --output random_10
python scripts/train_lstm.py --epochs 10 --embedding-dim 100 --embedding-file glove.6B.100d.txt --output glove_10
python scripts/train_bert.py --train-limit 16000 --val-limit 4000 --epochs 10 --max-len 256 --batch-size 16 --output bert_10
```

GloVe 版需先取得 Stanford GloVe 6B 的 `glove.6B.100d.txt`。LSTM 词表从训练集构建，序列转为整数并动态 padding；两种 LSTM 使用相同结构、100 维 embedding 和训练设置。上述命令默认**不评估测试集**；仅在最终确定模型后才传入 `--evaluate-test`。
