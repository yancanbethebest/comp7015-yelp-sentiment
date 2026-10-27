# Yelp Review Polarity 清洗数据

A comparative study of machine learning and deep learning approaches for binary sentiment classification on the Yelp Review Polarity dataset.

## 文件夹说明

| 文件夹 | 内容 | 组员如何使用 |
|---|---|---|
| [`data/`](data/) | 统一清洗后的 `train.csv`、`val.csv`、`test.csv`。 | 直接使用这些文件训练和评估，保持组内划分一致。 |
| [`scripts/`](scripts/) | `preprocessing.py` 清洗脚本、`train_lstm.py` 训练脚本。 | 复现数据或 LSTM 基线时运行。 |
| [`reports/`](reports/) | 清洗统计、异常记录、来源及校验和、原文审计文件、LSTM 实验报告。 | 核查数据质量、来源与实验设置。 |
| [`lstm_results/`](lstm_results/) | 最佳模型、完整指标、测试集预测、训练日志。 | 查看结果或加载模型进行推理。 |
| `raw/` | 官方原始 Parquet 数据，脚本运行时下载；不纳入 Git。 | 仅在重新清洗时需要，约 274 MB。 |

根目录的 [`Yelp_Polarity_Cleaning_Rules.md`](Yelp_Polarity_Cleaning_Rules.md) 是共享清洗规范；[`requirements.txt`](requirements.txt) 是数据清洗依赖。LSTM 训练另需 PyTorch。

此目录的 `data/train.csv`、`data/val.csv` 和 `data/test.csv` 是按 [清洗规则](Yelp_Polarity_Cleaning_Rules.md) v1.0 在 Windows 本地生成的统一数据版本。原始数据来自 [Hugging Face `fancyzhx/yelp_polarity`](https://huggingface.co/datasets/fancyzhx/yelp_polarity)，官方 train/test 各为 560,000/38,000 条。标签 `0` 为负面、`1` 为正面。

| 文件 | 条数 | 负面 | 正面 | 来源 |
|---|---:|---:|---:|---|
| `data/train.csv` | 16,000 | 8,000 | 8,000 | 官方 train |
| `data/val.csv` | 4,000 | 2,000 | 2,000 | 官方 train |
| `data/test.csv` | 5,000 | 2,500 | 2,500 | 官方 test |

每个 CSV 有 `sample_id,text,label` 三列。`sample_id` 采用 `train:原始行号` 或 `test:原始行号`，可与 `reports/*_audit.csv` 对照原文。`reports/cleaning_summary.csv` 记录清洗数量和长度分位数；`reports/flagged_records.csv` 供人工复核；`reports/manifest.json` 记录来源及输出的 SHA-256、随机种子和依赖版本。原始 Parquet 文件保存在 `raw/`，不纳入 Git，可按脚本重新下载。

## 复现

```powershell
python -m pip install -r requirements.txt
python scripts/preprocessing.py
```

脚本会下载完整官方 train/test，再清洗全部样本，隔离冲突标签，去除同分区重复及 train/test 重叠，最后以种子 `42` 分层抽样。`raw/` 约 274 MB；预留至少 1 GB 磁盘空间。脚本使用 `curl` 下载。建议组员直接使用仓库中已生成的 CSV，不要各自重新抽样。模型相关的词表、标准化参数仅在 `train.csv` 上拟合。

本版本保留大小写、标点、数字、否定词和 emoji；没有执行 NFKC 规范化或邮箱替换。`flagged_records.csv` 中的极短文本仅标记，仍可能保留在输出中。人工复核若决定修改冲突处理，须更新规则版本并重新生成三份 CSV 及校验和。

## LSTM 分类实验

在 GPU 服务器上用 [训练脚本](scripts/train_lstm.py) 对上述固定划分进行了 5 轮监督训练。模型是词级双向 LSTM（embedding 128、hidden 128），词表仅从训练集构建，最长 256 个 token。按验证集 Macro-F1 选出第 5 轮模型，然后在测试集评估一次。详见 [实验报告](reports/lstm_experiment.md)、`lstm_results/metrics.json`、`lstm_results/test_predictions.csv`、`lstm_results/best_model.pt` 和 `lstm_results/training.log`。

服务器复现命令（需安装 PyTorch）：

```bash
python scripts/train_lstm.py --epochs 5
```
