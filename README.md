# COMP7015 Yelp Sentiment

Yelp Review Polarity 二分类小组项目。标签：`0` 负面，`1` 正面。

## 每个模型一个文件夹

```text
models/
├── lstm_random/       # 随机词向量 BiLSTM
├── lstm_glove/        # GloVe 词向量 BiLSTM（词向量参与微调）
└── bert/              # 预训练 BERT 情感分类微调
```

**每个模型目录均包含** `best_model.pt`（最佳权重）、`model.py`（架构与加载函数）、`train.py`（训练代码）、`config.yaml`（训练配置），另附 `metrics.json` 和使用说明。BERT 目录还包含 tokenizer 文件。

| 模型目录 | 实际训练轮数 | 最佳轮次 | 最佳验证 Accuracy / Macro-F1 |
|---|---:|---:|---:|
| [随机词向量 LSTM](models/lstm_random/) | 71 | 69 | 89.25% / 89.25% |
| [GloVe LSTM](models/lstm_glove/) | 35 | 10 | 90.12% / 90.12% |
| [BERT](models/bert/) | 10 | 2 | 95.20% / 95.20% |

LSTM 原定上限各 100 轮，按用户要求手动提前停止；`config.yaml` 分别记录训练上限和历史实际轮次。全部权重均按验证 Macro-F1 选择，不是最后一轮权重。BERT 的 `.pt` 文件由原始 Hugging Face checkpoint 转换，张量完全一致，未重新训练。

## 下载最佳模型

权重采用 **Git LFS** 存储，真实文件位于各模型目录的 `best_model.pt`。首次克隆后执行：

```bash
git lfs install
git lfs pull
python -m pip install -r requirements-training.txt
```

如果下载 ZIP 得到的 `.pt` 只有几行文字，那是 LFS 指针，请使用上述命令获取权重。组员下载后即可在自己的训练环境使用。

之前的 [LSTM 最佳模型 Release](https://github.com/yancanbethebest/comp7015-yelp-sentiment/releases/tag/lstm-extended-2026-10-31) 和 [10 轮实验 Release](https://github.com/yancanbethebest/comp7015-yelp-sentiment/releases/tag/checkpoints-2026-10-30) 保留作为历史下载记录。

## 训练与加载

请在仓库根目录运行；每个脚本默认读取自己目录下的 `config.yaml`，命令行参数可以覆盖配置。

```bash
python models/lstm_random/train.py
python models/lstm_glove/train.py --embedding-file glove.6B.100d.txt
python models/bert/train.py
```

新的训练结果分别写入 `runs/lstm_random/`、`runs/lstm_glove/`、`runs/bert/`，不会覆盖发布权重。GloVe 原始词向量文件只在重新训练时需要。各目录的 `model.py` 提供 `load_checkpoint()`，用法见对应 README。checkpoint 不含优化器状态，不用于精确恢复中断训练。

## 其他文件在哪里

| 路径 | 内容 |
|---|---|
| [processed_data/](processed_data/) | 固定清洗数据：训练 16,000、验证 4,000、测试 5,000；正负类各半 |
| [experiments/lstm_100/](experiments/lstm_100/) | LSTM 延长训练的曲线、验证混淆矩阵、指标和日志；目录名表示原计划，实际为 71 / 35 轮 |
| [reports/](reports/) | 清洗审计、校验和、历史实验报告以及 [10 轮对比](reports/validation_10_epochs.md) |
| [scripts/](scripts/) | 数据清洗、绘图、预测工具及旧训练入口；新的统一训练入口在 `models/` |
| [Yelp_Polarity_Cleaning_Rules.md](Yelp_Polarity_Cleaning_Rules.md) | 数据清洗规范 |
| [NEXT_STEPS.md](NEXT_STEPS.md) | 模型加载、组员分工、超参数研究、BERT 对比和最终评估指南 |

旧 `lstm_results/` 已删除；其中早期 5 轮试跑结果不是当前交付模型。历史报告保留作过程记录。原始数据与新运行目录不上传，服务器地址、密码和访问令牌不进入仓库。

## 实验边界与下一步

所有模型使用相同的清洗划分和种子 42。训练/验证取自官方训练集，测试来自官方测试集；LSTM 词表只用训练集拟合，BERT 使用预训练 tokenizer。这些是抽样数据实验，不能写成使用全部官方训练集。

表中均为**验证指标**，本次发布的三组模型没有做统一最终测试。历史试跑曾查看测试结果，不能称测试集从未被接触。仍需完成超参数对照、多种子验证和统一最终评估；具体分工及交付物见 [组员下一步工作指南](NEXT_STEPS.md)。
