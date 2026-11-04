# lstm_gru_random — 成员二WU Yifei · 随机初始化 Embedding 的 LSTM/GRU 基础模型

> COMP7015 · Topic 2 · Yelp Review Polarity 情感二分类
> Member：WU Yifei (成员二LSTM/GRU 基础模型：随机初始化 Embedding）
> Version：v1.0（2026-11-04）

本目录是成员二的**可复用代码包**，一个命令就能从成员一交付的 `processed_data/` 跑到
`runs/<run_name>/` 下的完整结果（指标、日志、曲线、混淆矩阵、模型卡、checkpoint）。

---

## 1. 目录与职责

```text
models/lstm_gru_random/
├── __init__.py       # 对外统一入口（成员三/四/五 import 这里）
├── config.py         # 配置加载 / 合并 / 校验 / 路径解析
├── config.yaml       # 默认超参（命令行可覆盖任意一项）
├── data.py           # 词表、动态 Padding、DataLoader、成员一数据校验
├── model.py          # RecurrentClassifier（BiLSTM / BiGRU，随机 Embedding）
├── metrics.py        # Accuracy / P / R / F1 / 混淆矩阵 + 评估循环
├── train.py          # 训练主循环、早停、checkpoint、CLI
├── predict.py        # 推理 CLI（单句 / 批量 CSV）
├── plotting.py       # 训练曲线与混淆矩阵（matplotlib 可选）
└── README.md         # 本文件
```

| 模块 | 谁会用 | 说明 |
|---|---|---|
| `data.py` | 成员三 | 词表与 DataLoader 的**唯一口径**，改了这里就等于改了所有人的实验 |
| `model.py` | 成员三 | `embedding_weights=` 参数就是预训练词向量的接入点 |
| `metrics.py` | 成员五 | 统一的指标实现，避免各人公式不一致 |
| `train.py` | 成员二/三 | 训练入口与全套产物 |
| `predict.py` | 成员四/五 | 加载 checkpoint 做推理或批量打分 |

---

## 2. 环境准备

```bash
# 1) 克隆后必须先拉 LFS，否则 processed_data/*.csv 只是几行文字的指针
git lfs install && git lfs pull

# 2) 安装训练依赖（torch / transformers / matplotlib / PyYAML）
python -m pip install -r requirements-training.txt

# 3) 确认数据是真实文件（train.csv 应约 374 MB）
ls -lh processed_data/
```

> 没有 GPU 也能跑通 `--smoke-test` 与 `--limit-train` 的小样本实验，但**全量正式训练需要 GPU**。

---

## 3. 快速开始（4 条命令）

```bash
# ① 1~2 分钟链路自检：数据 → 词表 → 前向 → 反向 → 存/读 checkpoint
python -m models.lstm_gru_random.train --smoke-test

# ② 正式训练 BiLSTM（全量 503,979 训练 / 55,998 验证，20 轮预算 + 早停）
python -m models.lstm_gru_random.train

# ③ 正式训练 BiGRU 对照（除循环单元外全部相同，保证可归因）
python -m models.lstm_gru_random.train --rnn-type gru --run-name member2_bigru_random_full_seed42

# ④ 训练完用最佳 checkpoint 抽查几句
python -m models.lstm_gru_random.predict \
    --checkpoint runs/member2_bilstm_random_full_seed42/best_model.pt \
    --text "Great food and friendly service." \
    --text "Cold food, rude waiter, never again."
```

想先在小样本上试跑（约 10 分钟，确认流程再上全量）：

```bash
python -m models.lstm_gru_random.train --limit-train 20000 --limit-val 4000 --run-name debug_20k
```

配置锁定后，**只做一次**官方测试集评估：

```bash
python -m models.lstm_gru_random.train \
    --run-name member2_bilstm_random_full_seed42 \
    --evaluate-test --resume runs/member2_bilstm_random_full_seed42/last_full.pt
```

---

## 4. 命令行参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--config` | 包内 `config.yaml` | 指定配置文件 |
| `--run-name` | `member2_bilstm_random_full_seed42` | 输出目录名 |
| `--rnn-type` | `lstm` | `lstm` / `gru` |
| `--embedding-dim` | `100` | 与 GloVe 6B-100d 对齐 |
| `--hidden-dim` | `128` | 每方向隐藏单元 |
| `--num-layers` | `1` | 循环层数 |
| `--dropout` | `0.3` | 循环输出 dropout |
| `--embedding-dropout` | `0.1` | embedding dropout |
| `--pooling` | `last` | `last` / `max` |
| `--epochs` | `20` | 轮数上限（早停会提前结束） |
| `--batch-size` | `128` | 批大小 |
| `--lr` | `1e-3` | 学习率 |
| `--weight-decay` | `0.01` | AdamW 权重衰减 |
| `--patience` | `4` | 早停耐心 |
| `--max-len` | `256` | 截断长度 |
| `--vocab-size` | `30000` | 词表规模 |
| `--seed` | `42` | 随机种子 |
| `--amp` / `--no-amp` | 开 | 混合精度（仅 CUDA 生效） |
| `--limit-train` / `--limit-val` | 全量 | 调试用样本上限 |
| `--smoke-test` | 关 | 只跑链路自检 |
| `--resume` | 无 | 从 `last_full.pt` 精确续训 |
| `--evaluate-test` | 关 | 训练后评估官方测试集（锁模后一次） |
| `--skip-checksum` | 关 | 跳过 SHA-256 校验（不建议） |

以上任意参数都等价于修改 `config.yaml` 里对应的一项，**优先级：命令行 > YAML**。

---

## 5. 产物说明

训练结束后 `runs/<run_name>/` 下会有：

| 文件 | 内容 | 用途 |
|---|---|---|
| `config.json` | 完整配置 + 环境快照 + 词表元信息 + 参数量 + git commit | 报告里写"实验设置" |
| `vocab.json` | 训练集词表（PAD=0, UNK=1） | **成员三必须复用这一份** |
| `training_log.csv` | 逐轮 lr / train_loss / train_acc / val_loss / val_acc / val_macro_f1 / 耗时 | 直接画图或贴表格 |
| `metrics.jsonl` | 同上，JSON Lines 格式 | 程序消费 / 中断后回看 |
| `history.json` | 逐轮指标 + `stop_reason` + 总耗时 + 峰值显存 | 报告里写"训练过程" |
| `metrics.json` | 汇总（best val、训练成本、测试集结果） | 成员五汇总表的数据源 |
| `best_model.pt` | 验证 Macro-F1 最佳的权重（自包含：含词表与配置） | 推理 / 成员五评估 |
| `last_full.pt` | 最新一轮完整状态（含优化器） | `--resume` 续训 |
| `training_curves.png` | loss / accuracy / Macro-F1 三条曲线 | 报告图 |
| `confusion_matrix.png` | 最佳轮次的验证集混淆矩阵 | 报告图 |
| `model_card.md` | 模型卡（架构 / 指标 / 成本 / 局限） | 报告 Model 部分素材 |
| `test_predictions.csv` | 逐样本预测（仅 `--evaluate-test` 时生成） | 成员五错误分析 |

> `runs/` 已被 `.gitignore` 忽略，**不要提交训练输出**；需要分享时把
> `metrics.json`、`history.json`、`training_log.csv`、`*.png`、`model_card.md` 拷到
> `experiments/<run_name>/` 再提交，或把 `.pt` 放到 Release。

---

## 6. 给成员三：预训练词向量的接入方式

```python
import numpy as np
from models.lstm_gru_random import RecurrentClassifier, Vocabulary, load_config

config = load_config()                                   # 与成员二完全相同的超参
vocab = Vocabulary.load("runs/member2_bilstm_random_full_seed42/vocab.json")   # ★ 必须复用同一份

def build_glove_matrix(vocab, glove_path, dim=100):
    """按词表行顺序把 GloVe 填进矩阵；未登录词保持随机初始化。"""
    rng = np.random.RandomState(42)
    matrix = rng.uniform(-0.1, 0.1, size=(len(vocab), dim)).astype("float32")
    matrix[vocab.token_to_id["<pad>"]] = 0.0
    with open(glove_path, encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip().split(" ")
            if len(fields) != dim + 1:
                continue
            index = vocab.token_to_id.get(fields[0])
            if index is not None:
                matrix[index] = np.asarray(fields[1:], dtype="float32")
    return matrix

model = RecurrentClassifier(vocab_size=len(vocab),
                            embedding_weights=build_glove_matrix(vocab, "glove.6B.100d.txt"),
                            **config["model"])
# 需要"冻结 embedding"的对照时再调用：
# model.freeze_embeddings()
```

**公平对比的四条硬约束**（写进报告时也这么写）：

1. 同一份 `processed_data/` 划分；
2. 同一份 `vocab.json`（词表顺序会影响 embedding 行对应关系）；
3. 同一个 `max_len=256`、同一 batch size 与 epochs 预算；
4. 唯一变量是 embedding 初始化来源（以及是否冻结，需单独报告）。

另外注意：本包的词表按 `(频次降序, 字典序)` 排序，**与历史 16k checkpoint 的 token id 不同**，
旧 embedding 不能按 id 直接搬运，必须按 token 重新映射。

---

## 7. 给成员四/五：评估与错误分析

```python
from models.lstm_gru_random import load_checkpoint, predict_texts, classification_metrics

model, vocab, payload = load_checkpoint("runs/member2_bilstm_random_full_seed42/best_model.pt")
rows = predict_texts(model, vocab, ["not bad at all", "worst meal ever"],
                     max_len=payload["max_len"])
# rows -> [{"text", "label", "sentiment", "prob_positive"}, ...]
```

批量打分（自动保留 `sample_id` / `label` 列，方便与真实标签合并）：

```bash
python -m models.lstm_gru_random.predict \
    --checkpoint runs/member2_bilstm_random_full_seed42/best_model.pt \
    --input-csv processed_data/test.csv \
    --output-csv runs/member2_bilstm_random_full_seed42/test_predictions.csv
```

`test_predictions.csv` 的列约定：`sample_id, label, prediction, prob_positive`。

---

## 8. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `PackageNotFoundError` 或 CSV 只有几行文字 | Git LFS 指针未拉取，执行 `git lfs install && git lfs pull` |
| Windows 上 DataLoader 卡住 | 默认已把 `num_workers` 设为 0；如需加速用 `--config` 显式设置 |
| CUDA out of memory | 降低 `--batch-size`（如 64）或 `--max-len`（如 192） |
| 训练中断 | 用 `--resume runs/<run>/last_full.pt` 精确续训，不需要重跑 |
| `SHA-256 与 manifest.json 不一致` | 数据被改动过，重新 `git lfs pull`；确认后可用 `--skip-checksum` 跳过 |
| 忘了 `--evaluate-test` 就跑完了 | 不要为了测评重跑训练；用 `predict.py --input-csv processed_data/test.csv` 打分即可 |

---

## 9. 复现清单（写报告用）

```bash
git clone https://github.com/yancanbethebest/comp7015-yelp-sentiment.git
cd comp7015-yelp-sentiment
git lfs install && git lfs pull
python -m pip install -r requirements-training.txt

python -m models.lstm_gru_random.train --smoke-test
python -m models.lstm_gru_random.train
python -m models.lstm_gru_random.train --rnn-type gru --run-name member2_bigru_random_full_seed42
```

报告里需要记录：**种子 42、训练/验证 503,979 / 55,998、词表 30,000（仅训练集拟合）、
`max_len=256`、AdamW lr=1e-3 / wd=0.01、batch=128、dropout 0.3 + 0.1、早停 patience=4、
GPU 型号、每轮耗时、总时长、峰值显存、参数量（BiLSTM 3,236,034 / BiGRU 3,177,154）**。
