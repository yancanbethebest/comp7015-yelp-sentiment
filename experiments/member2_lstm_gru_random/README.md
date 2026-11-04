# member2_lstm_gru_random —— 成员二训练结果（可提交部分）

本目录存放成员二（随机初始化 Embedding 的 LSTM/GRU 基础模型）**可以进 git 的结果摘要**。

> 为什么不直接提交 `runs/`：完整训练目录里有 `best_model.pt`（13 MB）与 `last_full.pt`（39 MB），
> 而且 `runs/` 已在 `.gitignore` 中。仓库里只保留**体积小、可审阅**的指标、日志与图片，
> 权重需要分享时走 GitHub Release 或 Git LFS。

## 目录结构

```text
experiments/member2_lstm_gru_random/
├── README.md                # 本文件：两次训练的结果汇总与复现命令
├── bilstm/                  # ← runs/member2_bilstm_random_full_seed42/
│   ├── metrics.json         # 汇总指标（成员五汇总表的数据源）
│   ├── history.json         # 逐轮指标 + stop_reason + 训练成本
│   ├── training_log.csv     # 逐轮 loss/accuracy/Macro-F1 表格（可直接画图）
│   ├── config.json          # 完整配置 + 环境快照 + 词表元信息 + 参数量
│   ├── model_card.md        # 模型卡（报告 Model 部分素材）
│   ├── training_curves.png  # 训练/验证曲线
│   └── confusion_matrix.png # 验证集混淆矩阵
└── bigru/                   # ← runs/member2_bigru_random_full_seed42/（同上）
```

## 命名规范（全组统一，避免混淆）

| 类别 | 命名规则 | 本次实例 |
|---|---|---|
| 代码包（固定） | `models/lstm_gru_random/` | 成员二专用，不与 `models/lstm_random/`（历史 16k 初步模型）混淆 |
| 运行目录 | `runs/member2_<单元>_<embedding>_<数据规模>_seed<种子>/` | `runs/member2_bilstm_random_full_seed42/` |
| 提交的结果 | `experiments/member2_lstm_gru_random/<单元>/` | `experiments/member2_lstm_gru_random/bilstm/` |
| 报告章节 | `reports/member2_model_<内容>.md` | `reports/member2_model_random_embedding_lstm_gru.md` |
| 权重（可选发布） | `models/lstm_gru_random/best_model.pt`（走 LFS）或 Release 附件 | — |

三条约定：

1. **`member2_` 前缀**表示"成员二产出的、全量正式训练"的内容；仓库里原有的 `lstm_random` / `lstm_glove` / `bert`
   是**历史 16k 抽样**的初步 checkpoint，不属于成员二的正式结果。
2. 目录名里写清 `<单元>`（bilstm/bigru）、`<embedding>`（random/glove）、`<数据规模>`（full/20k/…）与 `seed`，
   任何人看到一个目录名就知道"谁、用什么、在多少数据上、哪个种子"跑的。
3. 结果文件与报告文件**同前缀**（`member2_`），评审时一眼能对上号。

## 结果汇总（2026-11-04 本地复现）

**实验条件**：全量清洗划分（train 503,979 / val 55,998），词表 30,000（仅训练集拟合），
`max_len=256`，AdamW lr=1e-3 / wd=0.01，batch=128，dropout 0.3 + 0.1，AMP，种子 42，5 轮预算（早停未触发）。
**硬件**：NVIDIA GTX 1650 Max-Q（4 GB）/ PyTorch 2.6.0+cu124 / CUDA 12.4。
**测试集未评估**（留到成员五统一最终评估）。

| 模型 | 完成/最佳轮次 | 验证 Accuracy | 验证 Macro-P | 验证 Macro-R | 验证 Macro-F1 | 每轮耗时 | 总时长 | 峰值显存 | 参数量 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BiLSTM | 5 / **3** | 95.33% | 95.34% | 95.33% | **95.33%** | 567 s | 47.3 min | 248.2 MB | 3,236,034 |
| BiGRU | 5 / **2** | 95.65% | 95.66% | 95.65% | **95.65%** | 422 s | 35.2 min | 227.4 MB | 3,177,154 |

**验证集混淆矩阵**（行 = 真实，列 = 预测）：

| 模型 |  | 预测负向 | 预测正向 |
|---|---|---:|---:|
| BiLSTM | 真实负向 | 26,799 | 1,200 |
| BiLSTM | 真实正向 | 1,413 | 26,586 |
| BiGRU | 真实负向 | 26,997 | 1,002 |
| BiGRU | 真实正向 | 1,432 | 26,567 |

**逐轮验证 Macro-F1**

| 轮次 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| BiLSTM | 94.55% | 95.00% | **95.33%** | 95.33% | 95.13% |
| BiGRU | 95.29% | **95.65%** | 95.61% | 95.41% | 95.09% |

**结论（写给报告）**

1. 两个模型都在 2–3 轮见顶，之后训练准确率继续上升（BiLSTM 末轮 97.80%、BiGRU 98.61%）而验证指标下滑 → 典型过拟合，故必须按验证 Macro-F1 保存最佳 checkpoint。
2. 同数据、同词表、同预算下，**BiGRU 的验证 Macro-F1 比 BiLSTM 高 0.32 个百分点**，同时**每轮快 1.34 倍**、参数少 1.8%。
3. 与历史 **16k 抽样**基线（Random BiLSTM 89.25%）相比提升约 6 个百分点，主要来自数据量（16k → 504k），这正说明"正式实验必须在全量上做"。
4. 局限：单种子、5 轮预算、未评估测试集；正式提交前建议在 RTX 4080 上按 20 轮预算补跑并补种子 43/44。

## 复现命令

```bash
# 环境
git lfs install && git lfs pull
python -m pip install -r requirements-training.txt

# 链路自检（1~2 分钟）
python -m models.lstm_gru_random.train --smoke-test

# 两个正式实验（其余超参完全一致，只有循环单元不同）
python -m models.lstm_gru_random.train --epochs 5 --run-name member2_bilstm_random_full_seed42
python -m models.lstm_gru_random.train --epochs 5 --rnn-type gru \
    --run-name member2_bigru_random_full_seed42
```

> 本目录的结果由上述命令产生；原始完整产物（含权重）在本地 `runs/<run_name>/`，未入库。
