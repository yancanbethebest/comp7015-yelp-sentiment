# 成员四 · Transformer 部分（DeBERTa-v3 / DistilBERT）

这个目录是我在 COMP7015 小组项目里负责的部分：Yelp 评论情感分类的 **Transformer 路线**。

小组的做法是分阶段的：先用 seed 42 跑 22 个候选配置筛一遍，选出胜者，再用 seed 123 和 2026 各重训一次确认结果稳定，然后冻结协议，最后只在官方 38,000 行测试集上评一次。我负责其中 Transformer 这一半的搜索，冠军配置也在这边儿，测试阶段的评估和训练成本对比也由我完成。

## 我做了什么

| 项目 | 内容 |
|---|---|
| 候选配置 | 9 个 Transformer 配置（DistilBERT 三组，覆盖不同长度、池化方式和学习率，加上 DeBERTa-v3-base 的几个变体） |
| 冠军模型 | `deberta_L384`：DeBERTa-v3-base，最大长度 384，学习率 2e-5，bf16，3 个 epoch |
| 确认种子 | seed 123 与 seed 2026，各完整重训 3 个 epoch |
| 测试评估 | 36 个冻结 run 全部在 38,000 行测试集上跑一遍 |
| 分析 | 验证集与测试集的一致性、训练成本与 F1 的换算、逐类别指标、配对种子比较 |

## 目录结构

```
member4/
├── README.md                     本文件（英文版）
├── README.zh-CN.md               本文件的中文版
├── requirements.txt
├── code/
│   ├── core.py                   数据读取、模型注册、指标计算、哈希校验
│   ├── prepare_full.py           从成员一的 processed_data 构建三份切分 + 词表
│   ├── run_experiment.py         单卡训练器（训练 / 评估）
│   ├── rnn_full.py               RNN 训练器（复算 imports 用）
│   ├── campaign.py               筛选 → 确认 → 冻结 → 测试 → 报告
│   ├── report.py                 指标重算、均值标准差、配对检验、画图
│   ├── recover.py                从已有权重恢复 run 的流程
│   ├── matrix.json               22 个候选配置（RNN 与 Transformer 都在里面）
│   └── original_rnn/             RNN 基线（成员二写的，成员三也在这套代码上做扩展）
├── configs/
│   └── transformer_configs.yaml  我负责的 9 个 Transformer 配置，人读版
```

## 运行环境

- Python 3.11 以上，一张 CUDA 卡（我们跑的时候用的是 96 GB 的 RTX PRO 6000 Blackwell）
- `transformers==4.57.1`，训练器里对这个版本有断言，版本不符会直接报错
- 用 `--precision bf16` 时需要显卡原生支持 bf16
- 依赖安装：`pip install -r requirements.txt`

全部流程都在单卡上跑，没有多卡或 TPU 分支。

## 数据准备

三份切分由成员一的 `processed_data/` 目录产出（`finalcode/member1/processed_data/{train,val,test}.csv`）。本目录放在 `finalcode/member4/` 时，从这儿执行：

```bash
python code/prepare_full.py --repo ../member1 --out ./data
```

`--repo` 指向的是**包含** `processed_data/` 的那一层目录。

脚本会做几件事：核对每份切分的 SHA-256，确认样本 ID 与正文在三份之间没有重叠，然后复制出 `train.csv`、`val.csv`、`test.csv`，再用训练集拟合 30k 词表（`vocab.json`）和 GloVe-100d 矩阵供 RNN 基线使用。行数应当是 503,979 / 55,998 / 38,000。

如果 CSV 被 Git LFS 存成了指针文件，脚本会直接拒绝并提示先 `git lfs pull`。

## 训练

跑冠军配置：

```bash
python code/run_experiment.py train \
  --data ./data --out ./runs/deberta_L384_s42 \
  --model deberta --length 384 --policy head --lr 2e-5 --seed 42 \
  --precision bf16 --micro-batch 8 --gradient-checkpointing
```

每个 epoch 结束后会写出一个自校验的完整 checkpoint 目录，包含 `model/`、`COMPLETE.json`、`metrics.json`、`validation.csv`、`training_state.pt`、`preprocessing.json`、`reload_check.json`。训练可以在 epoch 边界上用 `--resume` 续跑。

9 个配置可以逐个按 `configs/transformer_configs.yaml` 里的参数跑，也可以写个循环按顺序执行。

## 在冻结的测试集上评估

```bash
python code/run_experiment.py evaluate --data ./data --root ./runs --run deberta_L384_s42
```

会输出 38,000 行的 `predictions.csv` 和一组测试指标：accuracy、macro-F1、NLL、Brier、ECE。代码在 `FINAL_LOCK.json` 不存在时会拒绝碰测试集。

## 测试集结果

| 配置 | Macro-F1（3 个种子，均值 ± 标准差） | NLL | 训练耗时 |
|---|---|---|---|
| **deberta_L384** | **0.9831 ± 0.0004** | 0.0763 | 42,293 秒（3 个种子合计） |
| deberta_base | 0.9809 ± 0.0004 | 0.0920 | 20,158 秒 |
| distilbert_base | 0.9707 ± 0.0009 | 0.1328 | 3,375 秒 |
| distilbert_lr3 | 0.9701 ± 0.0003 | 0.1370 | 3,377 秒 |
| lstm_length256（RNN 最佳，成员三） | 0.9604 ± 0.0007 | 0.1077 | 2,726 秒 |

冠军的验证集 macro-F1 是 0.9821，测试集 0.9831，从验证到测试没有掉点。全部 36 个 run 的表和对比图都在项目报告里。

## 可复现性

- 每份切分的 SHA-256 写在 `code/core.py` 的 `PARENT_SHA` 里，加载时会重新核对，行数也有断言
- 模型权重钉在具体的 Hub 版本上（`microsoft/deberta-v3-base@8ccc9b6`、`distilbert-base-uncased@12040acc`）
- 确定性方面：固定种子，关掉 `cudnn.benchmark`，禁用 TF32，打开 `torch.use_deterministic_algorithms(True)`，并设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- `FINAL_LOCK.json` 记录了协议哈希、选择哈希和每个冻结的 run

## 关于变量名 `tokens` 的说明

静态检查工具偶尔会标记 `code/original_rnn/model.py`，因为它里面有个变量叫 `tokens`，旁边又出现 `logits`。

这儿的 `tokens` 用的是自然语言处理的常规含义：一个 batch 的评论文本经过 `Vocabulary.encode` 之后得到的 `(B, T)` 整数张量，每个数字是某个词在词表里的编号。整个项目里没有任何凭据、API 密钥或访问令牌，训练时也不会把样本内容写进日志，只记录标量指标。这条提示属于启发式规则造成的误报。

## 第三方库与素材

PyTorch、Hugging Face Transformers、NumPy、pandas、SciPy、Matplotlib、sentencepiece。GloVe-100d 词向量来自 gensim-data 的 `glove-wiki-gigaword-100` 发布包。`original_rnn/` 里的 RNN 模型与数据代码是成员二写的（成员三的 `lstm_pretrained` 也在这套代码上扩展），为了让本目录能独立跑通，这儿保留了一份。
