# 组员下一步工作指南

## 目前可以直接使用的成果

- `processed_data/`：固定训练 16,000 条、验证 4,000 条、测试 5,000 条。数据来源、清洗审计和校验和在 `reports/`，所有模型应使用同一划分。
- [两种 LSTM 延长训练结果](experiments/lstm_100/README.md)：两组独立训练；原定各 100 轮，后按用户要求手动提前停止。随机组完成 71 轮，最佳模型在第 69 轮，验证 Macro-F1 89.25%；GloVe 组完成 35 轮，最佳在第 10 轮，验证 Macro-F1 90.12%。这是手动停止，不是预先设定的 early-stopping patience。
- [最佳模型下载](https://github.com/yancanbethebest/comp7015-yelp-sentiment/releases/tag/lstm-extended-2026-10-31)：下载 `random_lstm_extended.tar.gz` 和 `glove_lstm_extended.tar.gz`。两份都包含 **best_model.pt**、指标和图片，保存的是验证 F1 最佳轮次，并非最后一轮。
- [此前 BERT 结果](reports/validation_10_epochs.md)：BERT 的 10 轮实验中，第 2 轮验证 Macro-F1 最佳，为 95.20%。其 checkpoint 在 [10 轮实验 Release](https://github.com/yancanbethebest/comp7015-yelp-sentiment/releases/tag/checkpoints-2026-10-30)。

这些实验使用的是清洗后的抽样数据，不能写成已经在全部官方训练数据上完成训练。当前延长训练没有评估测试集；仓库中的历史试跑曾查看过测试结果，报告应如实区分。

## 先确认可以加载模型

在克隆后的仓库目录安装 `requirements-training.txt`，下载上述两个压缩包并解压。以下命令适用于 Linux / Colab 终端；Colab 代码单元中的 shell 命令前加 `!`。

```bash
tar -xzf random_lstm_extended.tar.gz
tar -xzf glove_lstm_extended.tar.gz
python scripts/predict_lstm.py --checkpoint random/best_model.pt --text "Great food and friendly service."
python scripts/predict_lstm.py --checkpoint glove/best_model.pt --text "Terrible food and rude service."
```

标签 `0` 表示负面，`1` 表示正面。脚本读取 checkpoint 自带的词表、最大长度和模型配置，复用训练时的小写化、分词和 padding。**仅加载 GloVe 模型进行预测，不需要重新下载 GloVe 原始文件**。输出概率并未做校准，不应当作可靠置信度。两个 checkpoint 用于推理和评估，未保存优化器状态，不能声称可以精确恢复被中断的训练。

## 建议分工与交付物

| 工作 | 下一步操作 | 需要交付 |
|---|---|---|
| LSTM 实验负责人 | 在固定划分上做超参数对比；随机与 GloVe 使用相同结构和训练预算 | 参数表、验证指标、最佳配置、训练时间 |
| BERT 负责人 | 微调已有预训练 BERT，按验证 F1 选最佳模型，记录显存与耗时 | BERT 曲线、最佳 checkpoint、与 LSTM 的性能及成本对比 |
| 评估负责人 | 配置选择完成后，统一评估保存的最佳模型，避免为测试成绩反复调参 | Accuracy、每类及 Macro Precision/Recall/F1、混淆矩阵、错误案例 |
| 报告负责人 | 合并数据说明、模型方法、实验结果和局限 | 课程报告、图表、复现说明；明确区分验证结果和测试结果 |

## LSTM 超参数实验：补足 Minimum Requirements

当前只是词向量对比和延长训练，**还没有完成系统的超参数研究**。建议采用一次只改一个参数的小规模对照，先用相同 20 轮预算：

| 参数 | 当前基线 | 建议对照 |
|---|---:|---|
| Learning rate | 0.001 | 0.0005、0.002 |
| Batch size | 128 | 64、256 |
| Dropout | 0.3 | 0.5 |
| Hidden units（每方向） | 128 | 64、256 |

先在验证集上筛选配置；每次实验使用独立目录，不覆盖现有结果。两种 embedding 都使用相同候选参数和预算。对最终选出的配置，再用训练种子 42、43、44 重复，报告均值与标准差；**保持数据划分不变**。20 轮是筛选预算，并不证明模型已经收敛。

```bash
python scripts/train_lstm_random.py --epochs 20 --dropout 0.5 --seed 42 --output experiments/tuning/random_dropout05_seed42
python scripts/train_lstm_glove.py --epochs 20 --dropout 0.5 --seed 42 --embedding-file glove.6B.100d.txt --output experiments/tuning/glove_dropout05_seed42
```

这里的两种入口共享 `train_lstm.py` 的训练实现，便于公平对比。GloVe 训练需要原始 100 维词向量文件；默认参与微调。不要在筛选阶段加 `--evaluate-test`。该选项会在训练后评估测试集，不是用于直接加载已发布模型的命令。

## BERT 与最终评估应注意什么

1. BERT 使用自己的 tokenizer 和预训练词表；LSTM 词表只由训练集生成。保持样本划分一致，分别说明文本处理方式和截断长度。
2. BERT 是情感分类微调，不是从头预训练；GloVe 是预训练词向量初始化，也不意味着整个 LSTM 已预训练。
3. 比较验证性能之外，还记录训练时间、推理时间、GPU 型号、峰值显存、参数量和实现复杂度。当前两组 LSTM 曾同时占用同一 GPU，其墙钟耗时不能直接当作公平的计算成本比较。
4. 锁定配置与选择规则后，再对各模型最佳 checkpoint 做同一测试集评估；保存逐样本预测用于错误分析。测试结果不能反过来用于选超参数或最佳 epoch。
5. 分析否定表达、长文本截断、混合情感等错误案例。正式报告应说明测试集曾在历史试跑中被查看这一局限，不能称其从未被接触。

## 报告中应包含的图表

- 每种 LSTM 的训练/验证 loss、Accuracy、Macro-F1 曲线，以及最佳验证轮次标记。
- 两种 LSTM 的验证曲线对比与最佳轮次混淆矩阵。
- 随机 LSTM、GloVe LSTM、BERT 的统一指标对比表，明确数据集、划分、最佳轮次和种子。
- 超参数影响表及最终多种子结果；最终测试结果另列，不与验证结果混为一谈。

所有正式提交文件只包含代码、数据说明、指标、图片和 checkpoint 下载链接，不包含服务器地址、密码、访问令牌或私有路径。
