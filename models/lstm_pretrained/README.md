# lstm_pretrained —— 成员三 · 预训练词向量包

可复用代码包：加载 GloVe/Word2Vec 预训练词向量，按成员二词表对齐后注入 BiLSTM/BiGRU，做冻结/微调对比与超参数调优。

复用关系：模型、词表、训练循环、评估、绘图全部复用 `models/lstm_gru_random`，本包只新增“词向量加载与注入”。

## 目录结构

| 文件 | 作用 |
|---|---|
| embeddings.py | GloVe/Word2Vec 加载 + 按词表对齐 + 覆盖率统计 |
| train.py | 训练入口 CLI：注入词向量、冻结开关、调用成员二训练循环 |
| error_analysis.py | 错误分析 + 阈值校准 |
| config.yaml | 默认配置 |
| __init__.py | 包入口 |

## 快速开始

    # 安装依赖
    pip install -r requirements-training.txt

    # 小样本冒烟（先确认流水线）
    python -m models.lstm_pretrained.train --embedding-source random --limit-train 2000 --limit-val 500 --epochs 2 --batch-size 32 --max-len 128

    # 20k 冻结
    python -m models.lstm_pretrained.train --embedding-source glove --embedding-file glove.txt --freeze-embedding --limit-train 20000 --limit-val 4000 --epochs 20 --batch-size 64 --max-len 128 --patience 5

    # 全量微调
    python -m models.lstm_pretrained.train --embedding-source glove --embedding-file glove.txt --epochs 50 --patience 3 --batch-size 64 --max-len 256

## 命令行参数（train.py）

| 参数 | 说明 |
|---|---|
| --embedding-source | random / glove / word2vec |
| --embedding-file | 词向量文本文件路径 |
| --freeze-embedding | 冻结词向量（不传 = 微调） |
| --run-name / --data / --runs-dir | 运行标识与路径 |
| --limit-train / --limit-val | 小样本上限（全量不传） |
| --max-len / --batch-size / --lr / --hidden-dim / --dropout / --weight-decay / --epochs / --patience | 超参数 |
| --reuse-vocab | 复用已有 vocab.json |
| --evaluate-test | 锁定配置后评估测试集（一般不开启） |

## 输出

每个实验在 `runs/<run_name>/` 下输出：metrics.json、history.json、training_log.csv、config.json、vocab.json、best_model.pt、training_curves.png、confusion_matrix.png、model_card.md。

## 公平对比的关键约束

1. 复用成员二的 vocab.json（或按相同确定性规则重建），保证词表顺序一致。
2. 同一份数据划分、同样 max_len 与预算，唯一变量是词向量来源（随机 / GloVe / Word2Vec）与是否冻结。

## 词向量文件

- GloVe 6B-100d：https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz
- 解压后得到文本文件，用 `--embedding-file` 传入；大文件不要提交 git。