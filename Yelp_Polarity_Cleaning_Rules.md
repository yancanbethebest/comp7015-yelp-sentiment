# Yelp Review Polarity 数据清洗规则

> 版本：v1.0 | 日期：2026-10-27 | 用途：NLP 情感二分类小组项目的数据交付规范

## 1. 目标与原则

数据来源：[Hugging Face — fancyzhx/yelp_polarity](https://huggingface.co/datasets/fancyzhx/yelp_polarity)。官方训练集 560,000 条、测试集 38,000 条；标签 `0=negative`、`1=positive`。

- **最小必要清洗**：统一数据质量，不改变情感语义；保留否定词（not、never）、标点、数字、大小写和 emoji。
- **公平比较**：所有模型使用相同的样本 ID、训练/验证/测试划分和标签。
- **防止数据泄漏**：先检查官方 train/test 重叠，随后只从官方训练池生成 train/val；测试样本只来自官方 test。TF-IDF 词表、标准化参数和 tokenizer 词表等只在训练集拟合。
- **可追溯**：保留原始文本、清洗代码、随机种子、清洗日志、数据版本和抽样索引。

## 2. 清洗规则总表

| 编号 | 规则 | 具体操作 | 处理方式 | 优先级 |
|---|---|---|---|---|
| R01 | 缺失文本 | 检查 null、NaN、空字符串、仅空白字符 | 删除并计数 | 必须 |
| R02 | 标签合法性 | 仅允许整数标签 0/1；不自动推断或修复其他值 | 隔离异常记录并计数 | 必须 |
| R03 | Unicode 规范化 | 统一常见异常空格与引号；对 NFKC 先做抽样验证，避免语义变化 | 保留变更前后样本 | 推荐 |
| R04 | HTML 实体解码 | 例如 `&amp;` → `&`、`&quot;` → `"` | 解码并记录 | 推荐 |
| R05 | HTML 标签 | 仅移除确认属于 HTML 的标签，保留其文本内容；避免误删 `<3` 等表达 | 清洗并抽查 | 推荐 |
| R06 | 空白标准化 | 换行、制表符、重复空格 → 单个空格；首尾 trim | 规范化 | 必须 |
| R07 | URL 规范化 | 将明确 URL 替换为统一标记 `[URL]`，不误删普通单词 | 替换并计数 | 推荐 |
| R08 | 邮箱规范化 | 如需隐私处理，将邮箱替换为 `[EMAIL]` | 可选替换 | 可选 |
| R09 | 完全重复文本 | 在规范化后检测完全相同文本；同标签重复保留一条 | 去重并计数 | 必须 |
| R10 | 冲突标签 | 相同文本出现 0 与 1 两种标签 | 隔离冲突样本，人工审查；不擅自改标签 | 必须 |
| R11 | 极短文本 | 标记少于 3 个按空格分隔词的文本 | 仅标记，不自动删除 | 推荐 |
| R12 | 超长文本 | 统计字符数、词数及 P50/P95/P99；模型训练阶段再决定截断 | 仅统计，不截断 | 推荐 |
| R13 | 异常字符 | 检测不可见控制字符、替换字符 `�`、乱码迹象 | 去除无意义控制字符；其余标记复核 | 推荐 |
| R14 | 跨集合重复 | 检查官方 train/test、后续 train/val/test 的文本重叠 | 优先保留官方 test，剔除训练池重叠；重新检查 | 必须 |
| R15 | 类别分布 | 检查各集合 0/1 数量与比例 | 分层抽样，记录实际比例 | 必须 |

### 规则实施顺序

1. 读取原始官方 train/test，保存原始行 ID 和来源分区。
2. 执行 R01–R08、R13，输出 `clean_text`；不要覆盖 `raw_text`。
3. 执行 R09–R10：先在各官方分区内去重，再检测跨官方分区重叠；冲突样本进入审查清单。
4. 执行 R11–R12 统计，不按文本长度自动删除。
5. 执行 R14：对训练池与官方测试池做重叠排除，确保最终分区不泄漏。
6. 执行 R15：固定随机种子 `42`，按标签分层抽样和划分。
7. 导出交付数据、日志和质量报告，并执行验收。

> 注意：去重与冲突判断应基于统一的 `clean_text`；对大小写差异、标点差异等近重复，仅生成可疑清单，不应自动合并，以免删除有情感差异的评论。

## 3. 数据划分方案

| 文件 | 来源 | 目标条数 | 负面（0） | 正面（1） |
|---|---|---:|---:|---:|
| `train.csv` | 官方训练池 | 16,000 | 8,000 | 8,000 |
| `val.csv` | 官方训练池 | 4,000 | 2,000 | 2,000 |
| `test.csv` | 官方测试池 | 5,000 | 2,500 | 2,500 |

- 抽样必须在清洗、去重和跨集合重叠排除后进行；固定 `random_state=42`。
- 训练池中先分层选取 20,000 条，再分层划分 16,000/4,000；官方测试池单独分层选取 5,000 条。
- 若有效样本不足，**不得通过复制样本凑数**；记录实际规模及偏差。
- 所有模型使用完全一致的划分；测试集仅用于最终评估，不用于调参或挑选清洗规则。

## 4. 明确不在共享清洗阶段执行的操作

- **不统一删除停用词**：`not good` 中的 `not` 会改变情感极性。
- **不删除所有标点**：`Great!!!` 的情感强度信息可能有用。
- **不删除所有数字**：`1 star`、`waited 2 hours` 有潜在情感含义。
- **不统一小写**：保留原文供深度学习模型使用；传统模型可在自身流水线中设置小写。
- **不提前做词干提取、词形还原、TF-IDF、Tokenization 或 Padding**：这些属于模型特定预处理。
- **不基于标签或模型预测结果删除“难分类”文本**：避免人为提高评估结果。

## 5. 清洗示例

**原文**：

```text
  <p>I didn't like this restaurant!!!</p>\n  The food was cold &amp; service was terrible.
```

**清洗后**：

```text
I didn't like this restaurant!!! The food was cold & service was terrible.
```

保留 `didn't`、`!!!` 和负面词 `cold`、`terrible`。

## 6. 输出文件与字段约定

```text
Yelp_Sentiment_Project/
├── data/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── scripts/
│   └── preprocessing.py
├── reports/
│   ├── cleaning_summary.csv
│   └── flagged_records.csv
├── requirements.txt
└── README.md
```

CSV 推荐字段：`sample_id,text,label`。其中 `sample_id` 为可追踪到原始来源的稳定唯一标识，`text` 为 `clean_text`，`label` 为 0/1。内部审计文件另保留 `raw_text`、`source_split`、`source_row_id`、清洗规则版本。

`cleaning_summary.csv` 至少包含：原始条数、各规则影响条数、空值数、去重数、冲突数、跨分区重叠数、各集合实际条数及正负比例、长度分位数。`flagged_records.csv` 包含待人工复核的异常及原因。

## 7. 验收标准

- [ ] 三个 CSV 可用 `pandas.read_csv` 正常读取，编码为 UTF-8。
- [ ] 标签仅有 0/1，文本不为空。
- [ ] 各文件内无完全重复文本，三个集合间无完全重复文本。
- [ ] 训练、验证数据均来自官方训练池；测试数据仅来自官方测试池。
- [ ] 类别分布和实际数据量与报告一致。
- [ ] 随机种子、清洗版本、依赖版本和处理顺序可复现。
- [ ] 所有模型组员拿到相同的文件版本或校验和（SHA-256）。

## 8. 可选清洗消融实验

- A：原始文本（基线）。
- B：本规范的轻量清洗（推荐主版本）。
- C：轻量清洗 + 传统模型专属的小写/停用词/词形还原（对照实验）。

在相同的训练/验证划分上比较 Accuracy、Macro-F1、训练耗时。不要使用测试集挑选最佳清洗方案。

## 9. 参考资料

1. [Yelp Review Polarity — Hugging Face](https://huggingface.co/datasets/fancyzhx/yelp_polarity)
2. [Yelp Review Sentiment Analysis — 示例项目](https://juanxtron.github.io/YelpReviewSentiment.html)
3. [scikit-learn：train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
4. [scikit-learn：TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

> 本文是针对小组多模型对比任务制定的操作规范，不代表所有参考项目均采用相同清洗策略。
