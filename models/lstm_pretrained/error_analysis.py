# -*- coding: utf-8 -*-
"""错误分析与阈值校准。

对一个训练好的 checkpoint 在验证集上做推理，然后分析：
1. 按评论长度分桶的错误率
2. 含否定词 vs 不含否定词的错误率
3. 按置信度分桶的错误率
4. 阈值校准：扫描 0.01~0.99，找 F1 最高的阈值

用法：
    python -m models.lstm_pretrained.error_analysis \
        --checkpoint <best_model.pt> \
        --data processed_data \
        --output experiments/member3_pretrained/error_analysis
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from models.lstm_gru_random import (
    SentimentDataset,
    collate_batch,
    classification_metrics,
    load_checkpoint,
)
from models.lstm_gru_random.data import load_split

NEGATION_WORDS = (
    "not", "no", "never", "n't", "don't", "do not", "can't", "cannot",
    "won't", "will not", "isn't", "aren't", "wasn't", "weren't", "didn't",
    "doesn't", "couldn't", "shouldn't", "wouldn't", "haven't", "hasn't",
    "hadn't", "ain't",
)


def has_negation(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in NEGATION_WORDS)


def error_rate(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 0.0
    return float((df["prediction"] != df["label"]).mean())


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def threshold_sweep(labels: np.ndarray, probs: np.ndarray):
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for t in thresholds:
        pred = (probs >= t).astype(np.int64)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append((float(t), precision, recall, f1, float((pred == labels).mean())))
    return rows


def run_predictions(checkpoint: Path, data_dir: Path, split: str, batch_size: int):
    model, vocab, payload = load_checkpoint(checkpoint, device=torch.device("cpu"))
    frame = load_split(split, data_dir)
    dataset = SentimentDataset(frame, vocab, int(payload["max_len"]), name=split)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )

    model.eval()
    sample_ids, y_true, y_pred, probs = [], [], [], []
    with torch.no_grad():
        for tokens, lengths, labels, ids in loader:
            logits = model(tokens, lengths)
            prob = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            y_true.extend(labels.tolist())
            y_pred.extend(prob.argmax(axis=1).tolist())
            probs.extend(prob[:, 1].tolist())
            sample_ids.extend(ids)

    df = frame[["sample_id", "text"]].copy()
    df = df.merge(
        pd.DataFrame({
            "sample_id": sample_ids,
            "label": y_true,
            "prediction": y_pred,
            "prob_positive": probs,
        }),
        on="sample_id",
        how="inner",
    )
    df["length"] = df["text"].apply(lambda t: len(vocab.encode(t, int(payload["max_len"]))))
    df["is_error"] = (df["prediction"] != df["label"]).astype(int)
    df["has_negation"] = df["text"].apply(has_negation)
    df["confidence"] = np.abs(df["prob_positive"] - 0.5)
    return df, int(payload["max_len"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m models.lstm_pretrained.error_analysis")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("processed_data"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/member3_pretrained/error_analysis"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split", default="val")
    args = parser.parse_args(argv)

    df, max_len = run_predictions(args.checkpoint, args.data, args.split, args.batch_size)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    overall = classification_metrics(df["label"].tolist(), df["prediction"].tolist())

    # 长度分桶
    bins = [0, 40, 100, 200, 10**9]
    bin_labels = ["<40", "40-100", "100-200", ">200"]
    length_rows = []
    for i in range(len(bins) - 1):
        sub = df[(df["length"] >= bins[i]) & (df["length"] < bins[i + 1])]
        length_rows.append((bin_labels[i], len(sub), error_rate(sub)))

    # 否定词
    neg = df[df["has_negation"]]
    non_neg = df[~df["has_negation"]]

    # 置信度分桶
    conf_rows = []
    for name, lo, hi in (
        ("high confidence (>0.3)", 0.3, 0.5),
        ("mid confidence (0.1-0.3)", 0.1, 0.3),
        ("low confidence (<0.1)", 0.0, 0.1),
    ):
        sub = df[(df["confidence"] >= lo) & (df["confidence"] < hi)]
        conf_rows.append((name, len(sub), error_rate(sub)))

    # 阈值扫描
    sweep = threshold_sweep(df["label"].to_numpy(), df["prob_positive"].to_numpy())
    sweep_df = pd.DataFrame(sweep, columns=["threshold", "precision", "recall", "f1", "accuracy"])
    best_row = sweep_df.loc[sweep_df["f1"].idxmax()]
    pred_05 = (df["prob_positive"] >= 0.5).astype(int)
    f1_05 = classification_metrics(df["label"].tolist(), pred_05.tolist())["macro_f1"]

    # 图 1：阈值曲线
    plt.figure(figsize=(7, 4))
    plt.plot(sweep_df["threshold"], sweep_df["f1"], label="F1")
    plt.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision")
    plt.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall")
    plt.axvline(0.5, color="grey", linestyle="--", linewidth=0.8)
    plt.axvline(best_row["threshold"], color="red", linestyle="--", linewidth=0.8,
                label=f"best threshold {best_row['threshold']:.2f}")
    plt.xlabel("Threshold (prob_positive)")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "threshold_curve.png", dpi=160)
    plt.close()

    # 图 2：长度分桶错误率
    plt.figure(figsize=(6, 3.5))
    names = [r[0] for r in length_rows]
    errs = [r[2] * 100 for r in length_rows]
    plt.bar(names, errs)
    plt.ylabel("Error rate (%)")
    plt.title("Error rate by review length")
    for i, v in enumerate(errs):
        plt.text(i, v + 0.05, f"{v:.2f}%", ha="center")
    plt.tight_layout()
    plt.savefig(output / "error_by_length.png", dpi=160)
    plt.close()

    # 图 3：否定词错误率
    plt.figure(figsize=(4.5, 3.5))
    plt.bar(["with negation", "without negation"],
            [error_rate(neg) * 100, error_rate(non_neg) * 100])
    plt.ylabel("Error rate (%)")
    plt.title("Error rate vs negation")
    plt.tight_layout()
    plt.savefig(output / "error_by_negation.png", dpi=160)
    plt.close()

    # Markdown 报告
    lines = [
        "# 错误分析与阈值校准",
        "",
        f"- 模型：{Path(args.checkpoint).name}",
        f"- 数据：{args.split}，共 {len(df):,} 条，max_len={max_len}",
        f"- 总体：Accuracy {pct(overall['accuracy'])}，Macro-F1 {pct(overall['macro_f1'])}",
        f"- 混淆矩阵（行=真实，列=预测）：{overall['confusion_matrix']}",
        "",
        "## 1. 按评论长度分桶",
        "",
        "| 长度 | 样本数 | 错误率 |",
        "|---|---:|---:|",
    ]
    for name, count, err in length_rows:
        lines.append(f"| {name} | {count:,} | {pct(err)} |")

    lines += [
        "",
        "## 2. 否定词",
        "",
        "| 类别 | 样本数 | 错误率 |",
        "|---|---:|---:|",
        f"| 含否定词 | {len(neg):,} | {pct(error_rate(neg))} |",
        f"| 不含否定词 | {len(non_neg):,} | {pct(error_rate(non_neg))} |",
        "",
        "## 3. 按置信度分桶",
        "",
        "| 置信度 | 样本数 | 错误率 |",
        "|---|---:|---:|",
    ]
    for name, count, err in conf_rows:
        lines.append(f"| {name} | {count:,} | {pct(err)} |")

    lines += [
        "",
        "## 4. 阈值校准",
        "",
        f"- 默认阈值 0.5：Macro-F1 {pct(f1_05)}",
        f"- 最佳阈值 {best_row['threshold']:.2f}：Macro-F1 {pct(best_row['f1'])}，"
        f"Precision {pct(best_row['precision'])}，Recall {pct(best_row['recall'])}",
        "",
        "![threshold curve](threshold_curve.png)",
        "![error by length](error_by_length.png)",
        "![error by negation](error_by_negation.png)",
    ]
    (output / "error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("overall accuracy", pct(overall["accuracy"]), "macro_f1", pct(overall["macro_f1"]))
    print("default 0.5 f1", pct(f1_05), "| best threshold",
          f"{best_row['threshold']:.2f}", "f1", pct(best_row["f1"]))
    print("negation error", pct(error_rate(neg)), "| non-negation error", pct(error_rate(non_neg)))
    print("written to", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())