# -*- coding: utf-8 -*-
"""训练曲线与混淆矩阵 / Learning curves and confusion matrix.

绘图是**可选依赖**：matplotlib 只在真正画图时才导入，
因此没有安装 matplotlib 的环境依然可以训练，只是不生成 png。
风格与仓库已有的 ``experiments/lstm_100/*.png`` 保持一致，方便报告里并排比较。

Plotting is optional: matplotlib is imported lazily inside the functions, so a
run without matplotlib still trains — it just skips the figures.
"""

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


def _pyplot():
    """延迟导入 matplotlib，并固定无界面后端 / lazy import with a headless backend."""
    import matplotlib

    matplotlib.use("Agg")          # 服务器 / Notebook 无显示环境也能存图
    import matplotlib.pyplot as plt

    return plt


def plot_training_curves(history: List[Dict[str, Any]], best_epoch: int,
                         out_path: Path, title: str) -> Path:
    """画 loss / accuracy / Macro-F1 三条曲线（训练 vs 验证）。

    Args:
        history: 每轮的记录列表，元素至少含 ``epoch``、``train``、``val`` 三个键，
            其中 ``train`` / ``val`` 各含 ``loss`` / ``accuracy`` / ``macro_f1``。
        best_epoch: 最佳验证轮次；``0`` 表示不画竖线。
        out_path: 输出 png 路径。
        title: 图标题。

    Returns:
        写入的路径。
    """
    plt = _pyplot()
    epochs = [int(record["epoch"]) for record in history]
    specs = [("loss", "Cross-entropy loss"),
             ("accuracy", "Accuracy"),
             ("macro_f1", "Macro-F1")]

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for axis, (key, label) in zip(axes, specs):
        axis.plot(epochs, [record["train"][key] for record in history],
                  label="train", linewidth=1.6)
        axis.plot(epochs, [record["val"][key] for record in history],
                  label="validation", linewidth=1.6)
        if best_epoch:
            axis.axvline(best_epoch, color="grey", linestyle="--", linewidth=1.0,
                         label=f"best epoch {best_epoch}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    figure.suptitle(title)
    figure.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=170)
    plt.close(figure)
    return out_path


def plot_confusion_matrix(confusion: Sequence[Sequence[int]], out_path: Path,
                          title: str) -> Path:
    """画混淆矩阵（行 = 真实，列 = 预测）/ plot the confusion matrix.

    Args:
        confusion: ``[[TN, FP], [FN, TP]]``。
        out_path: 输出 png 路径。
        title: 图标题。

    Returns:
        写入的路径。
    """
    plt = _pyplot()
    matrix = np.asarray(confusion, dtype=np.int64)

    figure, axis = plt.subplots(figsize=(4.8, 4.2))
    image = axis.imshow(matrix, cmap="Blues")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:,}",
                      ha="center", va="center", fontsize=11,
                      color="white" if matrix[row, column] > threshold else "black")

    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    axis.set_xticklabels(["Negative (0)", "Positive (1)"])
    axis.set_yticklabels(["Negative (0)", "Positive (1)"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=170)
    plt.close(figure)
    return out_path
