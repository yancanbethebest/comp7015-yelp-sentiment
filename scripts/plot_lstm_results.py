"""Plot separate learning curves and a comparison from completed LSTM runs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("experiments/lstm_100"))
    args = parser.parse_args()
    runs = {}
    for folder, title in (("random", "Random embedding BiLSTM"), ("glove", "GloVe embedding BiLSTM")):
        path = args.root / folder
        report = json.loads((path / "metrics.json").read_text())
        runs[title] = report
        history = report["history"]
        epochs = [r["epoch"] for r in history]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for ax, metric, label in zip(axes, ("loss", "accuracy", "macro_f1"), ("Cross-entropy loss", "Accuracy", "Macro-F1")):
            for split in ("train", "val"):
                ax.plot(epochs, [r[split][metric] for r in history], label=split)
            ax.axvline(report["best_epoch"], color="gray", linestyle="--", label="Best validation F1")
            ax.set(xlabel="Epoch", ylabel=label)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle(f"{title} | 16,000 train / 4,000 validation | seed 42")
        fig.tight_layout()
        fig.savefig(path / "training_curves.png", dpi=170)
        plt.close(fig)
        best = history[report["best_epoch"] - 1]
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = best["val"]["confusion_matrix"]
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="white" if cm[i][j] > max(map(max, cm)) / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"], xlabel="Predicted", ylabel="Actual", title=f"Validation confusion matrix\n{title}, epoch {report['best_epoch']}")
        fig.tight_layout()
        fig.savefig(path / "validation_confusion_matrix.png", dpi=170)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for title, report in runs.items():
        for ax, metric in zip(axes, ("accuracy", "macro_f1")):
            ax.plot([r["epoch"] for r in report["history"]], [r["val"][metric] for r in report["history"]], label=title)
            ax.set(xlabel="Epoch", ylabel=f"Validation {metric}")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.root / "comparison.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
