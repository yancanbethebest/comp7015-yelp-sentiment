# -*- coding: utf-8 -*-
"""评估指标与评估循环 / Metrics and the evaluation loop.

自己实现 Accuracy / Precision / Recall / F1 / 混淆矩阵，**不依赖 scikit-learn**：
口径完全可控，成员五可以直接复用同一份实现，避免"两个模型用了不同公式"的争议。

约定 / convention
-----------------
* 混淆矩阵第一个维度是**真实**标签，第二个维度是**预测**标签；
* 类别顺序固定为 ``[0 = negative, 1 = positive]``；
* 选模指标是 **Macro-F1**（正负类均衡时与 accuracy 接近，但更抗类别偏斜）。
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int],
                           labels: Sequence[int] = (0, 1)) -> Dict[str, Any]:
    """计算二分类指标 / compute binary classification metrics.

    Args:
        y_true: 真实标签序列。
        y_pred: 预测标签序列。
        labels: 类别顺序，默认 ``(0, 1)``。

    Returns:
        含 ``accuracy``、macro/weighted 平均的 ``precision``/``recall``/``f1``、
        逐类指标（含 ``support``）与 ``confusion_matrix`` 的字典。

    Raises:
        ValueError: 两个序列长度不一致，或出现了 ``labels`` 之外的标签。
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true 与 y_pred 长度不一致 / length mismatch")
    if y_true.size == 0:
        raise ValueError("评估集为空 / empty evaluation set")

    label_list = list(labels)
    index_of = {label: position for position, label in enumerate(label_list)}
    unknown = set(np.unique(y_true).tolist()) | set(np.unique(y_pred).tolist())
    unknown -= set(label_list)
    if unknown:
        raise ValueError(f"出现标签顺序之外的标签 / unexpected labels: {sorted(unknown)}")

    confusion = np.zeros((len(label_list), len(label_list)), dtype=np.int64)
    for truth, guess in zip(y_true.tolist(), y_pred.tolist()):
        confusion[index_of[truth], index_of[guess]] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1_scores: List[float] = []
    supports: List[int] = []

    for label in label_list:
        position = index_of[label]
        tp = int(confusion[position, position])
        fp = int(confusion[:, position].sum()) - tp
        fn = int(confusion[position, :].sum()) - tp
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        per_class[str(label)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        supports.append(support)

    weights = np.asarray(supports, dtype=np.float64)
    weighted_f1 = float(np.average(f1_scores, weights=weights)) if weights.sum() else 0.0

    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "samples": int(y_true.size),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: Optional[torch.device] = None, use_amp: bool = False,
             collect_predictions: bool = False
             ) -> Tuple[Dict[str, Any], Optional[List[Dict[str, Any]]]]:
    """在给定 DataLoader 上评估模型 / evaluate a model on one loader.

    Args:
        model: 待评估模型。
        loader: 使用 :func:`data.collate_batch` 的 DataLoader。
        criterion: 损失函数（``CrossEntropyLoss``）。
        device: 设备；``None`` 表示 CPU。
        use_amp: 是否用混合精度前向。**最终报告一律用 False（FP32）**，
            避免半精度的数值噪声影响指标。
        collect_predictions: 为 ``True`` 时额外返回逐样本预测，供成员五做错误分析。

    Returns:
        ``(metrics, rows)``：``metrics`` 含 ``loss`` 与全部分类指标；
        ``rows`` 在未开启收集时为 ``None``。
    """
    device = device if device is not None else torch.device("cpu")
    model.eval()

    total_loss = 0.0
    total_samples = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    rows: Optional[List[Dict[str, Any]]] = [] if collect_predictions else None

    for tokens, lengths, labels, sample_ids in loader:
        tokens = tokens.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(tokens, lengths)
            loss = criterion(logits, labels)

        total_loss += float(loss.detach().item()) * labels.size(0)
        total_samples += labels.size(0)

        # softmax 在 FP32 上算，避免 FP16 溢出 / probabilities in FP32
        probabilities = torch.softmax(logits.detach().float(), dim=-1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(predictions.tolist())

        if rows is not None:
            for sample_id, truth, guess, probability in zip(
                sample_ids, labels.detach().cpu().tolist(), predictions.tolist(),
                probabilities[:, 1].tolist(),
            ):
                rows.append({
                    "sample_id": sample_id,
                    "label": int(truth),
                    "prediction": int(guess),
                    "prob_positive": float(probability),
                })

    metrics = classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, total_samples)
    return metrics, rows
