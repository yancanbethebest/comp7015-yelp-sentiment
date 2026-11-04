# -*- coding: utf-8 -*-
"""训练主流程 / Training pipeline.

一条命令跑完 / one command does everything:

    python -m models.lstm_gru_random.train                    # 用包内 config.yaml
    python -m models.lstm_gru_random.train --rnn-type gru     # 换循环单元
    python -m models.lstm_gru_random.train --smoke-test       # 1 分钟链路自检
    python -m models.lstm_gru_random.train --evaluate-test    # 锁定配置后才加

产物 / artefacts（全部写入 ``runs/<run_name>/``）::

    config.json          完整配置 + 环境快照 + 词表元信息 + 参数量
    vocab.json           训练集词表（PAD=0, UNK=1），成员三复用同一份
    metrics.jsonl        逐轮追加的日志（中断后仍可查看）
    training_log.csv     逐轮 loss/accuracy 表格（可直接贴进报告或画图）
    history.json         逐轮 train/val 指标 + stop_reason
    best_model.pt        验证 Macro-F1 最佳的 checkpoint（自包含：含词表与配置）
    last_full.pt         最新一轮完整状态（含优化器/调度器，可精确续训）
    metrics.json         汇总结果（best val、训练成本、测试集结果）
    training_curves.png  loss / accuracy / Macro-F1 曲线
    confusion_matrix.png 最佳轮次的验证集混淆矩阵
    model_card.md        报告 Model 部分可直接引用的模型卡
    test_predictions.csv 仅在显式开启测试集评估时生成
"""

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__:  # 作为包导入 / imported as a package
    from .config import REPO_ROOT, data_dir_of, load_config, run_dir_of
    from .data import Vocabulary, build_dataloaders, load_splits, stratified_subsample
    from .metrics import classification_metrics, evaluate
    from .model import RecurrentClassifier, count_parameters, load_checkpoint
    from .plotting import plot_confusion_matrix, plot_training_curves
else:            # 直接以脚本方式运行 / executed as a plain script
    from config import REPO_ROOT, data_dir_of, load_config, run_dir_of  # type: ignore
    from data import Vocabulary, build_dataloaders, load_splits, stratified_subsample  # type: ignore
    from metrics import classification_metrics, evaluate  # type: ignore
    from model import RecurrentClassifier, count_parameters, load_checkpoint  # type: ignore
    from plotting import plot_confusion_matrix, plot_training_curves  # type: ignore

#: 写进 checkpoint 的分词器标识，供推理时确认"训练与推理用的是同一套规则"
TOKENIZER_NAME = "lowercase_regex_word_or_punct"


# =============================================================================
# 环境与随机性 / environment and determinism
# =============================================================================
def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """固定 Python / NumPy / PyTorch 的随机源 / fix every random source.

    Args:
        seed: 随机种子。数据划分由成员一固定，这里的种子只影响初始化与 batch 顺序。
        deterministic: ``True`` 时强制 cuDNN 使用确定性算法（稍慢）。

    Note:
        即使全部打开，不同 GPU / cuDNN / 线程数仍可能带来 0.1~0.3 个百分点的差异，
        报告中应写"同一硬件上可复现"。
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def environment_snapshot() -> Dict[str, Any]:
    """记录运行环境，写进 ``config.json`` 供报告引用 / record the runtime environment."""
    has_cuda = torch.cuda.is_available()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if has_cuda else "cpu",
        "device": "cuda" if has_cuda else "cpu",
    }


def build_grad_scaler(enabled: bool) -> Optional[Any]:
    """构造 GradScaler，兼容不同 PyTorch 小版本 / GradScaler with a version fallback."""
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)      # torch >= 2.3 推荐写法
    except (TypeError, AttributeError):                        # 回退到旧 API
        return torch.cuda.amp.GradScaler(enabled=True)


_GIT_COMMIT_CACHE: Dict[str, Optional[str]] = {}


def git_commit_of(path: Path = REPO_ROOT) -> Optional[str]:
    """记录当前仓库 commit，说明"这份结果对应哪版代码"（结果缓存，避免每轮起子进程）。"""
    key = str(path)
    if key not in _GIT_COMMIT_CACHE:
        try:
            import subprocess

            result = subprocess.run(
                ["git", "-C", key, "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=15,
            )
            _GIT_COMMIT_CACHE[key] = result.stdout.strip() or None
        except Exception:
            _GIT_COMMIT_CACHE[key] = None
    return _GIT_COMMIT_CACHE[key]


# =============================================================================
# checkpoint / checkpoint I/O
# =============================================================================
def save_checkpoint(path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer],
                    scheduler: Optional[Any], vocab: Vocabulary, config: Dict[str, Any],
                    epoch: int, best_score: float, history: List[Dict[str, Any]],
                    include_optimizer: bool = True) -> Path:
    """保存 checkpoint / save a self-contained checkpoint.

    checkpoint 里同时保存模型权重、词表、模型配置与截断长度，因此推理只需要这一个文件。

    Args:
        path: 输出路径。
        model: 模型。
        optimizer: 优化器；``include_optimizer=False`` 时可为 ``None``。
        scheduler: 学习率调度器或 ``None``。
        vocab: 词表。
        config: 完整配置。
        epoch: 当前轮次。
        best_score: 目前最佳验证 Macro-F1。
        history: 逐轮记录。
        include_optimizer: 是否保存优化器/调度器状态（续训需要，会让文件大 3~4 倍）。

    Returns:
        写入的路径。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "vocab": vocab.token_to_id,
        "vocab_meta": vocab.meta,
        "vocab_size": len(vocab),
        "tokenizer": TOKENIZER_NAME,
        "max_len": int(config["max_len"]),
        "model": dict(config["model"]),
        "config": config,
        "epoch": int(epoch),
        "best_score": float(best_score),
        "history": history,
        "seed": int(config["seed"]),
        "torch_version": torch.__version__,
        "git_commit": git_commit_of(),
    }
    if include_optimizer:
        payload["optimizer_state"] = optimizer.state_dict() if optimizer is not None else None
        payload["scheduler_state"] = scheduler.state_dict() if scheduler is not None else None
    torch.save(payload, path)
    return path


# =============================================================================
# 单轮训练 / one epoch
# =============================================================================
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, device: torch.device,
                    grad_clip: float = 1.0, scaler: Optional[Any] = None,
                    use_amp: bool = False) -> Dict[str, Any]:
    """训练一个 epoch，返回训练集上的 loss 与完整分类指标。

    Train for one epoch; metrics are accumulated on the fly instead of re-running
    an evaluation pass over the full training set (which would roughly double the
    cost per epoch on 500k examples).

    返回的是 :func:`metrics.classification_metrics` 的完整结果（含 ``macro_f1``），
    这样训练曲线才能同时画 train / val 的 Macro-F1。

    Args:
        model: 模型。
        loader: 训练集 DataLoader。
        optimizer: 优化器。
        criterion: 损失函数。
        device: 设备。
        grad_clip: 梯度裁剪阈值；``0`` 表示不裁剪。
        scaler: ``GradScaler`` 或 ``None``。
        use_amp: 是否混合精度。

    Returns:
        含 ``loss`` / ``accuracy`` / ``macro_f1`` / ``per_class`` / ``confusion_matrix`` 的字典。
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    for tokens, lengths, labels, _sample_ids in loader:
        tokens = tokens.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(tokens, lengths)
            loss = criterion(logits, labels)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)     # 先反缩放再裁剪，否则裁的是放大后的梯度
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
        # 顺手收集预测，标量指标在 epoch 结束时统一算，比每 batch 调一次 sklearn 便宜得多
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.detach().argmax(dim=1).cpu().tolist())

    metrics = classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, total_samples)
    return metrics


# =============================================================================
# 训练主循环 / training loop
# =============================================================================
def fit(model: nn.Module, loaders: Dict[str, DataLoader], vocab: Vocabulary,
        config: Dict[str, Any], run_dir: Path, device: torch.device,
        resume_from: Optional[Path] = None) -> Dict[str, Any]:
    """训练并返回结果摘要（早停、逐轮日志、checkpoint、ETA、峰值显存）。

    Train the model with early stopping and write every artefact needed to
    reproduce the run.

    Args:
        model: 已 ``.to(device)`` 的模型。
        loaders: 至少包含 ``"train"`` 与 ``"val"``。
        vocab: 词表（写进 checkpoint）。
        config: 完整配置。
        run_dir: 输出目录。
        device: 训练设备。
        resume_from: 可选，``last_full.pt`` 路径，用于精确续训。

    Returns:
        含 ``history`` / ``best_epoch`` / ``best_val_macro_f1`` / ``epochs_completed`` /
        ``stop_reason`` / ``total_seconds`` / ``seconds_per_epoch`` /
        ``peak_gpu_memory_mb`` 的字典。
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    training = config["training"]
    use_amp = bool(config.get("amp")) and device.type == "cuda"
    scaler = build_grad_scaler(use_amp)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["lr"]), weight_decay=float(training["weight_decay"])
    )

    scheduler = None
    scheduler_cfg = training["scheduler"]
    if scheduler_cfg.get("enabled"):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",
            factor=float(scheduler_cfg["factor"]),
            patience=int(scheduler_cfg["patience"]),
        )

    history: List[Dict[str, Any]] = []
    best_score = -1.0
    best_epoch = 0
    start_epoch = 1
    stale_epochs = 0

    # ---- 续训 / resume ----------------------------------------------------
    if resume_from is not None and Path(resume_from).exists():
        payload = torch.load(Path(resume_from), map_location="cpu", weights_only=False)
        saved_model = {key: payload["model"][key] for key in payload["model"]}
        if saved_model != {key: config["model"][key] for key in saved_model}:
            raise ValueError(
                "续训的模型结构与当前配置不一致 / architecture mismatch:\n"
                f"  checkpoint: {saved_model}\n  config:     {config['model']}"
            )
        model.load_state_dict(payload["model_state"])
        if payload.get("optimizer_state"):
            optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None and payload.get("scheduler_state"):
            scheduler.load_state_dict(payload["scheduler_state"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_score = float(payload.get("best_score", -1.0))
        history = list(payload.get("history", []))
        # 从历史里取回真正的最佳轮次（不能简单取最后一轮）
        best_epoch = (max(history, key=lambda item: item["val"]["macro_f1"])["epoch"]
                      if history else 0)
        print(f"[resume] 从第 {start_epoch} 轮继续 / continue from epoch {start_epoch} "
              f"(best macro-F1 = {best_score:.4f})")

    early = training["early_stopping"]
    patience_limit = int(early["patience"]) if early.get("enabled") else None
    min_delta = float(early["min_delta"])

    log_jsonl = run_dir / "metrics.jsonl"
    log_csv = run_dir / "training_log.csv"
    if resume_from is None:
        # 新一轮训练：清空旧日志，避免和历史记录混在一起
        for path in (log_jsonl, log_csv):
            if path.exists():
                path.unlink()
        with log_csv.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(
                ["epoch", "lr", "train_loss", "train_accuracy", "val_loss",
                 "val_accuracy", "val_macro_f1", "epoch_seconds", "elapsed_seconds"]
            )

    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()      # 只统计本次训练的峰值显存

    stop_reason = "max_epochs"
    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        epoch_start = time.perf_counter()

        train_metrics = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device,
            grad_clip=float(training["grad_clip"]), scaler=scaler, use_amp=use_amp,
        )
        # 验证集固定 FP32 评估，避免 AMP 数值噪声影响选模 / evaluate in FP32
        val_metrics, _ = evaluate(model, loaders["val"], criterion, device=device, use_amp=False)

        epoch_seconds = time.perf_counter() - epoch_start
        elapsed = time.perf_counter() - started
        done_epochs = epoch - start_epoch + 1
        remaining = int(training["epochs"]) - epoch
        eta_minutes = (elapsed / max(1, done_epochs)) * remaining / 60.0

        record = {
            "epoch": epoch,
            # train 侧保留完整指标（含 macro_f1），这样曲线图能画 train/val 的 Macro-F1；
            # 逐类指标与混淆矩阵体积大，train 侧只留标量，val 侧保留完整结果。
            "train": {
                key: train_metrics[key]
                for key in ("loss", "accuracy", "macro_precision", "macro_recall", "macro_f1")
            },
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": round(epoch_seconds, 2),
            "elapsed_seconds": round(elapsed, 2),
            "eta_minutes": round(eta_minutes, 2),
        }
        history.append(record)

        # 逐轮落盘：训练被打断也不丢已有结果 / append-only logging
        with log_jsonl.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        with log_csv.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow([
                epoch, record["lr"], f"{train_metrics['loss']:.6f}",
                f"{train_metrics['accuracy']:.6f}", f"{val_metrics['loss']:.6f}",
                f"{val_metrics['accuracy']:.6f}", f"{val_metrics['macro_f1']:.6f}",
                f"{epoch_seconds:.2f}", f"{elapsed:.2f}",
            ])

        improved = val_metrics["macro_f1"] > best_score + min_delta
        if improved:
            best_score = float(val_metrics["macro_f1"])
            best_epoch = epoch
            stale_epochs = 0
            # 选模只看验证集 Macro-F1 / selection metric = validation macro-F1
            save_checkpoint(run_dir / "best_model.pt", model, optimizer, scheduler,
                            vocab, config, epoch, best_score, history,
                            include_optimizer=False)
        else:
            stale_epochs += 1

        if scheduler is not None:
            scheduler.step(val_metrics["macro_f1"])

        # 每轮存一份完整状态用于精确续训（含优化器，文件较大）
        save_checkpoint(run_dir / "last_full.pt", model, optimizer, scheduler,
                        vocab, config, epoch, best_score, history, include_optimizer=True)

        flag = "  *best" if improved else ""
        print(f"epoch {epoch:03d}/{int(training['epochs'])} | "
              f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} | "
              f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f} "
              f"f1 {val_metrics['macro_f1']:.4f} | {epoch_seconds:5.0f}s | "
              f"ETA {eta_minutes:4.0f} min{flag}", flush=True)

        if patience_limit is not None and stale_epochs >= patience_limit:
            stop_reason = f"early_stopping(patience={patience_limit})"
            print(f"验证 Macro-F1 连续 {patience_limit} 轮未提升，提前停止 / early stop.")
            break

    total_seconds = time.perf_counter() - started
    result: Dict[str, Any] = {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_score),
        "epochs_completed": len(history),
        "epochs_requested": int(training["epochs"]),
        "stop_reason": stop_reason,
        "total_seconds": round(total_seconds, 2),
        "seconds_per_epoch": round(total_seconds / max(1, len(history)), 2),
        "peak_gpu_memory_mb": (round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
                               if device.type == "cuda" else None),
        "amp": use_amp,
    }
    (run_dir / "history.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


# =============================================================================
# 模型卡 / model card for the report
# =============================================================================
def write_model_card(run_dir: Path, config: Dict[str, Any], result: Dict[str, Any],
                     val_metrics: Dict[str, Any], parameter_count: int,
                     data_sizes: Dict[str, int], vocab_size: int,
                     environment: Dict[str, Any]) -> Path:
    """生成报告 Model 部分可直接引用的 Markdown 模型卡。"""
    model_cfg = config["model"]
    training_cfg = config["training"]
    best_epoch = result["best_epoch"]

    best_val = val_metrics
    for record in result["history"]:
        if record["epoch"] == best_epoch:
            best_val = record["val"]
            break

    lines: List[str] = [
        f"# Model card — {config['run_name']}",
        "",
        f"- 架构 / architecture: Bidirectional **{model_cfg['rnn_type'].upper()}** "
        f"with randomly initialised embeddings",
        f"- Embedding: random U(-0.1, 0.1), trainable (`embedding_dim={model_cfg['embedding_dim']}`)",
        f"- 隐藏层 / hidden: {model_cfg['hidden_dim']} per direction × {model_cfg['num_layers']} layer(s), "
        f"bidirectional={model_cfg['bidirectional']}, pooling=`{model_cfg['pooling']}`",
        f"- Dropout: {model_cfg['dropout']} (recurrent output) / "
        f"{model_cfg['embedding_dropout']} (embedding)",
        f"- 参数量 / parameters: **{parameter_count:,}**",
        f"- 词表 / vocabulary: {vocab_size:,} ids (PAD=0, UNK=1), fitted on the training split only",
        f"- 截断长度 / max_len: {config['max_len']} tokens",
        f"- 数据 / data: train {data_sizes['train']:,} / val {data_sizes['val']:,} / "
        f"test {data_sizes['test']:,}（全量清洗划分，seed=42，由成员一固定）",
        f"- 优化 / optimisation: AdamW lr={training_cfg['lr']}, "
        f"weight_decay={training_cfg['weight_decay']}, batch={training_cfg['batch_size']}, "
        f"grad_clip={training_cfg['grad_clip']}",
        f"- 早停 / early stopping: enabled={training_cfg['early_stopping']['enabled']}, "
        f"patience={training_cfg['early_stopping']['patience']}, "
        f"min_delta={training_cfg['early_stopping']['min_delta']}（监控验证 Macro-F1）",
        f"- 完成轮数 / epochs: {result['epochs_completed']}"
        f"（预算 {result['epochs_requested']}，停止原因 `{result['stop_reason']}`）",
        f"- 最佳轮次 / best epoch: **{best_epoch}**",
        "",
        "## Validation performance (best epoch)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy | {best_val['accuracy']:.4f} |",
        f"| Macro Precision | {best_val['macro_precision']:.4f} |",
        f"| Macro Recall | {best_val['macro_recall']:.4f} |",
        f"| Macro-F1 | **{best_val['macro_f1']:.4f}** |",
        f"| Loss | {best_val['loss']:.4f} |",
    ]
    for class_label, class_metrics in best_val["per_class"].items():
        lines.append(
            f"| Class {class_label} — P / R / F1 | {class_metrics['precision']:.4f} / "
            f"{class_metrics['recall']:.4f} / {class_metrics['f1']:.4f} |"
        )

    matrix = best_val["confusion_matrix"]
    lines += [
        "",
        "## Confusion matrix (validation)",
        "",
        "| | Predicted negative | Predicted positive |",
        "|---|---:|---:|",
        f"| **True negative** | {matrix[0][0]:,} | {matrix[0][1]:,} |",
        f"| **True positive** | {matrix[1][0]:,} | {matrix[1][1]:,} |",
        "",
        "## Training cost",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| GPU | {environment['gpu']} |",
        f"| PyTorch / CUDA | {environment['torch']} / {environment['cuda']} |",
        f"| Epochs completed | {result['epochs_completed']} |",
        f"| Seconds per epoch | {result['seconds_per_epoch']:.1f} |",
        f"| Total wall time | {result['total_seconds'] / 60:.1f} min |",
        f"| Peak GPU memory (MB) | {result.get('peak_gpu_memory_mb')} |",
        f"| AMP | {result.get('amp')} |",
        "",
        "## Figures",
        "",
        "![training curves](training_curves.png)",
        "",
        "![validation confusion matrix](confusion_matrix.png)",
        "",
        "## Notes / limitations",
        "",
        "- 单种子结果，未做多种子方差分析。",
        "- `max_len=256` 会截断约 19% 的评论，长评论可能因此判错。",
        "- 随机初始化 embedding 不含外部语言知识；这正是成员三 GloVe/Word2Vec 组的比较目标。",
        "- 验证集只用于选模与早停；测试集未参与训练与选模。",
    ]

    path = Path(run_dir) / "model_card.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# =============================================================================
# 一次完整实验 / one complete experiment
# =============================================================================
def run_experiment(config: Dict[str, Any],
                   frames: Optional[Dict[str, pd.DataFrame]] = None,
                   evaluate_test: Optional[bool] = None,
                   reuse_vocab: Optional[Path] = None,
                   resume_from: Optional[Path] = None,
                   do_train: bool = True) -> Dict[str, Any]:
    """跑完一次完整实验：建词表 → 训练 → 复核 → 出图 → 模型卡 → 汇总。

    Run one complete experiment end to end.

    Args:
        config: 完整配置（建议来自 :func:`config.load_config`）。
        frames: 已校验的划分字典；``None`` 时自动调用 :func:`data.load_splits`。
        evaluate_test: ``None`` 时读取 ``config["eval_test_after_selection"]``。
        reuse_vocab: 复用已有 ``vocab.json``（成员三做 GloVe 对比时应传同一份）。
        resume_from: 断点续训的 ``last_full.pt``。
        do_train: ``False`` 时只构建词表与模型，不训练（结构自检用）。

    Returns:
        ``{"run_dir", "model", "vocab", "report"}``（``do_train=False`` 时无 ``report``）。
    """
    config = deepcopy(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if frames is None:
        frames, _audit = load_splits(data_dir_of(config))
    if evaluate_test is None:
        evaluate_test = bool(config.get("eval_test_after_selection", False))

    set_global_seed(int(config["seed"]))
    run_dir = run_dir_of(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = environment_snapshot()

    print("=" * 78)
    print(f"run: {config['run_name']}")
    print("=" * 78)

    # ---- 1. 词表：只由训练集拟合（或复用共享词表） -------------------------
    # 注意：如果用了 --limit-train 做小样本实验，词表也必须只在**同一个子集**上拟合，
    # 否则词表会包含这个子集根本没见过的词，实验口径就不一致了。
    train_frame_for_vocab = stratified_subsample(
        frames["train"], config.get("limit_train"), int(config["seed"])
    )
    if reuse_vocab is not None and Path(reuse_vocab).exists():
        vocab = Vocabulary.load(Path(reuse_vocab))
        print(f"复用词表 / reused vocabulary: {reuse_vocab}（{len(vocab):,} ids）")
    else:
        vocab = Vocabulary.build(
            train_frame_for_vocab["text"].tolist(),
            max_size=int(config["vocab_max_size"]),
            min_freq=int(config["vocab_min_freq"]),
        )
    vocab.save(run_dir / "vocab.json")

    # ---- 2. DataLoader -----------------------------------------------------
    splits = ("train", "val", "test") if evaluate_test else ("train", "val")
    loaders = build_dataloaders(frames, vocab, config, device=device, splits=splits)

    # ---- 3. 模型 -----------------------------------------------------------
    model = RecurrentClassifier(vocab_size=len(vocab), **config["model"]).to(device)
    parameter_count = count_parameters(model)

    snapshot = {
        "run_name": config["run_name"],
        "seed": config["seed"],
        "config": config,
        "environment": environment,
        "data": {
            "train": int(len(frames["train"])),
            "val": int(len(frames["val"])),
            "test": int(len(frames["test"])),
            # 实际用于训练/验证的样本数（--limit-* 生效后的值）
            "train_used": int(len(train_frame_for_vocab)),
            "val_used": int(len(stratified_subsample(frames["val"], config.get("limit_val"),
                                                     int(config["seed"])))),
        },
        "vocab": {"size": len(vocab), **vocab.meta},
        "val_oov_rate": round(vocab.oov_rate(frames["val"]["text"].tolist()), 6),
        "test_oov_rate": round(vocab.oov_rate(frames["test"]["text"].tolist()), 6),
        "parameters": parameter_count,
        "git_commit": git_commit_of(),
    }
    (run_dir / "config.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"vocab={len(vocab):,}  params={parameter_count:,}  "
          f"train={snapshot['data']['train_used']:,}  val={snapshot['data']['val_used']:,}  "
          f"oov(val)={snapshot['val_oov_rate']:.4%}")

    if not do_train:
        print("do_train=False：只构建，不训练 / built but not trained.")
        return {"run_dir": str(run_dir), "model": model, "vocab": vocab,
                "snapshot": snapshot, "loaders": loaders}

    # ---- 4. 训练 -----------------------------------------------------------
    result = fit(model, loaders, vocab, config, run_dir, device, resume_from=resume_from)

    # ---- 5. 用最佳 checkpoint 复核验证集（避免"内存里是最后一轮"的常见错误） --
    best_model, _, _payload = load_checkpoint(run_dir / "best_model.pt", device=device)
    criterion = nn.CrossEntropyLoss()
    val_metrics, _ = evaluate(best_model, loaders["val"], criterion, device=device, use_amp=False)
    print(f"\n用最佳 checkpoint（epoch {result['best_epoch']}）复核验证集：")
    print(f"  accuracy={val_metrics['accuracy']:.4f}  macro-F1={val_metrics['macro_f1']:.4f}  "
          f"loss={val_metrics['loss']:.4f}")

    # ---- 6. 出图 -----------------------------------------------------------
    plot_training_curves(
        result["history"], result["best_epoch"], run_dir / "training_curves.png",
        title=(f"{config['model']['rnn_type'].upper()} random embedding | "
               f"train {snapshot['data']['train_used']:,} / "
               f"val {snapshot['data']['val_used']:,} | seed {config['seed']}"),
    )
    plot_confusion_matrix(
        val_metrics["confusion_matrix"], run_dir / "confusion_matrix.png",
        title=(f"Validation confusion matrix\n"
               f"{config['model']['rnn_type'].upper()} random embedding, "
               f"epoch {result['best_epoch']}"),
    )

    # ---- 7. 模型卡 ---------------------------------------------------------
    # 用**实际训练用到的**样本数，而不是划分总条数：小样本实验时两者不同
    data_sizes = {
        "train": snapshot["data"]["train_used"],
        "val": snapshot["data"]["val_used"],
        "test": snapshot["data"]["test"],
    }
    write_model_card(run_dir, config, result, val_metrics, parameter_count,
                     data_sizes, len(vocab), environment)

    # ---- 8. 测试集：只在显式开启时评估一次 ----------------------------------
    test_metrics: Optional[Dict[str, Any]] = None
    if evaluate_test:
        print("\n" + "!" * 78)
        print("警告：正在评估官方测试集。测试集只能在配置锁定后评估一次，")
        print("      结果不得用于调参或挑选 checkpoint（题目 MR-1 要求）。")
        print("!" * 78)
        test_metrics, test_rows = evaluate(
            best_model, loaders["test"], criterion, device=device,
            use_amp=False, collect_predictions=True,
        )
        pd.DataFrame(test_rows).to_csv(run_dir / "test_predictions.csv",
                                       index=False, encoding="utf-8")
        print(f"测试集 accuracy={test_metrics['accuracy']:.4f}  "
              f"macro-F1={test_metrics['macro_f1']:.4f}")

    # ---- 9. 汇总 -----------------------------------------------------------
    report = {
        "run_name": config["run_name"],
        "model": config["model"],
        "training": config["training"],
        "max_len": config["max_len"],
        "vocab_size": len(vocab),
        "parameters": parameter_count,
        "best_epoch": result["best_epoch"],
        "epochs_completed": result["epochs_completed"],
        "stop_reason": result["stop_reason"],
        "seconds_per_epoch": result["seconds_per_epoch"],
        "total_seconds": result["total_seconds"],
        "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
        "val_best": val_metrics,
        "history": result["history"],
        "test": test_metrics,
        "environment": environment,
        "data": data_sizes,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n结果已写入 / artefacts written to: {run_dir}")
    print(f"  metrics.json  best val macro-F1 = {val_metrics['macro_f1']:.4f}")

    return {"run_dir": str(run_dir), "model": best_model, "vocab": vocab, "report": report}


# =============================================================================
# 冒烟测试 / smoke test
# =============================================================================
def run_smoke_test(config: Dict[str, Any], frames: Dict[str, pd.DataFrame],
                   train_size: int = 1024, val_size: int = 512, steps: int = 40,
                   batch_size: int = 32, learning_rate: float = 5e-3) -> Dict[str, Any]:
    """用极小样本验证整条链路，约 1~2 分钟。

    Quick end-to-end sanity check. 检查三件事：

    1. Dataset → collate → 模型 前向能跑通，形状正确；
    2. 反向传播能让 loss 下降（至少不爆炸）；
    3. checkpoint 能存、能读，读回来的模型输出与保存前一致。

    Args:
        config: 完整配置（只用 ``seed`` / ``model`` / ``amp``）。
        frames: 已加载的划分。
        train_size / val_size: 抽样规模。
        steps: 优化步数。
        batch_size: 批大小。
        learning_rate: 冒烟测试用的较大学习率，确保 40 步内能看出 loss 下降。

    Returns:
        ``{"first_loss", "last_loss", "improved", "checkpoint_ok"}``
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("冒烟测试 / smoke test")
    print("=" * 78)
    set_global_seed(int(config["seed"]))

    small_train = stratified_subsample(frames["train"], train_size, int(config["seed"]))
    small_val = stratified_subsample(frames["val"], val_size, int(config["seed"]))

    smoke_config = deepcopy(config)
    smoke_config["run_name"] = f"{config['run_name']}_smoke"
    smoke_config["max_len"] = min(int(config["max_len"]), 128)   # 冒烟测试不需要长序列
    smoke_config["training"] = deepcopy(config["training"])
    smoke_config["training"]["batch_size"] = batch_size
    smoke_config["dataloader"] = {"num_workers": 0, "pin_memory": False}

    smoke_vocab = Vocabulary.build(small_train["text"].tolist(), max_size=8000, min_freq=1)
    loaders = build_dataloaders({"train": small_train, "val": small_val},
                                smoke_vocab, smoke_config, device=device,
                                splits=("train", "val"))
    model = RecurrentClassifier(vocab_size=len(smoke_vocab), **config["model"]).to(device)
    print(f"vocab={len(smoke_vocab):,}  params={count_parameters(model):,}  "
          f"train={len(small_train):,}  val={len(small_val):,}  device={device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    use_amp = bool(config.get("amp")) and device.type == "cuda"
    scaler = build_grad_scaler(use_amp)

    losses: List[float] = []
    last_tokens: Optional[torch.Tensor] = None
    last_lengths: Optional[torch.Tensor] = None
    model.train()
    while len(losses) < steps:
        for tokens, lengths, labels, _sample_ids in loaders["train"]:
            tokens = tokens.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(tokens, lengths)
                loss = criterion(logits, labels)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(float(loss.detach().item()))
            last_tokens, last_lengths = tokens[:8].cpu(), lengths[:8]
            if len(losses) >= steps:
                break

    first_mean = float(np.mean(losses[:5]))
    last_mean = float(np.mean(losses[-5:]))
    improved = last_mean < first_mean
    print(f"loss 前 5 步均值 = {first_mean:.4f} → 后 5 步均值 = {last_mean:.4f} "
          f"({'PASS 下降' if improved else 'CHECK 未下降，请检查 lr / 数据'})")

    # checkpoint 往返一致性 / checkpoint round-trip
    smoke_dir = run_dir_of(smoke_config)
    smoke_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_checkpoint(
        smoke_dir / "smoke.pt", model, optimizer, None, smoke_vocab, smoke_config,
        epoch=1, best_score=0.0, history=[],
    )
    restored, _, payload = load_checkpoint(checkpoint_path, device=device)
    model.eval()
    with torch.no_grad():
        before = model(last_tokens.to(device), last_lengths)
        after = restored(last_tokens.to(device), last_lengths)
    checkpoint_ok = bool(torch.allclose(before, after, atol=1e-6))
    print(f"checkpoint 往返一致 = {checkpoint_ok}  "
          f"(max |Δ| = {float((before - after).abs().max()):.2e})  "
          f"vocab={payload['vocab_size']:,}  max_len={payload['max_len']}")
    if not checkpoint_ok:
        raise AssertionError("checkpoint 保存/加载后模型输出不一致")

    print(f"冒烟测试完成 / smoke test finished: improved={improved}, checkpoint_ok={checkpoint_ok}")
    return {"first_loss": first_mean, "last_loss": last_mean,
            "improved": improved, "checkpoint_ok": checkpoint_ok}


# =============================================================================
# 命令行入口 / CLI
# =============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器 / build the argparse parser."""
    parser = argparse.ArgumentParser(
        prog="python -m models.lstm_gru_random.train",
        description="训练随机初始化 Embedding 的 BiLSTM / BiGRU 情感分类器"
                    "（Yelp Review Polarity）。Train the random-embedding recurrent baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例 / examples:\n"
            "  # 1) 先跑 1 分钟的链路自检\n"
            "  python -m models.lstm_gru_random.train --smoke-test\n\n"
            "  # 2) 正式训练 BiLSTM（全量 503,979 / 55,998）\n"
            "  python -m models.lstm_gru_random.train\n\n"
            "  # 3) 正式训练 BiGRU 对照（其余超参完全一致）\n"
            "  python -m models.lstm_gru_random.train --rnn-type gru \\\n"
            "      --run-name member2_bigru_random_full_seed42\n\n"
            "  # 4) 想先在小样本上试跑（约 20k 训练样本）\n"
            "  python -m models.lstm_gru_random.train --limit-train 20000 \\\n"
            "      --limit-val 4000 --run-name debug_20k\n\n"
            "  # 5) 断点续训\n"
            "  python -m models.lstm_gru_random.train \\\n"
            "      --resume runs/member2_bilstm_random_full_seed42/last_full.pt\n\n"
            "  # 6) 配置锁定后的最终测试集评估（只做一次）\n"
            "  python -m models.lstm_gru_random.train --evaluate-test\n"
        ),
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML 配置路径（默认使用包内 config.yaml）")

    group = parser.add_argument_group("实验标识 / run identity")
    group.add_argument("--run-name", type=str, default=None, help="输出目录名，如 member2_bigru_random_full_seed42")
    group.add_argument("--data", type=Path, default=None, help="数据目录（默认 processed_data）")
    group.add_argument("--runs-dir", type=Path, default=None, help="输出根目录（默认 runs）")
    group.add_argument("--seed", type=int, default=None, help="随机种子（默认 42）")

    group = parser.add_argument_group("文本表示 / text representation")
    group.add_argument("--max-len", type=int, default=None, help="截断长度（默认 256）")
    group.add_argument("--vocab-size", type=int, default=None, help="词表大小（默认 30000）")

    group = parser.add_argument_group("模型 / model")
    group.add_argument("--rnn-type", choices=("lstm", "gru"), default=None, help="循环单元类型")
    group.add_argument("--embedding-dim", type=int, default=None, help="词向量维度（默认 100）")
    group.add_argument("--hidden-dim", type=int, default=None, help="每方向隐藏单元（默认 128）")
    group.add_argument("--num-layers", type=int, default=None, help="循环层数（默认 1）")
    group.add_argument("--dropout", type=float, default=None, help="dropout（默认 0.3）")
    group.add_argument("--embedding-dropout", type=float, default=None, help="embedding dropout（默认 0.1）")
    group.add_argument("--pooling", choices=("last", "max"), default=None, help="池化方式")

    group = parser.add_argument_group("训练 / optimisation")
    group.add_argument("--epochs", type=int, default=None, help="训练轮数上限（默认 20）")
    group.add_argument("--batch-size", type=int, default=None, help="批大小（默认 128）")
    group.add_argument("--lr", type=float, default=None, help="学习率（默认 1e-3）")
    group.add_argument("--weight-decay", type=float, default=None, help="AdamW 权重衰减（默认 0.01）")
    group.add_argument("--patience", type=int, default=None, help="早停 patience（默认 4）")
    group.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                       help="是否使用混合精度（--no-amp 关闭）")

    group = parser.add_argument_group("运行控制 / control")
    group.add_argument("--limit-train", type=int, default=None, help="调试：只用这么多训练样本")
    group.add_argument("--limit-val", type=int, default=None, help="调试：只用这么多验证样本")
    group.add_argument("--smoke-test", action="store_true", help="只跑 1 分钟的链路自检，不做正式训练")
    group.add_argument("--resume", type=Path, default=None, help="从 last_full.pt 续训")
    group.add_argument("--evaluate-test", action="store_true",
                       help="训练后在官方测试集上评估一次（配置锁定后才允许）")
    group.add_argument("--skip-checksum", action="store_true",
                       help="跳过 SHA-256 / 泄漏校验（不建议，仅用于数据未变更的快速重跑）")
    return parser


def _collect(args: argparse.Namespace, mapping: Dict[str, str]) -> Dict[str, Any]:
    """把命令行参数中"非 None"的项收集成配置覆盖字典 / collect non-None CLI overrides."""
    return {config_key: getattr(args, arg_name)
            for arg_name, config_key in mapping.items()
            if getattr(args, arg_name, None) is not None}


def build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """把命令行参数翻译成嵌套的配置覆盖字典 / map CLI args onto config keys."""
    overrides: Dict[str, Any] = _collect(args, {
        "run_name": "run_name",
        "data": "data_dir",
        "runs_dir": "runs_dir",
        "seed": "seed",
        "max_len": "max_len",
        "vocab_size": "vocab_max_size",
        "limit_train": "limit_train",
        "limit_val": "limit_val",
        "amp": "amp",
    })
    # argparse 用 Path 接收路径，但配置要写进 config.json，必须先转成字符串
    for key in ("data_dir", "runs_dir"):
        if overrides.get(key) is not None:
            overrides[key] = str(overrides[key])

    model_overrides = _collect(args, {
        "rnn_type": "rnn_type",
        "embedding_dim": "embedding_dim",
        "hidden_dim": "hidden_dim",
        "num_layers": "num_layers",
        "dropout": "dropout",
        "embedding_dropout": "embedding_dropout",
        "pooling": "pooling",
    })
    if model_overrides:
        overrides["model"] = model_overrides

    training_overrides = _collect(args, {
        "epochs": "epochs",
        "batch_size": "batch_size",
        "lr": "lr",
        "weight_decay": "weight_decay",
    })
    if training_overrides:
        overrides["training"] = training_overrides
    if args.patience is not None:
        overrides.setdefault("training", {}).setdefault("early_stopping", {})["patience"] = args.patience

    if args.evaluate_test:
        overrides["eval_test_after_selection"] = True
    return overrides


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口 / CLI entry point.

    Returns:
        进程退出码（0 = 成功）。
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config, overrides=build_overrides(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device} | run = {config['run_name']} | "
          f"rnn = {config['model']['rnn_type'].upper()}")

    frames, audit = load_splits(data_dir_of(config), verify=not args.skip_checksum)
    print(f"数据 / data: train={len(frames['train']):,}  val={len(frames['val']):,}  "
          f"test={len(frames['test']):,}")
    if not args.skip_checksum:
        checksum_ok = all(info.get("match", True)
                          for info in audit["checksums"].values()
                          if isinstance(info, dict))
        print(f"校验和 / checksums: {'OK' if checksum_ok else 'MISMATCH'} | "
              f"重复={sum(audit['leakage']['within_split_duplicates'].values())} | "
              f"泄漏={sum(audit['leakage']['cross_split_overlap'].values())}")

    if args.smoke_test:
        result = run_smoke_test(config, frames)
        return 0 if result["checkpoint_ok"] else 1

    output = run_experiment(
        config, frames=frames,
        evaluate_test=bool(config["eval_test_after_selection"]),
        resume_from=args.resume,
    )
    report = output["report"]
    print("\n" + "=" * 78)
    print("完成 / done")
    print("=" * 78)
    print(f"  run dir      : {output['run_dir']}")
    print(f"  best epoch   : {report['best_epoch']}")
    print(f"  val accuracy : {report['val_best']['accuracy']:.4f}")
    print(f"  val macro-F1 : {report['val_best']['macro_f1']:.4f}")
    print(f"  total time   : {report['total_seconds'] / 60:.1f} min")
    print(f"  next step    : python -m models.lstm_gru_random.predict "
          f"--checkpoint {Path(output['run_dir']) / 'best_model.pt'} --text \"not bad at all\"")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
