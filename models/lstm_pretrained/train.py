# -*- coding: utf-8 -*-
"""成员三：预训练词向量（GloVe / Word2Vec）训练入口。

一条命令跑完一个实验，例如：

    python -m models.lstm_pretrained.train --embedding-source random
    python -m models.lstm_pretrained.train --embedding-source glove --embedding-file glove.6B.100d.txt
    python -m models.lstm_pretrained.train --embedding-source glove --embedding-file glove.6B.100d.txt --freeze-embedding

只复用成员二的训练循环（fit / evaluate / 绘图 / 词表），本文件只负责
"把预训练词向量按词表对齐并注入模型"这一件新增的事情。
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
from torch import nn

from models.lstm_gru_random import (
    RecurrentClassifier,
    Vocabulary,
    build_dataloaders,
    count_parameters,
    data_dir_of,
    evaluate,
    fit,
    load_checkpoint,
    load_config,
    load_splits,
    run_dir_of,
    set_global_seed,
    stratified_subsample,
)
from models.lstm_gru_random.plotting import plot_confusion_matrix, plot_training_curves
from models.lstm_gru_random.train import environment_snapshot, git_commit_of
from models.lstm_pretrained import (
    build_embedding_matrix,
    coverage_report,
    load_glove,
    load_word2vec_text,
)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_embedding_vectors(source: str, path: Path, dim: int) -> Dict[str, Any]:
    """按来源读取词向量文件，返回 {word: vector}。"""
    if source == "glove":
        return load_glove(path, dim=dim)
    if source == "word2vec":
        return load_word2vec_text(path)
    raise ValueError(f"未知的词向量来源 / unknown embedding source: {source!r}")


def write_pretrained_model_card(
    run_dir: Path,
    config: Dict[str, Any],
    result: Dict[str, Any],
    val_metrics: Dict[str, Any],
    parameter_count: int,
    data_sizes: Dict[str, int],
    vocab_size: int,
    environment: Dict[str, Any],
    embedding_source: str,
    coverage: Optional[float],
    freeze: bool,
) -> Path:
    """写成员三自己的模型卡，词向量相关字段与成员二不同。"""
    model_cfg = config["model"]
    training_cfg = config["training"]
    lines = [
        f"# Model card — {config['run_name']}",
        "",
        f"- 架构 / architecture: Bidirectional **{model_cfg['rnn_type'].upper()}**",
        f"- Embedding source: **{embedding_source}**" + ("（frozen）" if freeze else "（fine-tuned）"),
    ]
    if coverage is not None:
        lines.append(f"- 预训练词表覆盖率 / coverage: **{coverage:.4f}**")
    lines += [
        f"- Embedding dim: {model_cfg['embedding_dim']}, hidden: {model_cfg['hidden_dim']} per direction",
        f"- Dropout: {model_cfg['dropout']} / embedding {model_cfg['embedding_dropout']}",
        f"- 参数量 / parameters: **{parameter_count:,}**",
        f"- 词表 / vocabulary: {vocab_size:,} ids (PAD=0, UNK=1)",
        f"- 截断长度 / max_len: {config['max_len']}",
        f"- 数据 / data: train {data_sizes['train']:,} / val {data_sizes['val']:,} / test {data_sizes['test']:,}",
        f"- 优化 / optimisation: AdamW lr={training_cfg['lr']}, batch={training_cfg['batch_size']}",
        f"- 最佳轮次 / best epoch: **{result['best_epoch']}**",
        "",
        "## Validation performance (best epoch)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy | {val_metrics['accuracy']:.4f} |",
        f"| Macro Precision | {val_metrics['macro_precision']:.4f} |",
        f"| Macro Recall | {val_metrics['macro_recall']:.4f} |",
        f"| Macro-F1 | **{val_metrics['macro_f1']:.4f}** |",
        "",
        "## Figures",
        "",
        "![training curves](training_curves.png)",
        "![validation confusion matrix](confusion_matrix.png)",
    ]
    path = Path(run_dir) / "model_card.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_pretrained_experiment(
    config: Dict[str, Any],
    frames: Optional[Dict[str, pd.DataFrame]] = None,
    vocab: Optional[Vocabulary] = None,
    embedding_source: str = "random",
    embedding_file: Optional[Path] = None,
    freeze: bool = False,
    evaluate_test: Optional[bool] = None,
    resume_from: Optional[Path] = None,
    do_train: bool = True,
) -> Dict[str, Any]:
    """跑完一次预训练词向量实验：词表 → 注入词向量 → 训练 → 出图 → 模型卡。"""
    config = deepcopy(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if frames is None:
        frames, _ = load_splits(data_dir_of(config), verify=True)
    if evaluate_test is None:
        evaluate_test = bool(config.get("eval_test_after_selection", False))

    set_global_seed(int(config["seed"]))
    run_dir = run_dir_of(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = environment_snapshot()

    # 词表：默认只在"实际用于训练的子集"上拟合；正式对比时用 --reuse-vocab 传共享词表
    if vocab is None:
        train_for_vocab = stratified_subsample(
            frames["train"], config.get("limit_train"), int(config["seed"])
        )
        vocab = Vocabulary.build(
            train_for_vocab["text"].tolist(),
            max_size=int(config["vocab_max_size"]),
            min_freq=int(config["vocab_min_freq"]),
        )
    vocab.save(run_dir / "vocab.json")

    splits = ("train", "val", "test") if evaluate_test else ("train", "val")
    loaders = build_dataloaders(frames, vocab, config, device=device, splits=splits)

    # 词向量矩阵：random 模式不加载，其余按词表对齐
    embedding_matrix = None
    coverage = None
    if embedding_source != "random":
        if embedding_file is None:
            raise ValueError("预训练模式需要 --embedding-file，例如 glove.6B.100d.txt")
        vectors = load_embedding_vectors(
            embedding_source, Path(embedding_file), int(config["model"]["embedding_dim"])
        )
        embedding_matrix, _matched = build_embedding_matrix(
            vocab, vectors, int(config["model"]["embedding_dim"]), seed=int(config["seed"])
        )
        coverage = coverage_report(vocab, vectors)

    model = RecurrentClassifier(
        vocab_size=len(vocab),
        embedding_weights=embedding_matrix,
        **config["model"],
    )
    if freeze:
        model.freeze_embeddings()
    model.to(device)
    parameter_count = count_parameters(model)

    train_used = int(len(stratified_subsample(
        frames["train"], config.get("limit_train"), int(config["seed"])
    )))
    val_used = int(len(stratified_subsample(
        frames["val"], config.get("limit_val"), int(config["seed"])
    )))

    snapshot = {
        "run_name": config["run_name"],
        "seed": config["seed"],
        "config": config,
        "environment": environment,
        "data": {
            "train": int(len(frames["train"])),
            "val": int(len(frames["val"])),
            "test": int(len(frames["test"])),
            "train_used": train_used,
            "val_used": val_used,
        },
        "vocab": {"size": len(vocab), **vocab.meta},
        "embedding": {
            "source": embedding_source,
            "file": str(embedding_file) if embedding_file else None,
            "freeze": bool(freeze),
            "coverage": coverage,
        },
        "parameters": parameter_count,
        "git_commit": git_commit_of(),
    }
    (run_dir / "config.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"run={config['run_name']}  source={embedding_source}  "
          f"freeze={freeze}  vocab={len(vocab):,}  params={parameter_count:,}  "
          f"train={train_used:,}  val={val_used:,}")

    if not do_train:
        return {"run_dir": str(run_dir), "model": model, "vocab": vocab,
                "snapshot": snapshot, "loaders": loaders}

    result = fit(model, loaders, vocab, config, run_dir, device, resume_from=resume_from)

    best_model, _, _ = load_checkpoint(run_dir / "best_model.pt", device=device)
    criterion = nn.CrossEntropyLoss()
    val_metrics, _ = evaluate(best_model, loaders["val"], criterion, device=device, use_amp=False)

    title_prefix = f"{config['model']['rnn_type'].upper()} {embedding_source}"
    if freeze:
        title_prefix += " frozen"
    plot_training_curves(
        result["history"], result["best_epoch"], run_dir / "training_curves.png",
        title=f"{title_prefix} | train {train_used:,} / val {val_used:,} | seed {config['seed']}",
    )
    plot_confusion_matrix(
        val_metrics["confusion_matrix"], run_dir / "confusion_matrix.png",
        title=f"Validation confusion matrix\n{title_prefix}, epoch {result['best_epoch']}",
    )

    data_sizes = {"train": train_used, "val": val_used, "test": int(len(frames["test"]))}
    write_pretrained_model_card(
        run_dir, config, result, val_metrics, parameter_count, data_sizes,
        len(vocab), environment, embedding_source, coverage, freeze,
    )

    test_metrics = None
    if evaluate_test:
        print("注意：正在评估官方测试集（仅允许在配置锁定后执行一次）。")
        test_metrics, test_rows = evaluate(
            best_model, loaders["test"], criterion, device=device,
            use_amp=False, collect_predictions=True,
        )
        pd.DataFrame(test_rows).to_csv(run_dir / "test_predictions.csv",
                                       index=False, encoding="utf-8")

    report = {
        "run_name": config["run_name"],
        "model": config["model"],
        "training": config["training"],
        "max_len": config["max_len"],
        "vocab_size": len(vocab),
        "parameters": parameter_count,
        "embedding": snapshot["embedding"],
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
    print(f"完成 / done: {run_dir}")
    print(f"  best val accuracy={val_metrics['accuracy']:.4f}  macro-F1={val_metrics['macro_f1']:.4f}")
    return {"run_dir": str(run_dir), "model": best_model, "vocab": vocab, "report": report}


def build_parser() -> argparse.ArgumentParser:
    """命令行参数：全部可选，默认来自 config.yaml。"""
    parser = argparse.ArgumentParser(prog="python -m models.lstm_pretrained.train")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--rnn-type", choices=("lstm", "gru"), default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--embedding-dropout", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--embedding-source", choices=("random", "glove", "word2vec"), default=None)
    parser.add_argument("--embedding-file", type=Path, default=None)
    parser.add_argument("--freeze-embedding", action="store_true")
    parser.add_argument("--reuse-vocab", type=Path, default=None)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--skip-checksum", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    overrides: Dict[str, Any] = {}
    if args.run_name is not None:
        overrides["run_name"] = args.run_name
    if args.data is not None:
        overrides["data_dir"] = str(args.data)
    if args.runs_dir is not None:
        overrides["runs_dir"] = str(args.runs_dir)
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.max_len is not None:
        overrides["max_len"] = args.max_len
    if args.vocab_size is not None:
        overrides["vocab_max_size"] = args.vocab_size
    if args.limit_train is not None:
        overrides["limit_train"] = args.limit_train
    if args.limit_val is not None:
        overrides["limit_val"] = args.limit_val
    if args.evaluate_test:
        overrides["eval_test_after_selection"] = True

    model_overrides: Dict[str, Any] = {}
    for arg_name, key in (
        ("rnn_type", "rnn_type"),
        ("embedding_dim", "embedding_dim"),
        ("hidden_dim", "hidden_dim"),
        ("dropout", "dropout"),
        ("embedding_dropout", "embedding_dropout"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            model_overrides[key] = value
    if model_overrides:
        overrides["model"] = model_overrides

    training_overrides: Dict[str, Any] = {}
    for arg_name, key in (("epochs", "epochs"), ("batch_size", "batch_size"), ("lr", "lr"), ("weight_decay", "weight_decay")):
        value = getattr(args, arg_name)
        if value is not None:
            training_overrides[key] = value
    if args.patience is not None:
        training_overrides.setdefault("early_stopping", {})["patience"] = args.patience
    if training_overrides:
        overrides["training"] = training_overrides

    config = load_config(args.config or DEFAULT_CONFIG_PATH, overrides=overrides)

    embedding_source = args.embedding_source or config.get("embedding", {}).get("source", "random")
    embedding_file = args.embedding_file or config.get("embedding", {}).get("file")
    freeze = args.freeze_embedding or bool(config.get("embedding", {}).get("freeze", False))

    vocab = None
    if args.reuse_vocab is not None:
        vocab = Vocabulary.load(Path(args.reuse_vocab))

    frames, _ = load_splits(data_dir_of(config), verify=not args.skip_checksum)
    output = run_pretrained_experiment(
        config,
        frames=frames,
        vocab=vocab,
        embedding_source=embedding_source,
        embedding_file=Path(embedding_file) if embedding_file else None,
        freeze=freeze,
        evaluate_test=bool(config.get("eval_test_after_selection", False)),
    )
    report = output["report"]
    print("")
    print("summary:", report["run_name"], "best_epoch", report["best_epoch"],
          "val_macro_f1", round(report["val_best"]["macro_f1"], 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())