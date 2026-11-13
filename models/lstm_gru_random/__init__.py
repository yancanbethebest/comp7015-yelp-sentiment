# -*- coding: utf-8 -*-
"""成员二 · 随机初始化 Embedding 的 LSTM/GRU 基础模型（可复用代码包）。

Member 2 deliverable: a small, importable package for the random-embedding
recurrent baseline on Yelp Review Polarity.

对外主要接口 / public API
------------------------
    from models.lstm_gru_random import (
        load_config,          # 读取 YAML 配置并用命令行覆盖
        load_splits,          # 读取并校验成员一交付的 processed_data/
        Vocabulary,           # 训练集词表（PAD=0, UNK=1）
        build_dataloaders,    # Dataset + 动态 padding + DataLoader
        RecurrentClassifier,  # BiLSTM / BiGRU，随机 Embedding
        classification_metrics,  # Accuracy / P / R / F1 / 混淆矩阵
        evaluate,             # 在 DataLoader 上评估
        run_experiment,       # 一条命令跑完：建词表 → 训练 → 出图 → 写结果
        load_checkpoint,      # 从 best_model.pt 还原模型与词表
        predict_texts,        # 任意文本推理
    )

给成员三的接口 / interface reserved for member 3
--------------------------------------------------
    RecurrentClassifier(..., embedding_weights=tensor)   # 传入 GloVe/Word2Vec 矩阵即可
    Vocabulary.load("runs/<run>/vocab.json")             # 复用同一份词表，保证对比公平

给成员四/五的接口 / interface for members 4 and 5
--------------------------------------------------
    metrics.json / history.json / training_log.csv / best_model.pt / test_predictions.csv
"""

if __package__:  # 作为包导入 / imported as a package
    from .config import (
        DEFAULT_CONFIG_PATH,
        PACKAGE_DIR,
        REPO_ROOT,
        data_dir_of,
        deep_update,
        load_config,
        run_dir_of,
        set_by_path,
        validate_config,
    )
    from .data import (
        PAD_ID,
        PAD_TOKEN,
        UNK_ID,
        UNK_TOKEN,
        SentimentDataset,
        Vocabulary,
        build_dataloaders,
        collate_batch,
        load_splits,
        resolve_num_workers,
        stratified_subsample,
        tokenize,
    )
    from .metrics import classification_metrics, evaluate
    from .model import RecurrentClassifier, count_parameters, load_checkpoint, predict_texts
    # 注意：这里**不**导入 .train。train.py 是命令行入口，用 ``python -m ...train`` 执行时
    # runpy 会先导入本包；如果本包已经把 .train 载入 sys.modules，就会触发
    # "found in sys.modules after import of package ... prior to execution" 的警告。
    # 因此训练相关函数改为按需懒加载（见文件末尾的 __getattr__）。
else:  # 直接以脚本方式运行同目录文件 / executed as a plain script
    from config import (  # type: ignore
        DEFAULT_CONFIG_PATH,
        PACKAGE_DIR,
        REPO_ROOT,
        data_dir_of,
        deep_update,
        load_config,
        run_dir_of,
        set_by_path,
        validate_config,
    )
    from data import (  # type: ignore
        PAD_ID,
        PAD_TOKEN,
        UNK_ID,
        UNK_TOKEN,
        SentimentDataset,
        Vocabulary,
        build_dataloaders,
        collate_batch,
        load_splits,
        resolve_num_workers,
        stratified_subsample,
        tokenize,
    )
    from metrics import classification_metrics, evaluate  # type: ignore
    from model import (  # type: ignore
        RecurrentClassifier,
        count_parameters,
        load_checkpoint,
        predict_texts,
    )
    from train import (  # type: ignore
        fit,
        run_experiment,
        run_smoke_test,
        set_global_seed,
        train_one_epoch,
    )

#: 需要从 train 模块按需取出的名字 / names resolved lazily from ``train``
_TRAIN_EXPORTS = frozenset({"fit", "run_experiment", "run_smoke_test",
                            "set_global_seed", "train_one_epoch"})


def __getattr__(name):
    """PEP 562：首次访问时才导入 train 模块，避免 `python -m` 时的重复导入警告。"""
    if name in _TRAIN_EXPORTS:
        if __package__:
            from . import train as _train
        else:  # pragma: no cover - 脚本方式下走上面的显式导入
            import train as _train  # type: ignore
        return getattr(_train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "1.0.0"

__all__ = [
    # config
    "DEFAULT_CONFIG_PATH",
    "PACKAGE_DIR",
    "REPO_ROOT",
    "data_dir_of",
    "deep_update",
    "load_config",
    "run_dir_of",
    "set_by_path",
    "validate_config",
    # data
    "PAD_ID",
    "PAD_TOKEN",
    "UNK_ID",
    "UNK_TOKEN",
    "SentimentDataset",
    "Vocabulary",
    "build_dataloaders",
    "collate_batch",
    "load_splits",
    "resolve_num_workers",
    "stratified_subsample",
    "tokenize",
    # metrics
    "classification_metrics",
    "evaluate",
    # model
    "RecurrentClassifier",
    "count_parameters",
    "load_checkpoint",
    "predict_texts",
    # training
    "fit",
    "run_experiment",
    "run_smoke_test",
    "set_global_seed",
    "train_one_epoch",
]
