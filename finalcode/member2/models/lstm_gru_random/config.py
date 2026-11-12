# -*- coding: utf-8 -*-
"""配置加载、合并与校验 / Configuration loading, merging and validation.

优先级 / precedence
-------------------
    命令行参数  >  --config 指定的 YAML  >  包内默认 config.yaml

路径约定 / path convention
--------------------------
    YAML 中的 ``data_dir`` / ``runs_dir`` 是**相对仓库根目录**的路径，
    由 :func:`data_dir_of` / :func:`run_dir_of` 解析成绝对路径。
    这样无论从仓库根目录、包目录还是 Notebook 里调用，结果都一致。
"""

import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

try:
    import yaml
except ImportError as error:  # pragma: no cover - 依赖缺失时给出明确提示
    raise ImportError("需要 PyYAML：python -m pip install -r requirements-training.txt") from error

PACKAGE_DIR: Path = Path(__file__).resolve().parent
"""本包目录 / this package directory（models/lstm_gru_random）。"""

DEFAULT_CONFIG_PATH: Path = PACKAGE_DIR / "config.yaml"
"""默认配置文件 / default YAML shipped with the package。"""

REPO_ROOT: Path = PACKAGE_DIR.parents[1]
"""仓库根目录 / repository root（包含 processed_data/ 的目录）。"""

#: 配置里必须存在的顶层键 / mandatory top-level keys
REQUIRED_KEYS = (
    "run_name",
    "seed",
    "data_dir",
    "runs_dir",
    "max_len",
    "vocab_max_size",
    "vocab_min_freq",
    "model",
    "training",
    "dataloader",
    "amp",
)

#: 模型配置必须存在的键（同时也是 RecurrentClassifier 的参数名）
MODEL_KEYS = (
    "rnn_type",
    "embedding_dim",
    "hidden_dim",
    "num_layers",
    "bidirectional",
    "dropout",
    "embedding_dropout",
    "pooling",
)

TRAINING_KEYS = ("epochs", "batch_size", "lr", "weight_decay", "grad_clip",
                 "early_stopping", "scheduler")


def deep_update(base: Mapping[str, Any], patch: Mapping[str, Any]) -> Dict[str, Any]:
    """递归合并两个字典，``patch`` 覆盖 ``base``。

    Deep-merge two mappings; values from ``patch`` win. Neither input is mutated.

    Examples:
        >>> deep_update({"a": {"b": 1, "c": 2}}, {"a": {"b": 9}})
        {'a': {'b': 9, 'c': 2}}
    """
    result: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def set_by_path(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """按 ``"training.lr"`` 这种点号路径写入配置 / set a nested value in place。

    Args:
        config: 待修改的配置字典（原地修改）。
        dotted_key: 点号分隔的路径，例如 ``"model.dropout"``。
        value: 新值。
    """
    keys = dotted_key.split(".")
    target: Dict[str, Any] = config
    for key in keys[:-1]:
        node = target.get(key)
        if not isinstance(node, dict):
            node = {}
            target[key] = node
        target = node
    target[keys[-1]] = value


def validate_config(config: Mapping[str, Any]) -> None:
    """校验配置的完整性与取值范围，错误信息尽量指出具体键名。

    Validate the config and raise ``ValueError`` with an actionable message.

    Raises:
        ValueError: 缺少必需键，或某键取值非法。
    """
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"配置缺少必需键 / missing keys: {missing}")

    model = config["model"]
    training = config["training"]
    missing_model = [key for key in MODEL_KEYS if key not in model]
    if missing_model:
        raise ValueError(f"配置 model 段缺少键 / missing model keys: {missing_model}")
    missing_training = [key for key in TRAINING_KEYS if key not in training]
    if missing_training:
        raise ValueError(f"配置 training 段缺少键 / missing training keys: {missing_training}")

    if model["rnn_type"] not in ("lstm", "gru"):
        raise ValueError(f"model.rnn_type 只能是 'lstm' 或 'gru'，当前为 {model['rnn_type']!r}")
    if model["pooling"] not in ("last", "max"):
        raise ValueError(f"model.pooling 只能是 'last' 或 'max'，当前为 {model['pooling']!r}")

    positive_checks = {
        "max_len": config["max_len"],
        "vocab_max_size": config["vocab_max_size"],
        "emb": model["embedding_dim"],
        "hid": model["hidden_dim"],
        "layers": model["num_layers"],
        "epochs": training["epochs"],
        "batch_size": training["batch_size"],
    }
    for name, value in positive_checks.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} 必须是正整数 / must be a positive integer，当前为 {value!r}")

    if not 0.0 <= float(model["dropout"]) < 1.0:
        raise ValueError("model.dropout 必须在 [0, 1) 内")
    if not 0.0 <= float(model["embedding_dropout"]) < 1.0:
        raise ValueError("model.embedding_dropout 必须在 [0, 1) 内")
    if float(training["lr"]) <= 0:
        raise ValueError("training.lr 必须大于 0")


def load_config(path: Optional[Path] = None,
                overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """读取 YAML 配置，叠加 ``overrides`` 后做校验。

    Load the YAML config, deep-merge ``overrides`` and validate it.

    Args:
        path: YAML 路径；``None`` 时使用包内 ``config.yaml``。
        overrides: 形如 ``{"training": {"lr": 5e-4}, "model": {"dropout": 0.5}}`` 的覆盖字典。

    Returns:
        完整配置字典（可直接传给 :func:`train.run_experiment`）。

    Raises:
        FileNotFoundError: 配置文件不存在。
        ValueError: 配置不合法。
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件 / config not found: {config_path}")

    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"配置文件顶层必须是映射 / YAML root must be a mapping: {config_path}")

    if overrides:
        config = deep_update(config, overrides)

    # 补上 optional 键的默认值，避免下游到处写 config.get(...)
    config.setdefault("eval_test_after_selection", False)
    config.setdefault("limit_train", None)
    config.setdefault("limit_val", None)
    config["dataloader"].setdefault("num_workers", None)
    config["dataloader"].setdefault("pin_memory", True)

    validate_config(config)
    return config


def _resolve(base: Path, value: str) -> Path:
    """相对路径按 ``base`` 解析，绝对路径原样返回。"""
    path = Path(value)
    return path if path.is_absolute() else (base / path)


def data_dir_of(config: Mapping[str, Any]) -> Path:
    """返回数据目录的绝对路径 / absolute path of ``processed_data``."""
    return _resolve(REPO_ROOT, str(config["data_dir"]))


def run_dir_of(config: Mapping[str, Any]) -> Path:
    """返回本次实验的输出目录 ``runs/<run_name>`` 的绝对路径。"""
    return _resolve(REPO_ROOT, str(config["runs_dir"])) / str(config["run_name"])
