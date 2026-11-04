# -*- coding: utf-8 -*-
"""数据加载：词表、动态 Padding、DataLoader / Data loading utilities.

本模块把"成员一交付的 CSV"变成一个可直接喂给模型的 DataLoader：

    processed_data/{train,val,test}.csv  (sample_id,text,label)
        └── tokenize()           小写化 + 词/标点切分（与成员一参考实现一致）
        └── Vocabulary.build()   只由训练集拟合，PAD=0 / UNK=1
        └── SentimentDataset     即时分词（lazy），不把 8,367 万 token 缓存在内存里
        └── collate_batch()      batch 内动态 padding + 每条真实长度
        └── DataLoader

为什么默认即时分词 / why lazy tokenisation:
    训练集共 83,672,736 个 token。若缓存成 Python ``list[int]``，仅 token 就要约
    2.34 GB；缓存成 int32 数组（``max_len=256`` 截断后）约 272 MB。改用即时分词后，
    内存只剩文本本身（约 0.4 GB），代价是每轮多几秒 CPU 时间——对 20 轮的训练完全划算。
"""

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# 特殊符号与分词规则 / special tokens and tokenisation rule
# ---------------------------------------------------------------------------
PAD_TOKEN: str = "<pad>"
UNK_TOKEN: str = "<unk>"
PAD_ID: int = 0
UNK_ID: int = 1

#: 与成员一 scripts/train_lstm.py:17 逐字一致的正则，保证历史结果可比
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

#: 交付数据的必备列 / required columns of the delivered CSV
REQUIRED_COLUMNS: Tuple[str, ...] = ("sample_id", "text", "label")

#: 成员一交付的划分规模（用于 fail-fast，不做静默容错）
EXPECTED_ROWS: Dict[str, int] = {"train": 503979, "val": 55998, "test": 38000}


def tokenize(text: str) -> List[str]:
    """小写化 + 词/标点切分 / lowercase and split into words or single punctuation.

    Args:
        text: 原始评论文本。

    Returns:
        token 列表。

    Examples:
        >>> tokenize("The food Wasn't GREAT!!")
        ['the', 'food', 'wasn', "'", 't', 'great', '!', '!']
    """
    return TOKEN_PATTERN.findall(str(text).lower())


# ---------------------------------------------------------------------------
# 词表 / vocabulary
# ---------------------------------------------------------------------------
class Vocabulary:
    """只由训练集拟合的双向映射 / training-only token<->id mapping.

    id 约定 / id convention:
        0 = ``<pad>``（padding；embedding 该行恒为 0 且不参与梯度）
        1 = ``<unk>``（未登录词）
        2..N 按 ``(频次降序, 字典序升序)`` 依次分配。

    排序规则是**确定性**的，这一点和成员一用 ``Counter.most_common`` 的实现不同：
    同频词的先后不再依赖插入顺序，因此同一份训练集在任何机器上都会得到同一份词表。
    代价是 token id 的顺序与历史 checkpoint 不兼容——旧 embedding 不能按 id 直接复用，
    必须按 token 重新映射（交接给成员三时已注明）。

    Args:
        token_to_id: 词 → id 的映射（必须已包含 ``<pad>`` 与 ``<unk>``）。
        meta: 构建信息（``max_size`` / ``min_freq`` / ``seen_types`` …），仅用于记录。
    """

    def __init__(self, token_to_id: Dict[str, int], meta: Optional[Dict[str, Any]] = None) -> None:
        self.token_to_id: Dict[str, int] = dict(token_to_id)
        self.meta: Dict[str, Any] = dict(meta or {})
        self.pad_id: int = self.token_to_id[PAD_TOKEN]
        self.unk_id: int = self.token_to_id[UNK_TOKEN]

    # -- 基础协议 / basic protocol -----------------------------------------
    def __len__(self) -> int:
        """返回 **embedding 行数**（含 pad/unk），不是普通词数。"""
        return len(self.token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id

    def __repr__(self) -> str:  # pragma: no cover - 仅调试展示
        return (f"Vocabulary(size={len(self)}, kept_types={self.meta.get('kept_types')}, "
                f"oov_token='{UNK_TOKEN}'->{self.unk_id})")

    # -- 构建与持久化 / build and persistence -------------------------------
    @classmethod
    def build(cls, texts: Sequence[str], max_size: int = 30000,
              min_freq: int = 1) -> "Vocabulary":
        """从**训练集**文本构建词表 / build the vocabulary from TRAINING texts only.

        Args:
            texts: 训练集文本。传入验证/测试集文本会构成数据泄漏。
            max_size: 词表总规模（含 ``<pad>``/``<unk>``），即 embedding 行数。
            min_freq: 低于该频次的词直接丢弃（本项目取 1，靠 ``max_size`` 截断）。

        Returns:
            构建好的 :class:`Vocabulary`。
        """
        counts: Counter = Counter()
        for text in texts:
            counts.update(tokenize(text))

        # 确定性排序：先按频次降序，再按字典序升序 / deterministic ordering
        candidates = [(token, freq) for token, freq in counts.items() if freq >= min_freq]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        candidates = candidates[: max(0, int(max_size) - 2)]      # 预留两个特殊符

        token_to_id: Dict[str, int] = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        for token, _freq in candidates:
            token_to_id[token] = len(token_to_id)

        meta = {
            "max_size": int(max_size),
            "min_freq": int(min_freq),
            "seen_types": int(len(counts)),
            "kept_types": int(len(candidates)),
            "train_tokens": int(sum(counts.values())),
        }
        return cls(token_to_id, meta)

    def save(self, path: Path) -> Path:
        """保存为 JSON（成员三复用同一份词表 / exported for the GloVe experiment）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"token_to_id": self.token_to_id, "meta": self.meta}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        """从 JSON 还原词表 / restore a vocabulary from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["token_to_id"], payload.get("meta"))

    # -- 编码与统计 / encoding and statistics -------------------------------
    def encode(self, text: str, max_len: Optional[int] = None) -> List[int]:
        """文本 → 整数 id 序列 / text to id sequence.

        Args:
            text: 原始文本。
            max_len: 截断长度；``None`` 表示不截断。

        Returns:
            非空的 id 列表（空文本用 ``<unk>`` 兜底，避免 pack_padded_sequence 报错）。
        """
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokenize(text)]
        if max_len is not None:
            ids = ids[: int(max_len)]
        return ids or [self.unk_id]

    def oov_rate(self, texts: Sequence[str]) -> float:
        """token 级未登录率 / token-level OOV rate（用验证集或测试集统计）。"""
        total = 0
        oov = 0
        for text in texts:
            for token in tokenize(text):
                total += 1
                if token not in self.token_to_id:
                    oov += 1
        return oov / total if total else 0.0


# ---------------------------------------------------------------------------
# Dataset 与 batch 组装 / dataset and collate
# ---------------------------------------------------------------------------
class SentimentDataset(Dataset):
    """情感二分类数据集 / binary sentiment dataset.

    Args:
        frame: 含 ``sample_id`` / ``text`` / ``label`` 三列的 DataFrame。
        vocab: 训练集拟合出来的词表。
        max_len: 截断长度。
        name: 仅用于日志与报错。
    """

    def __init__(self, frame: pd.DataFrame, vocab: Vocabulary, max_len: int,
                 name: str = "") -> None:
        # 先落成 Python list，避免每次 __getitem__ 都走 pandas 的索引开销
        self.texts: List[str] = frame["text"].tolist()
        self.labels: List[int] = frame["label"].astype("int64").tolist()
        self.sample_ids: List[str] = frame["sample_id"].tolist()
        self.vocab = vocab
        self.max_len = int(max_len)
        self.name = name

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        """返回 (token id 张量, 标签, sample_id)；分词在这里即时完成。"""
        ids = self.vocab.encode(self.texts[index], self.max_len)
        return torch.tensor(ids, dtype=torch.long), self.labels[index], self.sample_ids[index]


def collate_batch(batch: Sequence[Tuple[torch.Tensor, int, str]]
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """把一个 batch 补齐成 ``(B, T)``；``T`` 取 batch 内最长序列（动态 padding）。

    Args:
        batch: ``SentimentDataset`` 产出的样本列表。

    Returns:
        ``(tokens, lengths, labels, sample_ids)``

        - ``tokens``: ``(B, T)`` int64，0 = ``<pad>``
        - ``lengths``: ``(B,)`` int64，每条真实长度（供 ``pack_padded_sequence`` 使用）
        - ``labels``: ``(B,)`` int64
        - ``sample_ids``: 原始 ``sample_id``，用于导出逐样本预测
    """
    sequences, labels, sample_ids = zip(*batch)
    lengths = torch.tensor([int(sequence.numel()) for sequence in sequences], dtype=torch.long)
    tokens = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=PAD_ID)
    return tokens, lengths, torch.tensor(labels, dtype=torch.long), list(sample_ids)


def stratified_subsample(frame: pd.DataFrame, limit: Optional[int], seed: int) -> pd.DataFrame:
    """按标签分层抽取 ``limit`` 条，保持正负比例 / stratified head ``limit``.

    仅用于调试与"复现题目 20k 规模"的对照实验；正式实验请把 ``limit`` 设为 ``None``。
    """
    if not limit or int(limit) >= len(frame):
        return frame
    rng = np.random.RandomState(seed)
    total = len(frame)
    parts = []
    for _label, group in frame.groupby("label", sort=True):
        take = int(round(int(limit) * len(group) / total))
        take = max(1, min(take, len(group)))
        picked = rng.choice(len(group), size=take, replace=False)
        parts.append(group.iloc[np.sort(picked)])
    return pd.concat(parts).sort_index()


def resolve_num_workers(config: Dict[str, Any]) -> int:
    """按平台决定 DataLoader 进程数 / worker count.

    Windows 与 Jupyter 组合下 ``num_workers > 0`` 容易因 spawn 卡住，所以默认 0。
    """
    requested = config["dataloader"].get("num_workers")
    if requested is not None:
        return int(requested)
    return 0 if os.name == "nt" else 4


def build_dataloaders(frames: Dict[str, pd.DataFrame], vocab: Vocabulary,
                      config: Dict[str, Any], device: Optional[torch.device] = None,
                      splits: Sequence[str] = ("train", "val")) -> Dict[str, DataLoader]:
    """为指定划分构建 DataLoader / build loaders for the requested splits.

    Args:
        frames: ``{"train": DataFrame, "val": ..., "test": ...}``。
        vocab: 词表。
        config: 完整配置（用到 ``max_len`` / ``training.batch_size`` /
            ``dataloader.*`` / ``limit_train`` / ``limit_val`` / ``seed``）。
        device: 目标设备；为 CUDA 时开启 ``pin_memory``。
        splits: 需要构建的划分名。

    Returns:
        ``{split: DataLoader}``。训练集打乱，验证/测试集保持原顺序便于逐样本对齐。
    """
    num_workers = resolve_num_workers(config)
    want_pin = bool(config["dataloader"].get("pin_memory", True))
    pin_memory = want_pin and (device is not None and device.type == "cuda")
    batch_size = int(config["training"]["batch_size"])
    loaders: Dict[str, DataLoader] = {}

    for split in splits:
        if split not in frames:
            raise KeyError(f"没有名为 {split!r} 的划分，可用：{sorted(frames)}")
        limit_key = "limit_train" if split == "train" else "limit_val"
        frame = stratified_subsample(frames[split], config.get(limit_key), int(config["seed"]))
        dataset = SentimentDataset(frame, vocab, int(config["max_len"]), name=split)

        generator = None
        if split == "train":
            generator = torch.Generator()
            generator.manual_seed(int(config["seed"]))

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            drop_last=False,          # 不丢最后一个不满的 batch，评估才完整
            num_workers=num_workers,
            collate_fn=collate_batch,
            pin_memory=pin_memory,
            generator=generator,
            persistent_workers=(num_workers > 0),
        )
    return loaders


# ---------------------------------------------------------------------------
# 读取与校验成员一交付的数据 / load and verify member 1's delivery
# ---------------------------------------------------------------------------
def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """分块计算 SHA-256（避免把 374 MB 的 CSV 一次读进内存）。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split(split: str, data_dir: Path) -> pd.DataFrame:
    """读取单个划分并做基础合法性检查 / read one split and validate its schema.

    Raises:
        FileNotFoundError: 文件不存在（常见原因：忘了 ``git lfs pull``）。
        ValueError: 缺列、标签非法、空文本或 ``sample_id`` 重复。
    """
    path = Path(data_dir) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少数据文件 / missing split: {path}\n"
            f"如果文件只有几行文字，说明拿到的是 Git LFS 指针，请先执行：git lfs install && git lfs pull"
        )

    frame = pd.read_csv(path, encoding="utf-8")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} 缺少必要列 / missing columns: {missing}")
    frame = frame[list(REQUIRED_COLUMNS)].copy()

    labels = set(frame["label"].dropna().unique().tolist())
    if not labels.issubset({0, 1}):
        raise ValueError(f"{path.name} 出现非法标签 / invalid labels: {sorted(labels)}")

    frame["text"] = frame["text"].astype(str)
    if (frame["text"].str.strip() == "").any():
        raise ValueError(f"{path.name} 存在空文本 / empty text found")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{path.name} 的 sample_id 不唯一 / duplicated sample_id")

    frame["label"] = frame["label"].astype("int64")
    return frame


def load_splits(data_dir: Path, verify: bool = True) -> Tuple[Dict[str, pd.DataFrame],
                                                              Dict[str, Any]]:
    """读取三个划分，可选地完整核对"成员一交付"是否可用。

    Load all three splits and (optionally) run the full acceptance audit:
    checksums against ``reports/manifest.json``, row counts, label balance,
    within-split duplicates and cross-split leakage.

    Args:
        data_dir: 数据目录（通常是仓库下的 ``processed_data``）。
        verify: ``True`` 时执行校验和/重复/泄漏检查，任何一项失败都抛异常。

    Returns:
        ``(frames, audit)``：``frames`` 是三个 DataFrame；``audit`` 是可写进报告的统计字典。
    """
    data_dir = Path(data_dir)
    frames: Dict[str, pd.DataFrame] = {}
    audit: Dict[str, Any] = {"data_dir": str(data_dir), "splits": {}, "checksums": {}, "leakage": {}}

    for split in ("train", "val", "test"):
        frame = load_split(split, data_dir)
        frames[split] = frame
        audit["splits"][split] = {
            "rows": int(len(frame)),
            "negative": int((frame["label"] == 0).sum()),
            "positive": int((frame["label"] == 1).sum()),
            "matches_expected": bool(len(frame) == EXPECTED_ROWS[split]),
        }

    if not verify:
        return frames, audit

    # 1) 集合内重复 / within-split duplicates
    audit["leakage"]["within_split_duplicates"] = {
        split: int(frames[split]["text"].duplicated().sum()) for split in frames
    }

    # 2) 集合间重叠（数据泄漏）/ cross-split leakage
    text_sets = {split: set(frames[split]["text"]) for split in frames}
    audit["leakage"]["cross_split_overlap"] = {
        "train_vs_val": int(frames["train"]["text"].isin(text_sets["val"]).sum()),
        "train_vs_test": int(frames["train"]["text"].isin(text_sets["test"]).sum()),
        "val_vs_test": int(frames["val"]["text"].isin(text_sets["test"]).sum()),
    }

    # 3) 校验和对照 manifest.json / checksums recorded by member 1
    manifest_path = data_dir.parent / "reports" / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.get("output_sha256", {})
        for split in ("train", "val", "test"):
            name = f"{split}.csv"
            actual = sha256_of_file(data_dir / name)
            audit["checksums"][name] = {
                "expected": expected.get(name),
                "actual": actual,
                "match": bool(expected.get(name) == actual),
            }
        audit["manifest"] = {
            "source": manifest.get("source"),
            "cleaning_version": manifest.get("cleaning_version"),
            "seed": manifest.get("seed"),
            "source_revision": manifest.get("source_revision"),
        }
    else:
        audit["checksums"] = {"skipped": f"manifest.json 不存在：{manifest_path}"}

    # 4) fail fast：任何一项不满足都不允许继续训练
    wrong_rows = [split for split, info in audit["splits"].items() if not info["matches_expected"]]
    if wrong_rows:
        detail = {split: (audit["splits"][split]["rows"], EXPECTED_ROWS[split]) for split in wrong_rows}
        raise ValueError(f"行数与成员一交付不一致 / unexpected row counts (actual, expected): {detail}")

    duplicates = sum(audit["leakage"]["within_split_duplicates"].values())
    if duplicates:
        raise ValueError(f"存在集合内重复文本 / duplicated texts: {audit['leakage']['within_split_duplicates']}")

    overlaps = sum(audit["leakage"]["cross_split_overlap"].values())
    if overlaps:
        raise ValueError(f"存在跨集合重叠（数据泄漏）/ leakage: {audit['leakage']['cross_split_overlap']}")

    for name, info in audit["checksums"].items():
        if isinstance(info, dict) and "match" in info and not info["match"]:
            raise ValueError(f"{name} 的 SHA-256 与 reports/manifest.json 不一致 / checksum mismatch")

    return frames, audit
