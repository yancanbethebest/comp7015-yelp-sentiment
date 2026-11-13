# -*- coding: utf-8 -*-
"""Data loading: vocabulary, dynamic padding, DataLoader.

This module turns "the CSV delivered by Member 1" into a DataLoader that feeds
the model directly:

    processed_data/{train,val,test}.csv  (sample_id,text,label)
        └── tokenize()           lowercase + word/punctuation splitting (identical to Member 1's reference implementation)
        └── Vocabulary.build()   fitted on the training split only, PAD=0 / UNK=1
        └── SentimentDataset     lazy tokenisation; the 83.67M tokens are never cached in memory
        └── collate_batch()      per-batch dynamic padding + true length of every sequence
        └── DataLoader

Why lazy tokenisation is the default:
    The training split holds 83,672,736 tokens. Caching them as Python
    ``list[int]`` would cost about 2.34 GB for the tokens alone; caching them as
    an int32 array (after ``max_len=256`` truncation) still costs about 272 MB.
    With lazy tokenisation only the raw text stays in memory (about 0.4 GB), at
    the price of a few extra seconds of CPU time per epoch, a clear win for a
    20-epoch run.
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
# Special tokens and tokenisation rule
# ---------------------------------------------------------------------------
PAD_TOKEN: str = "<pad>"
UNK_TOKEN: str = "<unk>"
PAD_ID: int = 0
UNK_ID: int = 1

#: Regex copied verbatim from Member 1's scripts/train_lstm.py:17 so historical results stay comparable
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

#: Required columns of the delivered CSV
REQUIRED_COLUMNS: Tuple[str, ...] = ("sample_id", "text", "label")

#: Split sizes delivered by Member 1 (fail-fast checks, no silent tolerance)
EXPECTED_ROWS: Dict[str, int] = {"train": 503979, "val": 55998, "test": 38000}


def tokenize(text: str) -> List[str]:
    """Lowercase and split into words or single punctuation marks.

    Args:
        text: Raw review text.

    Returns:
        List of tokens.

    Examples:
        >>> tokenize("The food Wasn't GREAT!!")
        ['the', 'food', 'wasn', "'", 't', 'great', '!', '!']
    """
    return TOKEN_PATTERN.findall(str(text).lower())


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
class Vocabulary:
    """Training-only token<->id mapping.

    id convention:
        0 = ``<pad>`` (padding; its embedding row stays zero and receives no gradient)
        1 = ``<unk>`` (out-of-vocabulary token)
        2..N assigned by ``(frequency descending, lexicographic ascending)``.

    The ordering is **deterministic**, which differs from Member 1's
    ``Counter.most_common`` implementation: ties no longer depend on insertion
    order, so the same training set yields the same vocabulary on any machine.
    The trade-off is that the token id order is incompatible with historical
    checkpoints, so old embeddings cannot be reused by id and must be remapped by
    token (this was flagged when handing over to Member 3).

    Args:
        token_to_id: token -> id mapping (must already contain ``<pad>`` and ``<unk>``).
        meta: Build metadata (``max_size`` / ``min_freq`` / ``seen_types`` ...), recorded only.
    """

    def __init__(self, token_to_id: Dict[str, int], meta: Optional[Dict[str, Any]] = None) -> None:
        self.token_to_id: Dict[str, int] = dict(token_to_id)
        self.meta: Dict[str, Any] = dict(meta or {})
        self.pad_id: int = self.token_to_id[PAD_TOKEN]
        self.unk_id: int = self.token_to_id[UNK_TOKEN]

    # == basic protocol =====================================================
    def __len__(self) -> int:
        """Return the **number of embedding rows** (including pad/unk), not the token count."""
        return len(self.token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id

    def __repr__(self) -> str:  # pragma: no cover - debug display only
        return (f"Vocabulary(size={len(self)}, kept_types={self.meta.get('kept_types')}, "
                f"oov_token='{UNK_TOKEN}'->{self.unk_id})")

    # == build and persistence ==============================================
    @classmethod
    def build(cls, texts: Sequence[str], max_size: int = 30000,
              min_freq: int = 1) -> "Vocabulary":
        """Build the vocabulary from TRAINING texts only.

        Args:
            texts: Training texts. Passing validation/test texts here would leak data.
            max_size: Total vocabulary size (including ``<pad>``/``<unk>``), i.e. the embedding row count.
            min_freq: Tokens below this frequency are dropped (this project uses 1 and relies on ``max_size`` truncation).

        Returns:
            The built :class:`Vocabulary`.
        """
        counts: Counter = Counter()
        for text in texts:
            counts.update(tokenize(text))

        # Deterministic ordering: frequency descending, then lexicographic ascending
        candidates = [(token, freq) for token, freq in counts.items() if freq >= min_freq]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        candidates = candidates[: max(0, int(max_size) - 2)]      # reserve two slots for special tokens

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
        """Save to JSON (Member 3 reuses the same vocabulary for the GloVe experiment)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"token_to_id": self.token_to_id, "meta": self.meta}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        """Restore a vocabulary from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["token_to_id"], payload.get("meta"))

    # == encoding and statistics ============================================
    def encode(self, text: str, max_len: Optional[int] = None) -> List[int]:
        """Text -> id sequence.

        Args:
            text: Raw text.
            max_len: Truncation length; ``None`` means no truncation.

        Returns:
            A non-empty id list (empty text falls back to ``<unk>`` so ``pack_padded_sequence`` does not fail).
        """
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokenize(text)]
        if max_len is not None:
            ids = ids[: int(max_len)]
        return ids or [self.unk_id]

    def oov_rate(self, texts: Sequence[str]) -> float:
        """Token-level OOV rate (computed on the validation or test split)."""
        total = 0
        oov = 0
        for text in texts:
            for token in tokenize(text):
                total += 1
                if token not in self.token_to_id:
                    oov += 1
        return oov / total if total else 0.0


# ---------------------------------------------------------------------------
# Dataset and batch collation
# ---------------------------------------------------------------------------
class SentimentDataset(Dataset):
    """Binary sentiment dataset.

    Args:
        frame: DataFrame with ``sample_id`` / ``text`` / ``label`` columns.
        vocab: Vocabulary fitted on the training split.
        max_len: Truncation length.
        name: Used in logs and error messages only.
    """

    def __init__(self, frame: pd.DataFrame, vocab: Vocabulary, max_len: int,
                 name: str = "") -> None:
        # Materialise Python lists once so __getitem__ avoids pandas indexing overhead
        self.texts: List[str] = frame["text"].tolist()
        self.labels: List[int] = frame["label"].astype("int64").tolist()
        self.sample_ids: List[str] = frame["sample_id"].tolist()
        self.vocab = vocab
        self.max_len = int(max_len)
        self.name = name

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        """Return (token id tensor, label, sample_id); tokenisation happens lazily here."""
        ids = self.vocab.encode(self.texts[index], self.max_len)
        return torch.tensor(ids, dtype=torch.long), self.labels[index], self.sample_ids[index]


def collate_batch(batch: Sequence[Tuple[torch.Tensor, int, str]]
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Pad a batch to ``(B, T)``, with ``T`` the longest sequence in the batch (dynamic padding).

    Args:
        batch: Samples produced by ``SentimentDataset``.

    Returns:
        ``(tokens, lengths, labels, sample_ids)``

        - ``tokens``: ``(B, T)`` int64, 0 = ``<pad>``
        - ``lengths``: ``(B,)`` int64, true length of each sequence (used by ``pack_padded_sequence``)
        - ``labels``: ``(B,)`` int64
        - ``sample_ids``: Original ``sample_id`` values, used to export per-sample predictions
    """
    sequences, labels, sample_ids = zip(*batch)
    lengths = torch.tensor([int(sequence.numel()) for sequence in sequences], dtype=torch.long)
    tokens = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=PAD_ID)
    return tokens, lengths, torch.tensor(labels, dtype=torch.long), list(sample_ids)


def stratified_subsample(frame: pd.DataFrame, limit: Optional[int], seed: int) -> pd.DataFrame:
    """Stratified head of ``limit`` rows that preserves the positive/negative ratio.

    Only for debugging and the "reproduce the 20k-scale task" control experiment;
    set ``limit`` to ``None`` for real experiments.
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
    """Decide the DataLoader worker count per platform.

    On Windows combined with Jupyter, ``num_workers > 0`` often hangs on spawn, so the default is 0.
    """
    requested = config["dataloader"].get("num_workers")
    if requested is not None:
        return int(requested)
    return 0 if os.name == "nt" else 4


def build_dataloaders(frames: Dict[str, pd.DataFrame], vocab: Vocabulary,
                      config: Dict[str, Any], device: Optional[torch.device] = None,
                      splits: Sequence[str] = ("train", "val")) -> Dict[str, DataLoader]:
    """Build loaders for the requested splits.

    Args:
        frames: ``{"train": DataFrame, "val": ..., "test": ...}``.
        vocab: Vocabulary.
        config: Full configuration (uses ``max_len`` / ``training.batch_size`` /
            ``dataloader.*`` / ``limit_train`` / ``limit_val`` / ``seed``).
        device: Target device; ``pin_memory`` is enabled when it is CUDA.
        splits: Names of the splits to build.

    Returns:
        ``{split: DataLoader}``. The training split is shuffled; validation/test keep their original order so per-sample alignment stays possible.
    """
    num_workers = resolve_num_workers(config)
    want_pin = bool(config["dataloader"].get("pin_memory", True))
    pin_memory = want_pin and (device is not None and device.type == "cuda")
    batch_size = int(config["training"]["batch_size"])
    loaders: Dict[str, DataLoader] = {}

    for split in splits:
        if split not in frames:
            raise KeyError(f"No split named {split!r}; available: {sorted(frames)}")
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
            drop_last=False,          # keep the last partial batch so evaluation is complete
            num_workers=num_workers,
            collate_fn=collate_batch,
            pin_memory=pin_memory,
            generator=generator,
            persistent_workers=(num_workers > 0),
        )
    return loaders


# ---------------------------------------------------------------------------
# Load and verify Member 1's delivery
# ---------------------------------------------------------------------------
def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 in chunks (avoids reading the 374 MB CSV into memory at once)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split(split: str, data_dir: Path) -> pd.DataFrame:
    """Read one split and validate its schema.

    Raises:
        FileNotFoundError: File does not exist (usually: ``git lfs pull`` was skipped).
        ValueError: Missing column, invalid label, empty text or duplicated ``sample_id``.
    """
    path = Path(data_dir) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing split: {path}\n"
            f"If the file holds only a few lines of text, it is a Git LFS pointer; run: git lfs install && git lfs pull"
        )

    frame = pd.read_csv(path, encoding="utf-8")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    frame = frame[list(REQUIRED_COLUMNS)].copy()

    labels = set(frame["label"].dropna().unique().tolist())
    if not labels.issubset({0, 1}):
        raise ValueError(f"{path.name} contains invalid labels: {sorted(labels)}")

    frame["text"] = frame["text"].astype(str)
    if (frame["text"].str.strip() == "").any():
        raise ValueError(f"{path.name}: empty text found")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{path.name}: duplicated sample_id")

    frame["label"] = frame["label"].astype("int64")
    return frame


def load_splits(data_dir: Path, verify: bool = True) -> Tuple[Dict[str, pd.DataFrame],
                                                              Dict[str, Any]]:
    """Load all three splits and optionally audit Member 1's delivery end to end.

    The audit covers checksums against ``reports/manifest.json``, row counts,
    per-split label counts, within-split duplicates and cross-split leakage.

    Args:
        data_dir: Data directory (usually ``processed_data`` in the repository).
        verify: When ``True``, run the checksum/duplicate/leakage checks and raise on any failure.

    Returns:
        ``(frames, audit)``: ``frames`` holds the three DataFrames; ``audit`` is the statistics dict that can go into the report.
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

    # 1) within-split duplicates
    audit["leakage"]["within_split_duplicates"] = {
        split: int(frames[split]["text"].duplicated().sum()) for split in frames
    }

    # 2) cross-split overlap (data leakage)
    text_sets = {split: set(frames[split]["text"]) for split in frames}
    audit["leakage"]["cross_split_overlap"] = {
        "train_vs_val": int(frames["train"]["text"].isin(text_sets["val"]).sum()),
        "train_vs_test": int(frames["train"]["text"].isin(text_sets["test"]).sum()),
        "val_vs_test": int(frames["val"]["text"].isin(text_sets["test"]).sum()),
    }

    # 3) checksums recorded by Member 1
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
        audit["checksums"] = {"skipped": f"manifest.json does not exist: {manifest_path}"}

    # 4) fail fast: refuse to train if any check fails
    wrong_rows = [split for split, info in audit["splits"].items() if not info["matches_expected"]]
    if wrong_rows:
        detail = {split: (audit["splits"][split]["rows"], EXPECTED_ROWS[split]) for split in wrong_rows}
        raise ValueError(f"Unexpected row counts vs. Member 1's delivery (actual, expected): {detail}")

    duplicates = sum(audit["leakage"]["within_split_duplicates"].values())
    if duplicates:
        raise ValueError(f"Duplicated texts: {audit['leakage']['within_split_duplicates']}")

    overlaps = sum(audit["leakage"]["cross_split_overlap"].values())
    if overlaps:
        raise ValueError(f"Cross-split overlap (leakage): {audit['leakage']['cross_split_overlap']}")

    for name, info in audit["checksums"].items():
        if isinstance(info, dict) and "match" in info and not info["match"]:
            raise ValueError(f"{name}: SHA-256 does not match reports/manifest.json")

    return frames, audit
