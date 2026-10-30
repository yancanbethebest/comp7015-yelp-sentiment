"""Reproducible Yelp Polarity cleaning and stratified delivery (rules v1.0)."""

import argparse
import csv
import hashlib
import html
import json
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow
import sklearn
from sklearn.model_selection import train_test_split


DATASET = "https://huggingface.co/datasets/fancyzhx/yelp_polarity/resolve/main/plain_text"
SEED = 42
VERSION = "v1.0"
TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9]*(?:\s+[^<>]*?)?\s*/?>")
URL = re.compile(r"(?<![\w@])(?:https?://|www\.)[^\s<>]+", re.I)
SPACE = re.compile(r"\s+")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean(value, counts):
    value2 = value.translate(str.maketrans({"\u00a0": " ", "\u2007": " ", "\u202f": " ", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'}))
    counts["unicode_replaced"] += value2 != value
    value = value2
    value2 = html.unescape(value)
    counts["html_entity_decoded"] += value2 != value
    value = value2
    value2 = TAG.sub(" ", value)
    counts["html_tag_removed"] += value2 != value
    value = value2
    value2 = URL.sub("[URL]", value)
    counts["url_replaced"] += value2 != value
    value = value2
    counts["replacement_character_flagged"] += "\ufffd" in value
    value2 = CONTROL.sub("", value)
    counts["control_character_removed"] += value2 != value
    value = value2
    value2 = SPACE.sub(" ", value).strip()
    counts["whitespace_normalized"] += value2 != value
    return value2


def download(source, destination):
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(".part")
    try:
        subprocess.run(["curl", "-L", "--fail", "--retry", "3", "--silent", "--show-error",
                        "-o", str(tmp), source], check=True)
        tmp.replace(destination)
    finally:
        tmp.unlink(missing_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.output.resolve()
    raw_dir = root / "raw"
    data_dir = root / "processed_data"
    report_dir = root / "reports"
    for directory in (raw_dir, data_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    records = {}
    flags = []
    for split in ("train", "test"):
        filename = f"{split}-00000-of-00001.parquet"
        source = f"{DATASET}/{filename}"
        local = raw_dir / filename
        download(source, local)
        frame = pd.read_parquet(local)
        counts[f"raw_{split}"] = len(frame)
        rows = []
        for row_id, (raw_text, label) in enumerate(frame[["text", "label"]].itertuples(index=False, name=None)):
            sample_id = f"{split}:{row_id}"
            if not isinstance(raw_text, str) or not raw_text.strip():
                counts["missing_text"] += 1
                flags.append((sample_id, split, row_id, label, "missing_text", raw_text))
                continue
            if type(label) is not int or label not in (0, 1):
                counts["invalid_label"] += 1
                flags.append((sample_id, split, row_id, label, "invalid_label", raw_text))
                continue
            text = clean(raw_text, counts)
            if not text:
                counts["empty_after_cleaning"] += 1
                flags.append((sample_id, split, row_id, label, "empty_after_cleaning", raw_text))
                continue
            if len(text.split()) < 3:
                counts["very_short_flagged"] += 1
                flags.append((sample_id, split, row_id, label, "very_short", raw_text))
            if "\ufffd" in text:
                flags.append((sample_id, split, row_id, label, "replacement_character", raw_text))
            rows.append({"sample_id": sample_id, "text": text, "label": label,
                         "raw_text": raw_text, "source_split": split, "source_row_id": row_id})
        records[split] = pd.DataFrame(rows)

    combined = pd.concat(records.values(), ignore_index=True)
    conflict = combined.groupby("text").label.nunique()
    conflict_texts = set(conflict[conflict > 1].index)
    counts["conflicting_texts"] = len(conflict_texts)
    counts["conflicting_rows"] = int(combined.text.isin(conflict_texts).sum())
    for split in records:
        frame = records[split]
        bad = frame[frame.text.isin(conflict_texts)]
        flags.extend((r.sample_id, split, r.source_row_id, r.label, "conflicting_label", r.raw_text)
                     for r in bad.itertuples(index=False))
        frame = frame[~frame.text.isin(conflict_texts)].copy()
        counts[f"duplicate_{split}"] = int(frame.duplicated("text").sum())
        records[split] = frame.drop_duplicates("text", keep="first")

    test_texts = set(records["test"].text)
    overlap = records["train"].text.isin(test_texts)
    counts["train_test_overlap_removed"] = int(overlap.sum())
    records["train"] = records["train"][~overlap].copy()
    counts["eligible_train"] = len(records["train"])
    counts["eligible_test"] = len(records["test"])

    def balanced_sample(frame, per_class):
        parts = []
        for label in (0, 1):
            pool = frame[frame.label == label]
            n = min(len(pool), per_class)
            parts.append(pool.sample(n=n, random_state=SEED))
        return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

    train_pool = balanced_sample(records["train"], 10000)
    if len(train_pool) == 20000:
        train, val = train_test_split(train_pool, test_size=0.2, stratify=train_pool.label, random_state=SEED)
    else:
        train_parts, val_parts = [], []
        for label in (0, 1):
            cls = train_pool[train_pool.label == label]
            val_parts.append(cls.iloc[: min(2000, len(cls) // 5)])
            train_parts.append(cls.iloc[min(2000, len(cls) // 5):])
        train, val = pd.concat(train_parts), pd.concat(val_parts)
    outputs = {"train": train, "val": val, "test": balanced_sample(records["test"], 2500)}

    for split, frame in outputs.items():
        frame = frame[["sample_id", "text", "label", "raw_text", "source_split", "source_row_id"]]
        frame[["sample_id", "text", "label"]].to_csv(data_dir / f"{split}.csv", index=False, encoding="utf-8")
        frame.assign(cleaning_version=VERSION).to_csv(report_dir / f"{split}_audit.csv", index=False, encoding="utf-8")
        counts[f"output_{split}"] = len(frame)
        for label in (0, 1):
            counts[f"output_{split}_label_{label}"] = int((frame.label == label).sum())
            counts[f"output_{split}_label_{label}_ratio"] = round(float((frame.label == label).mean()), 6) if len(frame) else 0
        for unit, lengths in (("chars", frame.text.str.len()), ("words", frame.text.str.split().str.len())):
            for percentile in (50, 95, 99):
                counts[f"{split}_{unit}_p{percentile}"] = float(lengths.quantile(percentile / 100)) if len(lengths) else 0

    with (report_dir / "flagged_records.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "source_split", "source_row_id", "label", "reason", "raw_text"])
        writer.writerows(flags)
    counts["flagged_rows"] = len(flags)
    with (report_dir / "cleaning_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        writer.writerows(sorted(counts.items()))
    manifest = {"source": "fancyzhx/yelp_polarity", "cleaning_version": VERSION, "seed": SEED,
                "source_sha256": {p.name: sha256(p) for p in raw_dir.glob("*.parquet")},
                "output_sha256": {p.name: sha256(p) for p in data_dir.glob("*.csv")},
                "pandas": pd.__version__, "pyarrow": pyarrow.__version__,
                "scikit_learn": sklearn.__version__, "source_revision": "bbf1c97a1f0cf005e5aded43839fd814654a1557"}
    (report_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: counts[k] for k in ("raw_train", "raw_test", "output_train", "output_val", "output_test", "conflicting_rows", "train_test_overlap_removed")}, indent=2))


if __name__ == "__main__":
    main()
