"""Shared experiment contracts. No training or test evaluation on import."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

COMMIT = '1d0113e5f0aa3f3f2a5d01ff50ee3fb763136c54'
IDS_SHA = '54a144b385a35b6e58c47dcd4490d9d5f3d8baba41e60f032b5fcbd34970371b'
MODELS = {
    'distilbert': ('distilbert/distilbert-base-uncased', '12040accade4e8a0f71eabdb258fecc2e7e948be'),
    'deberta': ('microsoft/deberta-v3-base', '8ccc9b6f36199bec6961081d44eb72fb3f7353f3'),
}
PARENT_SHA = {
    'train': '632ded109817ba4d4571a1b3f56b9b3dd5973981c3539cc6e312d90fd3e8cdce',
    'val': 'babbe945a973fc84a736cffa14e49f661f15d23048b777e73244b60d65c2a9c0',
    'test': 'd966f8bc9aa2c37b785ba7757df125eb8b21b8e986176f062e79c8bef62ad372',
}
SPLIT_SHA = PARENT_SHA.copy()
COUNTS = {'train': 503979, 'val': 55998, 'test': 38000}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def hashes(folder):
    return {p.relative_to(folder).as_posix(): sha(p) for p in sorted(Path(folder).rglob('*')) if p.is_file() and p.name != 'COMPLETE.json'}

def verify_folder(folder):
    record = read_json(Path(folder) / 'COMPLETE.json')
    require(hashes(folder) == record, f'Checkpoint files changed/incomplete: {folder}')

def read_frame(path):
    f = pd.read_csv(path, dtype={'sample_id': str, 'text': str}, keep_default_na=False)
    require({'sample_id', 'text', 'label'} <= set(f.columns), f'Wrong CSV columns: {path}')
    require(f.sample_id.is_unique and f.sample_id.str.len().gt(0).all(), 'Invalid IDs')
    require(f.text.str.strip().str.len().gt(0).all(), 'Blank text')
    require(f.label.isin([0, 1]).all(), 'Invalid labels')
    return f

def load_split(data, split):
    data = Path(data)
    manifest = read_json(data / 'data_contract.json')
    path = data / f'{split}.csv'
    require(manifest['files'][split]['sha256'] == SPLIT_SHA[split] and manifest['files'][split]['rows'] == COUNTS[split], 'Data contract is not the pinned historical split')
    require(sha(path) == manifest['files'][split]['sha256'], f'CSV changed: {path}')
    f = read_frame(path)
    require(len(f) == manifest['files'][split]['rows'], 'Wrong row count')
    return f

def truncate(ids, max_length, special_count, policy):
    budget = max_length - special_count
    if len(ids) <= budget:
        return list(ids)
    if policy == 'head':
        return ids[:budget]
    # Protocol deliberately uses 128 + 126 at length 256, unlike old 127 + 127.
    front = max_length // 2
    return ids[:front] + ids[-(budget - front):]

def choose_epoch(rows):
    # Anchor tolerance to the global maximum; avoids order-dependent tolerance drift.
    top = max(r['macro_f1'] for r in rows)
    eligible = [r for r in rows if top - r['macro_f1'] <= 1e-4]
    return min(eligible, key=lambda r: (r['nll'], r['epoch']))

def scores(logits, labels):
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    require(z.shape == (len(y), 2) and np.isfinite(z).all(), 'Invalid logits')
    shifted = z - z.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    p = np.exp(logp)
    pred = z.argmax(1)
    cm = np.zeros((2, 2), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    den = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * cm.diagonal(), den, out=np.zeros(2), where=den != 0)
    conf = p.max(1)
    ece = sum(float(mask.mean()) * abs(float((pred[mask] == y[mask]).mean()) - float(conf[mask].mean()))
              for i in range(15) if (mask := (conf > i / 15) & (conf <= (i + 1) / 15)).any())
    return {'rows': len(y), 'accuracy': float((pred == y).mean()), 'correct': int((pred == y).sum()),
            'macro_f1': float(f1.mean()), 'nll': float(-logp[np.arange(len(y)), y].mean()),
            'brier_positive_class': float(((p[:, 1] - y)**2).mean()), 'ece_15_equal_width': ece,
            'confusion_matrix_true_rows': cm.tolist()}
