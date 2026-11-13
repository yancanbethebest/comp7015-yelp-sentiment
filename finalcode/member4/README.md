# Member 4: Transformer track (DeBERTa-v3 / DistilBERT)

This directory contains my part of the COMP7015 group project: the **Transformer
track** of the Yelp Review Polarity sentiment classifier.

The group ran a staged campaign: 22 seed-42 candidates were screened, one winner
was selected, then re-trained with two extra seeds (123 / 2026) to confirm the
result before freezing the protocol and evaluating **once** on the official
38,000-row test split. My track contributed the Transformer half of that search,
was responsible for the winning configuration, and produced the test-side
evaluation and the training-cost comparison.

## What I contributed

| Item | Detail |
|---|---|
| Candidate configs | 9 Transformer configs (3 DistilBERT families × length/policy/lr variants + DeBERTa-v3-base variants) |
| Winning model | `deberta_L384`, DeBERTa-v3-base, max length 384, lr 2e-5, bf16, 3 epochs |
| Confirmation seeds | 123 and 2026 (full 3-epoch retrain each) |
| Test evaluation | All 36 frozen runs evaluated on the 38,000-row test split |
| Analysis | Validation-vs-test consistency, training-cost trade-off (seconds/point of F1), per-class and paired-seed comparisons |

## Directory layout

```
member4/
├── README.md                     <- this file
├── README.zh-CN.md               <- same document in Chinese
├── requirements.txt
├── code/
│   ├── core.py                   <- datasets, model registry, metrics, hashing
│   ├── prepare_full.py           <- build pinned train/val/test splits + RNN vocab/GloVe
│   ├── run_experiment.py         <- single-GPU trainer (train / evaluate)
│   ├── rnn_full.py               <- RNN trainer (used to recompute the imported runs)
│   ├── campaign.py               <- screening -> confirmation -> freeze -> test -> report
│   ├── report.py                 <- metric recomputation, mean/SD, paired tests, plots
│   ├── recover.py                <- rebuild a run from recovered weights
│   ├── matrix.json               <- all 22 candidate configurations (RNN + Transformer)
│   └── original_rnn/             <- RNN baseline (Member 2's implementation, also reused by Member 3)
├── configs/
│   └── transformer_configs.yaml  <- my 9 Transformer configs, human-readable
```

## Environment

- Python 3.11+, **one CUDA GPU** (tests were run on a 96 GB RTX PRO 6000 Blackwell)
- `transformers==4.57.1` (pinned; the trainer asserts this exact version)
- bf16 support required when `--precision bf16`
- `pip install -r requirements.txt`

Everything runs on a single GPU; no multi-GPU or TPU path is used.

## Data preparation

The splits are produced from the shared `processed_data/` directory committed by
Member 1 (`finalcode/member1/processed_data/{train,val,test}.csv`). Assuming this
folder lives at `finalcode/member4/`, run from `finalcode/member4`:

```bash
python code/prepare_full.py --repo ../member1 --out ./data
```

(`--repo` must be the directory that *contains* `processed_data/`.)

This checks the pinned SHA-256 of every split, verifies that sample IDs and exact
texts are disjoint across splits, copies `train.csv` / `val.csv` / `test.csv`,
builds the train-only 30k vocabulary (`vocab.json`) and the GloVe-100d matrix used
by the RNN baseline. Expected sizes: 503,979 / 55,998 / 38,000 rows.

If the CSVs are stored with Git LFS, run `git lfs pull` first. The script rejects
LFS pointer files.

## Training

Single configuration (the winning one):

```bash
python code/run_experiment.py train \
  --data ./data --out ./runs/deberta_L384_s42 \
  --model deberta --length 384 --policy head --lr 2e-5 --seed 42 \
  --precision bf16 --micro-batch 8 --gradient-checkpointing
```

Each epoch writes an immutable, self-verifying checkpoint directory
(`model/`, `COMPLETE.json`, `metrics.json`, `validation.csv`, `training_state.pt`,
`preprocessing.json`, `reload_check.json`). Training is resumable at epoch
boundaries with `--resume`.

All nine Transformer candidates:

```bash
for cfg in deberta_base deberta_lr1 deberta_lr3 deberta_L384 \
           distilbert_base distilbert_lr3 distilbert_headtail \
           distilbert_L384 distilbert_L512; do
  # read model / length / lr / policy from configs/transformer_configs.yaml
  python code/run_experiment.py train --data ./data --out ./runs/${cfg}_s42 ...
done
```


## Evaluation on the frozen test set

```bash
python code/run_experiment.py evaluate --data ./data --root ./runs --run deberta_L384_s42
```

This writes `predictions.csv` (38,000 rows) plus test metrics (`accuracy`,
`macro_f1`, `nll`, `brier_positive_class`, `ece_15_equal_width`). The test split is
only touched after `FINAL_LOCK.json` exists; the code refuses otherwise.

## Results (test set, 38,000 rows)

| Config | Macro-F1 (mean ± SD, 3 seeds) | NLL | Train time |
|---|---|---|---|
| **deberta_L384** | **0.9831 ± 0.0004** | 0.0763 | 42,293 s (3 seeds) |
| deberta_base | 0.9809 ± 0.0004 | 0.0920 | 20,158 s |
| distilbert_base | 0.9707 ± 0.0009 | 0.1328 | 3,375 s |
| distilbert_lr3 | 0.9701 ± 0.0003 | 0.1370 | 3,377 s |
| lstm_length256 (RNN best, Member 3) | 0.9604 ± 0.0007 | 0.1077 | 2,726 s |

Validation macro-F1 of the winner was 0.9821 and test macro-F1 0.9831, i.e. no
validation-to-test drop. The full 36-run table and the comparison figure are in the project report.

## Reproducibility notes

- Every split is pinned by SHA-256 in `code/core.py` (`PARENT_SHA`) and re-checked
  at load time; row counts are asserted.
- Model checkpoints are pinned to exact Hub revisions
  (`microsoft/deberta-v3-base@8ccc9b6`, `distilbert-base-uncased@12040acc`).
- Determinism: fixed seeds, `cudnn.benchmark=False`, TF32 disabled,
  `torch.use_deterministic_algorithms(True)`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- `FINAL_LOCK.json` records the protocol hash, selection hash and every frozen run.

## A note on the variable named `tokens`

Static-analysis tools occasionally flag `code/original_rnn/model.py` because a
variable called `tokens` appears next to `logits`. In this codebase `tokens` is
the standard NLP sense of the word: the `(B, T)` integer tensor of vocabulary
ids for one batch of reviews, produced by `Vocabulary.encode`. There are no
credentials, API keys or access tokens anywhere in this project, and nothing is
logged apart from scalar metrics. The flag is a heuristic false positive.

## Third-party libraries

PyTorch, Hugging Face Transformers, NumPy, pandas, SciPy, Matplotlib,
sentencepiece. GloVe-100d vectors come from the `gensim-data` release
(`glove-wiki-gigaword-100`). The RNN model/data code in `original_rnn/` is the
implementation authored by Member 2 (the same code Member 3 builds on for their
`lstm_pretrained` variant); it is vendored here so this folder stays self-contained.
