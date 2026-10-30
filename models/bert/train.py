"""Fine-tune a pretrained BERT classifier on the fixed Yelp splits."""

import argparse
import yaml
from model import build_model
import csv
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


def read_rows(path, limit=None):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if limit is not None:
        rows = rows[:limit]
    return rows


class Reviews(Dataset):
    def __init__(self, rows, tokenizer, max_len):
        self.rows = rows
        self.encodings = tokenizer([r["text"] for r in rows], truncation=True, max_length=max_len)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = int(self.rows[index]["label"])
        return item


def scores(truth, predicted):
    cm = [[0, 0], [0, 0]]
    for y, p in zip(truth, predicted):
        cm[y][p] += 1
    per_class = {}
    for cls in (0, 1):
        tp, fp, fn = cm[cls][cls], cm[1 - cls][cls], cm[cls][1 - cls]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[str(cls)] = {"precision": precision, "recall": recall,
                               "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    return {"accuracy": sum(cm[i][i] for i in (0, 1)) / len(truth),
            "macro_precision": sum(x["precision"] for x in per_class.values()) / 2,
            "macro_recall": sum(x["recall"] for x in per_class.values()) / 2,
            "macro_f1": sum(x["f1"] for x in per_class.values()) / 2,
            "per_class": per_class, "confusion_matrix": cm}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    truth, predicted = [], []
    for batch in loader:
        labels = batch.pop("labels")
        logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
        truth.extend(labels.tolist())
        predicted.extend(logits.argmax(dim=-1).cpu().tolist())
    return scores(truth, predicted), predicted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("processed_data"))
    parser.add_argument("--output", type=Path, default=Path("bert_trial"))
    parser.add_argument("--model", default="google-bert/bert-base-uncased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--val-limit", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    config_path = parser.parse_known_args()[0].config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    defaults = config["training"].copy()
    for key in ("data", "output", "embedding_file"):
        if defaults.get(key) is not None:
            defaults[key] = Path(defaults[key])
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = read_rows(args.data / "train.csv", args.train_limit)
    val_rows = read_rows(args.data / "val.csv", args.val_limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    train = DataLoader(Reviews(train_rows, tokenizer, args.max_len), batch_size=args.batch_size,
                       shuffle=True, collate_fn=collator)
    val = DataLoader(Reviews(val_rows, tokenizer, args.max_len), batch_size=args.batch_size,
                     collate_fn=collator)
    model = build_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    history, best_f1, best_epoch = [], -1.0, None
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train:
            optimizer.zero_grad(set_to_none=True)
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_metrics, _ = evaluate(model, val, device)
        history.append({"epoch": epoch, "validation": val_metrics, "elapsed_seconds": round(time.time() - started, 1)})
        print(f"epoch={epoch} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f}", flush=True)
        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = val_metrics["macro_f1"], epoch
            model.save_pretrained(args.output / "best_model", safe_serialization=True)
            tokenizer.save_pretrained(args.output / "best_model")
            torch.save({"model": model.state_dict(), "config": model.config.to_dict(),
                        "epoch": epoch, "max_len": args.max_len}, args.output / "best_model.pt")
    test_metrics = None
    if args.evaluate_test:
        model = AutoModelForSequenceClassification.from_pretrained(args.output / "best_model").to(device)
        test_rows = read_rows(args.data / "test.csv")
        test = DataLoader(Reviews(test_rows, tokenizer, args.max_len), batch_size=args.batch_size,
                          collate_fn=collator)
        test_metrics, predictions = evaluate(model, test, device)
        with (args.output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sample_id", "label", "prediction"])
            writer.writerows((row["sample_id"], row["label"], pred) for row, pred in zip(test_rows, predictions))
    result = {"seed": args.seed, "model": args.model, "device": str(device), "torch": torch.__version__,
              "train_size": len(train_rows), "val_size": len(val_rows), "max_len": args.max_len,
              "batch_size": args.batch_size, "learning_rate": args.lr, "epochs": args.epochs,
              "best_epoch": best_epoch, "history": history, "test": test_metrics}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "best_validation_f1": best_f1, "test": test_metrics}), flush=True)


if __name__ == "__main__":
    main()
