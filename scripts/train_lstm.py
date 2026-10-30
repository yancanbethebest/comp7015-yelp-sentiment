"""Train a word-level BiLSTM polarity classifier on the fixed cleaned splits."""

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def read_split(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [(row["sample_id"], row["text"], int(row["label"])) for row in rows]


class Reviews(Dataset):
    def __init__(self, rows, vocab, max_len):
        self.items = []
        for sample_id, text, label in rows:
            ids = [vocab.get(token, 1) for token in TOKEN.findall(text.lower())[:max_len]]
            self.items.append((sample_id, ids or [1], label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(batch):
    ids, tokens, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in tokens], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence([torch.tensor(x) for x in tokens], batch_first=True)
    return ids, padded, lengths, torch.tensor(labels, dtype=torch.long)


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(2 * hidden_dim, 2)

    def forward(self, tokens, lengths):
        embedded = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.output(self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1)))


@torch.no_grad()
def evaluate(model, loader, device, predictions=None):
    model.eval()
    confusion = [[0, 0], [0, 0]]
    total_loss = 0.0
    count = 0
    criterion = nn.CrossEntropyLoss()
    for ids, tokens, lengths, labels in loader:
        logits = model(tokens.to(device), lengths)
        total_loss += criterion(logits, labels.to(device)).item() * len(labels)
        pred = logits.argmax(dim=1).cpu().tolist()
        for sample_id, truth, guess in zip(ids, labels.tolist(), pred):
            confusion[truth][guess] += 1
            if predictions is not None:
                predictions.append((sample_id, truth, guess))
        count += len(labels)
    f1 = []
    for cls in (0, 1):
        tp = confusion[cls][cls]
        fp = confusion[1 - cls][cls]
        fn = confusion[cls][1 - cls]
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    per_class = {}
    for cls in (0, 1):
        tp, fp, fn = confusion[cls][cls], confusion[1 - cls][cls], confusion[cls][1 - cls]
        per_class[str(cls)] = {"precision": tp / (tp + fp) if tp + fp else 0.0,
                               "recall": tp / (tp + fn) if tp + fn else 0.0, "f1": f1[cls]}
    return {"loss": total_loss / count, "accuracy": (confusion[0][0] + confusion[1][1]) / count,
            "macro_f1": sum(f1) / 2, "macro_precision": sum(x["precision"] for x in per_class.values()) / 2,
            "macro_recall": sum(x["recall"] for x in per_class.values()) / 2,
            "per_class": per_class, "confusion_matrix": confusion}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("processed_data"))
    parser.add_argument("--output", type=Path, default=Path("lstm_results"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--embedding-file", type=Path, help="GloVe text file; omitted means random trainable embeddings")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--freeze-embedding", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true", help="Run official test only after final model selection")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = read_split(args.data / "train.csv")
    val_rows = read_split(args.data / "val.csv")
    test_rows = read_split(args.data / "test.csv")
    frequencies = Counter(token for _, text, _ in train_rows for token in TOKEN.findall(text.lower()))
    vocab = {token: index + 2 for index, (token, _) in enumerate(frequencies.most_common(args.vocab_size - 2))}
    loaders = {}
    for split, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        loaders[split] = DataLoader(Reviews(rows, vocab, args.max_len), batch_size=args.batch_size,
                                    shuffle=split == "train", collate_fn=collate, num_workers=2)
    model = BiLSTM(len(vocab) + 2, args.embedding_dim, args.hidden_dim, args.dropout)
    matched = 0
    if args.embedding_file:
        with args.embedding_file.open(encoding="utf-8") as stream:
            for line in stream:
                fields = line.rstrip().split()
                if len(fields) != args.embedding_dim + 1:
                    continue
                index = vocab.get(fields[0])
                if index is not None:
                    model.embedding.weight.data[index] = torch.tensor([float(x) for x in fields[1:]])
                    matched += 1
        if args.freeze_embedding:
            model.embedding.weight.requires_grad_(False)
    model = model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    history = []
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for _, tokens, lengths, labels in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tokens.to(device), lengths), labels.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        train_metrics = evaluate(model, loaders["train"], device)
        val_metrics = evaluate(model, loaders["val"], device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(f"epoch={epoch} train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f}", flush=True)
        if val_metrics["macro_f1"] > best:
            best = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "vocab": vocab, "max_len": args.max_len,
                        "epoch": epoch}, args.output / "best_model.pt")
    checkpoint = torch.load(args.output / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = None
    if args.evaluate_test:
        predictions = []
        test_metrics = evaluate(model, loaders["test"], device, predictions)
        with (args.output / "test_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sample_id", "label", "prediction"])
            writer.writerows(predictions)
    report = {"seed": 42, "device": str(device), "torch_version": torch.__version__,
              "architecture": "word-level bidirectional LSTM",
              "tokenizer": "lowercase regex word-or-punctuation; vocabulary fit on train only",
              "vocab_size": len(vocab) + 2, "max_len": args.max_len, "batch_size": args.batch_size,
              "embedding_source": str(args.embedding_file) if args.embedding_file else "random_trainable",
              "embedding_dim": args.embedding_dim, "embedding_coverage": matched / len(vocab) if vocab else 0,
              "embedding_frozen": args.freeze_embedding, "hidden_dim": args.hidden_dim,
              "dropout": args.dropout, "learning_rate": args.lr,
              "epochs_requested": args.epochs, "best_epoch": best_epoch, "history": history,
              "test": test_metrics}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "test": test_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
