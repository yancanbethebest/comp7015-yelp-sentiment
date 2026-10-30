"""Run sentiment inference using either published best BiLSTM checkpoint."""
import argparse
import json
from pathlib import Path

import torch
from train_lstm import BiLSTM, Reviews, collate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text", action="append", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = BiLSTM(len(checkpoint["vocab"]) + 2, checkpoint["embedding_dim"],
                   checkpoint["hidden_dim"], checkpoint["dropout"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    rows = [(str(i), text, 0) for i, text in enumerate(args.text)]
    dataset = Reviews(rows, checkpoint["vocab"], checkpoint["max_len"])
    _, tokens, lengths, _ = collate([dataset[i] for i in range(len(dataset))])
    with torch.no_grad():
        probabilities = model(tokens.to(device), lengths).softmax(-1).cpu().tolist()
    for text, probabilities in zip(args.text, probabilities):
        label = int(probabilities[1] > probabilities[0])
        print(json.dumps({"text": text, "label": label,
                          "sentiment": "positive" if label else "negative",
                          "positive_probability": probabilities[1]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
