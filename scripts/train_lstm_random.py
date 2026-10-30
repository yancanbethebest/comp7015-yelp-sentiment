"""Separate entry point for the random trainable embedding experiment."""
import sys
from train_lstm import main

if __name__ == "__main__":
    if "--embedding-file" in sys.argv:
        raise SystemExit("Use train_lstm_glove.py for pretrained embeddings.")
    sys.argv[1:1] = ["--epochs", "100", "--embedding-dim", "100", "--output", "experiments/lstm_100/random"]
    main()
