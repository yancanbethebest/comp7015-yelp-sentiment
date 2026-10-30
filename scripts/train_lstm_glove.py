"""Separate entry point for the fine-tuned GloVe embedding experiment."""
import sys
from train_lstm import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--epochs", "100", "--embedding-dim", "100", "--embedding-file", "glove.6B.100d.txt", "--output", "experiments/lstm_100/glove"]
    main()
