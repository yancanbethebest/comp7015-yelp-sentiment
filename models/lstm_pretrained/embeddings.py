# -*- coding: utf-8 -*-
"""Load pre-trained word vectors and align them to member 2's vocabulary.

This is member 3's core helper. It turns a big text file of word vectors
(GloVe or word2vec) into a numpy matrix whose row order matches the vocabulary
built by member 2. Row alignment is what makes the random-vs-pretrained
comparison fair: if the order is wrong, every word gets the wrong vector and
the experiment silently becomes meaningless.
"""

from pathlib import Path
from typing import Dict

import numpy as np

from models.lstm_gru_random import PAD_TOKEN, Vocabulary


def load_glove(path, dim=100):
    """Read a GloVe text file (word + space-separated floats on each line).

    Tolerates both the plain Stanford format and the gensim format whose first
    line is the header "<vocab_size> <dim>".
    """
    path = Path(path)
    vectors = {}
    with path.open(encoding="utf-8") as stream:
        first_line = True
        for line in stream:
            parts = line.rstrip("\n").split(" ")
            if first_line:
                first_line = False
                if len(parts) == 2:
                    continue
            if len(parts) != int(dim) + 1:
                continue
            try:
                vectors[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
            except ValueError:
                continue
    return vectors


def load_word2vec_text(path):
    """Read a word2vec text-format file (first line: vocab_size dimension)."""
    path = Path(path)
    vectors = {}
    with path.open(encoding="utf-8") as stream:
        header = stream.readline().split()
        if len(header) == 2:
            dim = int(header[1])
        else:
            raise ValueError("word2vec text file must start with '<vocab_size> <dim>'")
        for line in stream:
            parts = line.rstrip("\n").split(" ")
            if len(parts) != dim + 1:
                continue
            try:
                vectors[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
            except ValueError:
                continue
    return vectors


def build_embedding_matrix(vocab, vectors, dim, seed=42):
    """Return (len(vocab), dim) matrix and the number of matched words.

    Words missing from the pre-trained file keep the same random uniform
    initialization member 2 uses, so the only difference between the random
    baseline and the pre-trained version is where known words start from.
    """
    rng = np.random.RandomState(seed)
    matrix = rng.uniform(-0.1, 0.1, size=(len(vocab), dim)).astype(np.float32)
    matrix[vocab.token_to_id[PAD_TOKEN]] = 0.0

    matched = 0
    for token, index in vocab.token_to_id.items():
        vector = vectors.get(token)
        if vector is not None and int(vector.shape[0]) == int(dim):
            matrix[index] = vector
            matched += 1
    return matrix, matched


def coverage_report(vocab, vectors):
    """Fraction of vocabulary tokens that have a pre-trained vector."""
    if len(vocab) == 0:
        return 0.0
    known = sum(1 for token in vocab.token_to_id if token in vectors)
    return known / len(vocab)