# -*- coding: utf-8 -*-
"""Member 3: pre-trained word-embedding package (GloVe / word2vec)."""
from .embeddings import (
    build_embedding_matrix,
    coverage_report,
    load_glove,
    load_word2vec_text,
)

__all__ = [
    "build_embedding_matrix",
    "coverage_report",
    "load_glove",
    "load_word2vec_text",
]
