# -*- coding: utf-8 -*-
"""Model definition and inference.

============================ Architecture ============================
    tokens ──► Embedding(V, d, padding_idx=0)   ← randomly initialised, trainable
           ──► EmbeddingDropout(0.1)
           ──► pack_padded_sequence             ← PAD never enters the recurrence
           ──► Bi{LSTM|GRU}(hidden, num_layers) ← uni-/bidirectional
           ──► [final bidirectional states | masked max pooling]
           ──► Dropout(0.3)
           ──► Linear(2*hidden, 2)

Parameters with the default config:
    Embedding 30,000 × 100 = 3,000,000
    BiLSTM bidirectional   =   235,520     (BiGRU: 176,640)
    Linear 256 → 2         =       514
    total BiLSTM = 3,236,034 / BiGRU = 3,177,154

======================= Interface for Member 3 =======================
    The ``embedding_weights`` argument of ``RecurrentClassifier`` accepts any
    ``(vocab_size, embedding_dim)`` tensor. Pass a GloVe/Word2Vec matrix aligned
    to the vocabulary to obtain the "pretrained embedding" variant; the recurrent
    layer, pooling, dropout and classification head stay exactly the same, so the
    "random vs pretrained" comparison changes **one** variable: where the
    initialisation comes from.

    ```python
    from models.lstm_gru_random import RecurrentClassifier, Vocabulary
    vocab = Vocabulary.load("runs/member2_bilstm_random_full_seed42/vocab.json")   # must reuse the same vocabulary
    weights = build_glove_matrix(vocab, "glove.6B.100d.txt")   # (len(vocab), 100)
    # NOTE: build_glove_matrix is not defined in this folder. It lives in
    # Member 3's models/lstm_pretrained/embeddings.py, next to load_glove()
    # and coverage_report(). The snippet shows the calling convention only.
    model = RecurrentClassifier(vocab_size=len(vocab), embedding_weights=weights)
    ```
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

if __package__:  # imported as a package
    from .data import PAD_ID, Vocabulary
else:            # executed as a plain script
    from data import PAD_ID, Vocabulary  # type: ignore


class RecurrentClassifier(nn.Module):
    """Bidirectional LSTM / GRU sentiment classifier with randomly initialised embeddings.

    Args:
        vocab_size: Number of embedding rows, equal to ``len(Vocabulary)`` (including ``<pad>``/``<unk>``).
        embedding_dim: Token embedding dimension; 100 matches GloVe 6B-100d.
        hidden_dim: Hidden units per direction.
        num_layers: Number of recurrent layers.
        rnn_type: ``"lstm"`` or ``"gru"``.
        bidirectional: Whether the recurrence runs in both directions.
        dropout: Dropout between the recurrent output and the classification head.
        embedding_dropout: Dropout on the embedding output (Gal & Ghahramani, 2016).
        pooling: ``"last"`` (concatenated final bidirectional states) or ``"max"`` (masked max pooling).
        num_classes: Number of output classes; 2 for binary classification.
        padding_idx: id of PAD; its embedding row is never updated.
        embedding_weights: **The interface for Member 3**. Optional pretrained embedding matrix.
            Shape must be ``(vocab_size, embedding_dim)``; when given, the layer is
            initialised with these values, and the caller decides whether to freeze
            it via ``model.embedding.weight.requires_grad_(False)``.

    Note:
        The ``config["model"]`` stored in a checkpoint can be expanded directly
        into this class's arguments with ``**``, so the ``model`` section of
        ``config.yaml`` must keep the same key names.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 100, hidden_dim: int = 128,
                 num_layers: int = 1, rnn_type: str = "lstm", bidirectional: bool = True,
                 dropout: float = 0.3, embedding_dropout: float = 0.1,
                 pooling: str = "last", num_classes: int = 2,
                 padding_idx: int = PAD_ID,
                 embedding_weights: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.padding_idx = int(padding_idx)
        self.pooling = pooling
        self.bidirectional = bool(bidirectional)
        self.num_directions = 2 if bidirectional else 1
        self.rnn_type = str(rnn_type).lower()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

        # ---- Embedding: random init, or load a pretrained matrix --------------
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.padding_idx)
        if embedding_weights is not None:
            weights = torch.as_tensor(embedding_weights, dtype=torch.float32)
            if tuple(weights.shape) != (int(vocab_size), int(embedding_dim)):
                raise ValueError(
                    f"embedding_weights shape {tuple(weights.shape)} does not match "
                    f"(vocab_size={vocab_size}, embedding_dim={embedding_dim})"
                )
            with torch.no_grad():
                self.embedding.weight.copy_(weights)
                self.embedding.weight[self.padding_idx].zero_()
        else:
            # A small uniform distribution is more stable than the default N(0,1):
            # 100-dim dot products have too much variance and saturate the LSTM immediately
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].zero_()

        self.embedding_dropout = nn.Dropout(embedding_dropout)

        # ---- Recurrent layer: LSTM and GRU differ only in this class ----------
        if self.rnn_type not in ("lstm", "gru"):
            raise ValueError(f"rnn_type must be 'lstm' or 'gru', got {rnn_type!r}")
        rnn_class = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_class(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            # PyTorch only allows inter-layer dropout when num_layers > 1
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * self.num_directions, num_classes)

    # ------------------------------------------------------------------ forward
    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Naming note: ``tokens`` here means *vocabulary ids of the input words*,
        i.e. the ``(B, T)`` integer tensor produced by ``Vocabulary.encode``.
        This code handles no credentials, API keys or access tokens of any kind;
        the name is the standard NLP sense of the word (a token = one word piece
        mapped to its index in the vocabulary).

        Args:
            tokens: ``(B, T)`` int64, 0 = ``<pad>``.
            lengths: ``(B,)`` int64, true length of each sequence. Must be a CPU
                tensor (a hard requirement of ``pack_padded_sequence``).

        Returns:
            Logits of shape ``(B, num_classes)`` (before softmax).
        """
        embedded = self.embedding_dropout(self.embedding(tokens))

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.to("cpu"), batch_first=True, enforce_sorted=False
        )
        packed_output, state = self.rnn(packed)

        if self.pooling == "last":
            # ★ Key difference: nn.LSTM returns (h_n, c_n) as its second value,
            #   while nn.GRU returns only h_n. Indexing an LSTM result with
            #   [-2]/[-1] therefore mixes the hidden state with the cell state,
            #   and the concatenation has shape (num_layers*num_directions, B, 2H)
            #   One of the most common implementation bugs.
            hidden = state[0] if self.rnn_type == "lstm" else state
            # hidden: (num_layers * num_directions, B, H)
            # The last layer's forward direction is [-2], the backward one is [-1]
            if self.bidirectional:
                features = torch.cat([hidden[-2], hidden[-1]], dim=-1)
            else:
                features = hidden[-1]
        elif self.pooling == "max":
            padded_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=tokens.size(1)
            )
            mask = (tokens != self.padding_idx).unsqueeze(-1)          # (B, T, 1)
            # Use -1e4 instead of -inf: under FP16, -inf in max produces NaN
            features = padded_output.masked_fill(~mask, -1e4).max(dim=1).values
        else:  # pragma: no cover - already rejected by config validation
            raise ValueError(f"unknown pooling: {self.pooling!r}")

        return self.classifier(self.dropout(features))

    # -------------------------------------------------------------- utilities
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return count_parameters(self)

    def freeze_embeddings(self) -> None:
        """Freeze the embedding layer (used for the "pretrained + frozen" ablation)."""
        self.embedding.weight.requires_grad_(False)


def count_parameters(model: nn.Module) -> int:
    """Count the trainable parameters of any module."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


# ---------------------------------------------------------------------------
# checkpoint I/O
# ---------------------------------------------------------------------------
def load_checkpoint(path: Path, device: Optional[torch.device] = None
                    ) -> Tuple[RecurrentClassifier, Vocabulary, Dict[str, Any]]:
    """Rebuild the model and the vocabulary from a self-contained checkpoint.

    Args:
        path: ``best_model.pt`` or ``last_full.pt``.
        device: Target device; ``None`` means CPU (moving it elsewhere is left to the caller).

    Returns:
        ``(model, vocab, payload)``; ``model`` is already in ``eval()`` mode.

    Note:
        The checkpoint stores ``vocab`` / ``vocab_meta`` / ``vocab_size`` /
        ``model`` / ``model_state`` / ``max_len``, so inference needs only this
        single ``.pt`` file, with no separate lookup of a vocabulary or config.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = RecurrentClassifier(vocab_size=int(payload["vocab_size"]), **payload["model"])
    model.load_state_dict(payload["model_state"])
    model.eval()
    if device is not None:
        model.to(device)
    vocab = Vocabulary(payload["vocab"], payload.get("vocab_meta"))
    return model, vocab, payload


@torch.no_grad()
def predict_texts(model: RecurrentClassifier, vocab: Vocabulary, texts: Sequence[str],
                  max_len: int = 256, device: Optional[torch.device] = None,
                  batch_size: int = 32) -> List[Dict[str, Any]]:
    """Predict sentiment for arbitrary texts.

    Args:
        model: Loaded model (in ``eval()`` mode).
        vocab: Vocabulary that belongs to this model.
        texts: Texts to predict.
        max_len: **Must match training**, otherwise the train/inference input distributions differ.
        device: Device; ``None`` means CPU.
        batch_size: Inference batch size.

    Returns:
        One dict per text, of the form ``{"text", "label", "sentiment", "prob_positive"}``.

    Note:
        The probabilities are not temperature-calibrated and must not be read as
        reliable confidence scores; they are only suitable for ranking or error
        analysis. The report gives the Brier and ECE figures for the calibrated
        evaluation of the frozen models.
    """
    device = device if device is not None else torch.device("cpu")
    model.eval().to(device)
    results: List[Dict[str, Any]] = []

    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start:start + batch_size])
        sequences = [torch.tensor(vocab.encode(text, max_len), dtype=torch.long) for text in chunk]
        lengths = torch.tensor([int(sequence.numel()) for sequence in sequences], dtype=torch.long)
        tokens = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=PAD_ID).to(device)

        logits = model(tokens, lengths)
        probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        for text, row in zip(chunk, probabilities):
            label = int(row.argmax())
            results.append({
                "text": text,
                "label": label,
                "sentiment": "positive" if label == 1 else "negative",
                "prob_positive": float(row[1]),
            })
    return results
