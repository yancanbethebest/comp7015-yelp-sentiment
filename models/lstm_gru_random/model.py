# -*- coding: utf-8 -*-
"""模型定义与推理 / Model definition and inference.

================================ 架构 ================================
    tokens ──► Embedding(V, d, padding_idx=0)   ← 随机初始化，可训练
           ──► EmbeddingDropout(0.1)
           ──► pack_padded_sequence             ← PAD 不参与循环计算
           ──► Bi{LSTM|GRU}(hidden, num_layers) ← 单向/双向
           ──► [双向末状态拼接 | 掩码最大池化]
           ──► Dropout(0.3)
           ──► Linear(2*hidden, 2)

默认超参下参数量 / parameters with the default config:
    Embedding 30,000 × 100 = 3,000,000
    BiLSTM 双向             =   235,520     （BiGRU 为 176,640）
    Linear 256 → 2          =       514
    合计 BiLSTM = 3,236,034 / BiGRU = 3,177,154

================================ 给成员三的接口 ================================
    ``RecurrentClassifier`` 的 ``embedding_weights`` 参数接受任意
    ``(vocab_size, embedding_dim)`` 的张量。传入按词表对齐的 GloVe/Word2Vec 矩阵即可得到
    "预训练 embedding" 版本；循环层、池化、dropout 与分类头完全不变，
    因此"随机 vs 预训练"的对比只有**初始化来源**一个变量。

    ```python
    from models.lstm_gru_random import RecurrentClassifier, Vocabulary
    vocab = Vocabulary.load("runs/member2_bilstm_random_full_seed42/vocab.json")   # 必须复用同一份词表
    weights = build_glove_matrix(vocab, "glove.6B.100d.txt")               # (len(vocab), 100)
    model = RecurrentClassifier(vocab_size=len(vocab), embedding_weights=weights)
    ```
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

if __package__:  # 作为包导入 / imported as a package
    from .data import PAD_ID, Vocabulary
else:            # 直接以脚本方式运行 / executed as a plain script
    from data import PAD_ID, Vocabulary  # type: ignore


class RecurrentClassifier(nn.Module):
    """随机初始化 Embedding 的双向 LSTM / GRU 情感分类器。

    Args:
        vocab_size: embedding 行数，等于 ``len(Vocabulary)``（含 ``<pad>``/``<unk>``）。
        embedding_dim: 词向量维度；100 与 GloVe 6B-100d 对齐。
        hidden_dim: 每个方向的隐藏单元数。
        num_layers: 循环层数。
        rnn_type: ``"lstm"`` 或 ``"gru"``。
        bidirectional: 是否双向。
        dropout: 循环层输出 → 分类头之间的 dropout。
        embedding_dropout: embedding 输出上的 dropout（Gal & Ghahramani, 2016）。
        pooling: ``"last"``（双向末状态拼接）或 ``"max"``（掩码最大池化）。
        num_classes: 输出类别数，二分类为 2。
        padding_idx: PAD 的 id，其 embedding 行不更新。
        embedding_weights: **成员三的接口**——可选的预训练词向量矩阵。
            形状必须为 ``(vocab_size, embedding_dim)``；传入后该层以预训练值初始化，
            是否冻结由调用方通过 ``model.embedding.weight.requires_grad_(False)`` 控制。

    Note:
        checkpoint 中保存的 ``config["model"]`` 可以直接 ``**`` 展开成本类的参数，
        因此 ``config.yaml`` 的 ``model`` 段键名必须与这里保持一致。
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

        # ---- Embedding：随机初始化，或载入预训练矩阵 --------------------------
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.padding_idx)
        if embedding_weights is not None:
            weights = torch.as_tensor(embedding_weights, dtype=torch.float32)
            if tuple(weights.shape) != (int(vocab_size), int(embedding_dim)):
                raise ValueError(
                    f"embedding_weights 形状 {tuple(weights.shape)} 与 "
                    f"(vocab_size={vocab_size}, embedding_dim={embedding_dim}) 不匹配"
                )
            with torch.no_grad():
                self.embedding.weight.copy_(weights)
                self.embedding.weight[self.padding_idx].zero_()
        else:
            # 小均匀分布比默认 N(0,1) 更稳：100 维点积方差过大会让 LSTM 一开始就饱和
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].zero_()

        self.embedding_dropout = nn.Dropout(embedding_dropout)

        # ---- 循环层：LSTM 与 GRU 只差这一行的类 -------------------------------
        if self.rnn_type not in ("lstm", "gru"):
            raise ValueError(f"rnn_type 必须是 'lstm' 或 'gru'，收到 {rnn_type!r}")
        rnn_class = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_class(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            # PyTorch 只在层数 > 1 时允许层间 dropout
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * self.num_directions, num_classes)

    # ------------------------------------------------------------------ 前向
    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """前向传播 / forward pass.

        Args:
            tokens: ``(B, T)`` int64，0 = ``<pad>``。
            lengths: ``(B,)`` int64，每条序列的真实长度。必须是 CPU 张量
                （``pack_padded_sequence`` 的硬性要求）。

        Returns:
            ``(B, num_classes)`` 的 logits（未过 softmax）。
        """
        embedded = self.embedding_dropout(self.embedding(tokens))

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.to("cpu"), batch_first=True, enforce_sorted=False
        )
        packed_output, state = self.rnn(packed)

        if self.pooling == "last":
            # ★ 关键区别：nn.LSTM 的第二个返回值是 (h_n, c_n) 元组，nn.GRU 只返回 h_n。
            #   若直接对 LSTM 的返回值取 [-2]/[-1]，取到的其实是"隐状态 + 细胞状态"，
            #   拼接后形状变成 (num_layers*num_directions, B, 2H)，是最常见的一类实现 bug。
            hidden = state[0] if self.rnn_type == "lstm" else state
            # hidden: (num_layers * num_directions, B, H)
            # 最后一层的正向是 [-2]、反向是 [-1]
            if self.bidirectional:
                features = torch.cat([hidden[-2], hidden[-1]], dim=-1)
            else:
                features = hidden[-1]
        elif self.pooling == "max":
            padded_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=tokens.size(1)
            )
            mask = (tokens != self.padding_idx).unsqueeze(-1)          # (B, T, 1)
            # 用 -1e4 而不是 -inf：FP16 下 -inf 参与 max 会产生 NaN
            features = padded_output.masked_fill(~mask, -1e4).max(dim=1).values
        else:  # pragma: no cover - 配置校验已拦截
            raise ValueError(f"未知的 pooling 方式 / unknown pooling: {self.pooling!r}")

        return self.classifier(self.dropout(features))

    # -------------------------------------------------------------- 工具方法
    def count_parameters(self) -> int:
        """可训练参数量 / number of trainable parameters."""
        return count_parameters(self)

    def freeze_embeddings(self) -> None:
        """冻结 embedding 层（对比"预训练 + 冻结"时使用 / freeze for ablation)."""
        self.embedding.weight.requires_grad_(False)


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数 / count trainable parameters of any module."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


# ---------------------------------------------------------------------------
# checkpoint 读写 / checkpoint I/O
# ---------------------------------------------------------------------------
def load_checkpoint(path: Path, device: Optional[torch.device] = None
                    ) -> Tuple[RecurrentClassifier, Vocabulary, Dict[str, Any]]:
    """从 checkpoint 还原模型与词表（checkpoint 是自包含的）。

    Rebuild the model and the vocabulary from a self-contained checkpoint.

    Args:
        path: ``best_model.pt`` 或 ``last_full.pt``。
        device: 目标设备；``None`` 表示 CPU（迁移到别处由调用方决定）。

    Returns:
        ``(model, vocab, payload)``；``model`` 已切到 ``eval()`` 模式。

    Note:
        checkpoint 里保存了 ``vocab`` / ``vocab_size`` / ``model`` / ``max_len``，
        所以推理只需要这一个 ``.pt`` 文件，不需要额外找词表或配置。
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
    """对任意文本做情感预测 / predict sentiment for arbitrary texts.

    Args:
        model: 已加载的模型（在 ``eval()`` 模式）。
        vocab: 与该模型配套的词表。
        texts: 待预测文本列表。
        max_len: **必须与训练时一致**，否则训练/推理的输入分布不同。
        device: 设备；``None`` 表示 CPU。
        batch_size: 推理批大小。

    Returns:
        每条形如 ``{"text", "label", "sentiment", "prob_positive"}`` 的字典。

    Note:
        输出概率未做温度校准，不应作为可靠的置信度使用（与仓库
        ``NEXT_STEPS.md`` 的说明一致），仅适合排序或错误分析。
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
