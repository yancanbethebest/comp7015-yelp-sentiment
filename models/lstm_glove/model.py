import torch
from torch import nn

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


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = BiLSTM(len(checkpoint["vocab"]) + 2, checkpoint["embedding_dim"], checkpoint["hidden_dim"], checkpoint["dropout"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint
