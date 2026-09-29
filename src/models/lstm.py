import torch
import torch.nn as nn
from torchcrf import CRF


class AspectExtractor(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_dim, output_dim, dropout, tag2idx):
        super().__init__()
        self.tag2idx = tag2idx
        self.idx2tag = {v: k for k, v in tag2idx.items()}

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(
            embedding_size, hidden_dim // 2,
            num_layers=1, bidirectional=True, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.crf = CRF(output_dim, batch_first=True)

    def forward(self, input_ids, attention_mask=None, labels=None):
        embedded = self.dropout(self.embedding(input_ids))
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(self.layer_norm(lstm_out))

        att_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = lstm_out * att_weights + lstm_out
        emissions = self.fc(self.dropout(context))

        loss = None
        if labels is not None:
            crf_mask = (labels != -100).bool()
            clean_labels = torch.where(labels != -100, labels, torch.zeros_like(labels))
            log_likelihood = self.crf(
                emissions, clean_labels, mask=crf_mask, reduction="token_mean"
            )
            loss = -log_likelihood

        return {"loss": loss, "logits": emissions}


class SentinentLSTM(nn.Module):
    # Class name intentionally preserved from the original notebook.
    def __init__(self, vocab_size, embedding_size, hidden_dim, output_dim, dropout, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.aspect_proj = nn.Linear(embedding_size, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text_ids, aspect_ids, text_mask=None, labels=None):
        text_emb = self.dropout(self.embedding(text_ids))
        lstm_out, _ = self.lstm(text_emb)

        aspect_emb = self.dropout(self.embedding(aspect_ids))
        aspect_query = self.aspect_proj(aspect_emb)

        text_emb = self.dropout(text_emb)
        aspect_emb = self.dropout(aspect_emb)

        key_padding_mask = ~text_mask if text_mask is not None else None
        attn_output, _ = self.attention(
            query=aspect_query,
            key=lstm_out,
            value=lstm_out,
            key_padding_mask=key_padding_mask,
        )

        context = self.layer_norm(attn_output + aspect_query)
        context_vec = context.mean(dim=1)
        aspect_vec = aspect_query.mean(dim=1)
        features = torch.cat([context_vec, aspect_vec], dim=-1)
        logits = self.fc(self.dropout(features))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}
