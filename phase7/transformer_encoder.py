import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

torch.manual_seed(42)

# ── 1. Positional Encoding ────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ── 2. Multi-Head Attention ───────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch, seq_len, d_model = x.shape

        Q = self.W_q(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1,2)
        K = self.W_k(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1,2)
        V = self.W_v(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1,2)

        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, V)
        attended = attended.transpose(1,2).contiguous().view(batch, seq_len, d_model)
        return self.W_o(attended)

# ── 3. Feed-Forward Network ───────────────────────────
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# ── 4. Single Encoder Block ───────────────────────────
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention + residual connection
        attended = self.attention(x)
        x = self.norm1(x + self.dropout(attended))   # Add & Norm

        # Feed-forward + residual connection
        fed = self.ff(x)
        x = self.norm2(x + self.dropout(fed))        # Add & Norm
        return x

# ── 5. Full Transformer Encoder ───────────────────────
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.embedding(x)        # token indices → embeddings
        x = self.pos_encoding(x)     # add position info
        for layer in self.layers:
            x = layer(x)             # stack of encoder blocks
        return self.norm(x)

# ── 6. Build and analyze ──────────────────────────────
print("=== Transformer Encoder ===\n")

# BERT-base scale
vocab_size = 30522   # BERT's actual vocabulary size
d_model = 768        # embedding dimension
num_heads = 12       # attention heads
d_ff = 3072          # feed-forward inner dimension
num_layers = 12      # stacked encoder blocks

encoder = TransformerEncoder(vocab_size, d_model, num_heads, d_ff, num_layers)
total_params = sum(p.numel() for p in encoder.parameters())
print(f"Architecture: BERT-base scale")
print(f"Vocabulary:   {vocab_size:,} tokens")
print(f"d_model:      {d_model} (embedding dimension)")
print(f"Heads:        {num_heads}")
print(f"Layers:       {num_layers}")
print(f"Total params: {total_params:,}")
print(f"BERT-base actual: ~110,000,000 — close!")

# ── 7. Forward pass ───────────────────────────────────
print("\n=== Forward Pass ===")
batch_size = 2
seq_len = 16

tokens = torch.randint(1, vocab_size, (batch_size, seq_len))
print(f"Input tokens shape:  {tokens.shape}")

with torch.no_grad():
    output = encoder(tokens)
print(f"Output shape:        {output.shape}")
print(f"(batch={batch_size}, seq_len={seq_len}, d_model={d_model})")
print(f"\nEvery token now has a {d_model}-dim context-aware representation")
print(f"'bank' in 'river bank' ≠ 'bank' in 'money bank' — different vectors!")

# ── 8. Positional encoding visualization ──────────────
print("\n=== Positional Encoding ===")
pe = PositionalEncoding(d_model=8, max_seq_len=5)
dummy = torch.zeros(1, 5, 8)
encoded = pe(dummy)
print("Positional encodings for 5 positions (d_model=8):")
print(encoded.squeeze(0).numpy().round(3))
print("\nEach position has a unique pattern — model uses this to understand word order")
