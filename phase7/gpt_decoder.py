import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(42)

# ── 1. Causal Mask ────────────────────────────────────
def create_causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask

print("=== Causal Mask (GPT-style) ===")
mask = create_causal_mask(5)
print(mask)
print("0 = cannot attend, 1 = can attend")
print("Each position only sees current + previous tokens")

# ── 2. Decoder Block ──────────────────────────────────
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch, seq_len, d_model = x.shape

        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)

        # Masked self-attention
        Q = self.W_q(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1,2)
        K = self.W_k(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1,2)
        V = self.W_v(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1,2)

        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, -1e9)  # apply causal mask
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, V)
        attended = attended.transpose(1,2).contiguous().view(batch, seq_len, d_model)
        x = self.norm1(x + self.W_o(attended))

        # Feed-forward
        ff_out = self.ff2(self.relu(self.ff1(x)))
        x = self.norm2(x + ff_out)
        return x

# ── 3. GPT-style model ────────────────────────────────
class GPTDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(512, d_model)  # learned positional
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        positions = torch.arange(x.size(1)).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.output(x)

# Small GPT-style model
gpt = GPTDecoder(vocab_size=1000, d_model=128, num_heads=4, d_ff=512, num_layers=4)
params = sum(p.numel() for p in gpt.parameters())
print(f"\n=== GPT-style Decoder ===")
print(f"Parameters: {params:,}")

tokens = torch.randint(0, 1000, (2, 10))
output = gpt(tokens)
print(f"Input shape:  {tokens.shape}")
print(f"Output shape: {output.shape}")
print(f"Output = logits over {1000} vocab tokens for each position")

# ── 4. BERT vs GPT summary ────────────────────────────
print("\n=== BERT vs GPT ===")
print(f"{'':20} {'BERT':15} {'GPT':15}")
print(f"{'Architecture':20} {'Encoder-only':15} {'Decoder-only':15}")
print(f"{'Attention':20} {'Bidirectional':15} {'Causal (left→)':15}")
print(f"{'Training task':20} {'Masked LM':15} {'Next token pred':15}")
print(f"{'Best for':20} {'Understanding':15} {'Generation':15}")
print(f"{'Examples':20} {'BERT,RoBERTa':15} {'GPT,LLaMA':15}")
