import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(42)

# ── 1. Scaled Dot-Product Attention from scratch ──────
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]

    # Step 1: Compute similarity scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    print(f"Attention scores shape: {scores.shape}")
    print(f"Raw scores:\n{scores.detach().numpy().round(3)}")

    # Step 2: Softmax → attention weights
    attention_weights = F.softmax(scores, dim=-1)
    print(f"\nAttention weights (sum to 1):\n{attention_weights.detach().numpy().round(3)}")
    print(f"Weights sum: {attention_weights.sum(dim=-1).detach().numpy().round(3)}")

    # Step 3: Weighted sum of values
    output = torch.matmul(attention_weights, V)
    return output, attention_weights

print("=== Scaled Dot-Product Attention ===")
seq_len = 4   # 4 tokens
d_k = 8       # dimension

Q = torch.randn(seq_len, d_k)
K = torch.randn(seq_len, d_k)
V = torch.randn(seq_len, d_k)

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"\nOutput shape: {output.shape}")

# ── 2. Visualize attention weights ────────────────────
print("\n=== Attention Visualization ===")
words = ["The", "cat", "sat", "mat"]

# Simulate trained attention weights for "sat"
# "sat" should attend to "cat" (subject) and "mat" (object)
simulated_weights = torch.tensor([
    [0.1, 0.4, 0.4, 0.1],  # "The" attends to...
    [0.2, 0.5, 0.2, 0.1],  # "cat" attends to...
    [0.1, 0.5, 0.2, 0.2],  # "sat" attends to... (high on "cat")
    [0.1, 0.2, 0.3, 0.4],  # "mat" attends to...
])

print("Attention weights (rows=query word, cols=key word):")
print(f"{'':8}", end="")
for w in words:
    print(f"{w:8}", end="")
print()
for i, word in enumerate(words):
    print(f"{word:8}", end="")
    for j in range(len(words)):
        print(f"{simulated_weights[i][j].item():.2f}    ", end="")
    print()

print("\n'sat' attends most to:", words[simulated_weights[2].argmax()])

# ── 3. Multi-Head Attention concept ───────────────────
print("\n=== Multi-Head Attention ===")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)

        scores = torch.matmul(Q, K.transpose(-2,-1)) / (self.d_k ** 0.5)
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, V)

        attended = attended.transpose(1,2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(attended)
        return output, weights

d_model = 64
num_heads = 8
mha = MultiHeadAttention(d_model, num_heads)

x = torch.randn(2, 10, d_model)  # batch=2, seq_len=10, d_model=64
output, weights = mha(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"(batch=2, heads=8, seq=10, seq=10)")

params = sum(p.numel() for p in mha.parameters())
print(f"\nMulti-Head Attention parameters: {params:,}")
print(f"num_heads={num_heads}, each head sees d_k={d_model//num_heads} dims")

# ── 4. Why multi-head? ────────────────────────────────
print("\n=== Why Multiple Heads? ===")
print("Each head learns to attend to DIFFERENT relationships:")
print("Head 1 → syntactic relationships (subject-verb)")
print("Head 2 → semantic relationships (word meanings)")
print("Head 3 → positional relationships (nearby words)")
print("Head 4 → co-reference (pronouns → nouns they refer to)")
print("...")
print("All heads run in PARALLEL → efficient on GPUs")
