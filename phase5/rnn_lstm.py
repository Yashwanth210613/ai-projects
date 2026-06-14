import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)

# ── 1. RNN vs LSTM architecture comparison ────────────
print("--- RNN vs LSTM Architecture ---")

input_size = 10   # embedding dimension
hidden_size = 32  # hidden state size
num_layers = 2    # stacked layers

rnn  = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

rnn_params  = sum(p.numel() for p in rnn.parameters())
lstm_params = sum(p.numel() for p in lstm.parameters())

print(f"RNN  parameters: {rnn_params:,}")
print(f"LSTM parameters: {lstm_params:,}")
print(f"LSTM has ~4x more parameters — the 4 gates")

# ── 2. Forward pass through RNN and LSTM ──────────────
batch_size = 2
seq_len = 5

x = torch.randn(batch_size, seq_len, input_size)
print(f"\nInput shape: {x.shape}  (batch, seq_len, input_size)")

rnn_out, rnn_hidden = rnn(x)
lstm_out, (lstm_hidden, lstm_cell) = lstm(x)

print(f"RNN  output shape: {rnn_out.shape}")
print(f"LSTM output shape: {lstm_out.shape}")
print(f"LSTM has extra cell state: {lstm_cell.shape}")

# ── 3. Sentiment Analysis with LSTM ───────────────────
print("\n--- Sentiment Analysis with LSTM ---")

# Simple sentiment dataset
sentences = [
    "i love this movie",
    "this film is amazing",
    "great acting and story",
    "i hate this movie",
    "terrible film very bad",
    "worst movie ever seen",
    "pretty good film overall",
    "not bad could be better",
]
labels = [1, 1, 1, 0, 0, 0, 1, 0]  # 1=positive, 0=negative

# Build vocabulary
vocab = {"<PAD>": 0, "<UNK>": 1}
for sentence in sentences:
    for word in sentence.split():
        if word not in vocab:
            vocab[word] = len(vocab)

print(f"Vocabulary size: {len(vocab)}")

def encode_sentence(sentence, vocab, max_len=8):
    tokens = [vocab.get(w, 1) for w in sentence.split()]
    tokens = tokens[:max_len]
    tokens += [0] * (max_len - len(tokens))
    return tokens

max_len = 8
X = torch.tensor([encode_sentence(s, vocab, max_len) for s in sentences])
y = torch.tensor(labels, dtype=torch.float32)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# ── 4. LSTM Sentiment Model ────────────────────────────
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)       # indices → vectors
        lstm_out, (hidden, cell) = self.lstm(embedded)
        last_hidden = hidden[-1]           # take final hidden state
        output = self.sigmoid(self.fc(last_hidden))
        return output

model = SentimentLSTM(vocab_size=len(vocab), embed_dim=16, hidden_dim=32)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params}")

# ── 5. Train ───────────────────────────────────────────
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("\n--- Training ---")
for epoch in range(100):
    predictions = model(X).squeeze()
    loss = loss_fn(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        predicted_labels = (predictions > 0.5).float()
        accuracy = (predicted_labels == y).float().mean()
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Accuracy: {accuracy.item():.2f}")

# ── 6. Test on new sentences ───────────────────────────
print("\n--- Testing on new sentences ---")
test_sentences = [
    "i love this film",
    "terrible and boring movie",
    "not bad actually good",
]

model.eval()
with torch.no_grad():
    for sentence in test_sentences:
        encoded = torch.tensor([encode_sentence(sentence, vocab, max_len)])
        pred = model(encoded).item()
        sentiment = "Positive 😊" if pred > 0.5 else "Negative 😞"
        print(f"'{sentence}' → {sentiment} ({pred:.3f})")
