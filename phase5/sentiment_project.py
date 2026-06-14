import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re

torch.manual_seed(42)

# ── 1. Dataset ─────────────────────────────────────────
reviews = [
    ("this movie is absolutely wonderful", 1),
    ("great film loved every minute", 1),
    ("amazing story and brilliant acting", 1),
    ("one of the best films ever made", 1),
    ("i really enjoyed this movie", 1),
    ("fantastic cinematography and direction", 1),
    ("this was terrible and boring", 0),
    ("worst film i have ever seen", 0),
    ("complete waste of time and money", 0),
    ("awful acting and poor story", 0),
    ("i hated every minute of this", 0),
    ("disappointing and poorly made film", 0),
]

# ── 2. Preprocessing ───────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()

# ── 3. Build vocabulary ────────────────────────────────
vocab = {"<PAD>": 0, "<UNK>": 1}
for text, _ in reviews:
    for word in preprocess(text):
        if word not in vocab:
            vocab[word] = len(vocab)

print(f"Vocabulary size: {len(vocab)}")

# ── 4. Dataset class ───────────────────────────────────
class SentimentDataset(Dataset):
    def __init__(self, reviews, vocab, max_len=10):
        self.data = []
        for text, label in reviews:
            tokens = preprocess(text)
            encoded = [vocab.get(w, 1) for w in tokens][:max_len]
            encoded += [0] * (max_len - len(encoded))
            self.data.append((torch.tensor(encoded), torch.tensor(label, dtype=torch.float32)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

dataset = SentimentDataset(reviews, vocab)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
test_loader = DataLoader(test_set, batch_size=4)

# ── 5. Model with Attention ────────────────────────────
class AttentionSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)

        # Attention: score each hidden state
        attn_scores = self.attention(lstm_out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Weighted sum of hidden states
        context = (lstm_out * attn_weights.unsqueeze(-1)).sum(dim=1)

        output = self.sigmoid(self.fc(context))
        return output

model = AttentionSentiment(vocab_size=len(vocab), embed_dim=32, hidden_dim=64)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# ── 6. Train ───────────────────────────────────────────
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n--- Training ---")
for epoch in range(50):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        pred = model(X_batch).squeeze(-1)
        loss = loss_fn(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        model.eval()
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                pred = model(X_batch).squeeze(-1)
                correct += ((pred > 0.5).float() == y_batch).sum().item()
        print(f"Epoch {epoch:3d} | Loss: {total_loss:.4f} | Test Acc: {correct/test_size:.2f}")

# ── 7. Final test ──────────────────────────────────────
print("\n--- Final Predictions ---")
test_reviews = [
    "this film is absolutely brilliant",
    "terrible waste of time",
    "pretty good movie overall",
]

model.eval()
with torch.no_grad():
    for review in test_reviews:
        tokens = preprocess(review)
        encoded = [vocab.get(w, 1) for w in tokens][:10]
        encoded += [0] * (10 - len(encoded))
        x = torch.tensor([encoded])
        pred = model(x).item()
        sentiment = "Positive 😊" if pred > 0.5 else "Negative 😞"
        print(f"'{review}' → {sentiment} ({pred:.3f})")
