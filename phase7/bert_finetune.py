import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.optim as optim

torch.manual_seed(42)

# ── 1. Load real BERT tokenizer ───────────────────────
print("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print(f"Vocabulary size: {tokenizer.vocab_size:,}")

# ── 2. Sentiment dataset ──────────────────────────────
texts = [
    "This movie is absolutely wonderful",
    "Great film, loved every minute of it",
    "Amazing story with brilliant acting",
    "One of the best films ever made",
    "I really enjoyed this movie a lot",
    "Fantastic direction and cinematography",
    "This was terrible and extremely boring",
    "Worst film I have ever seen in my life",
    "Complete waste of time and money",
    "Awful acting and a very poor story",
    "I hated every single minute of this",
    "Disappointing and very poorly made",
]
labels = [1,1,1,1,1,1,0,0,0,0,0,0]

# ── 3. Dataset class ───────────────────────────────────
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.data = []
        for text, label in zip(texts, labels):
            encoding = tokenizer(
                text,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            self.data.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'label': torch.tensor(label, dtype=torch.float32)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

dataset = SentimentDataset(texts, labels, tokenizer)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
test_loader = DataLoader(test_set, batch_size=4)
print(f"Train size: {train_size}, Test size: {test_size}")

# ── 4. BERT classifier model ──────────────────────────
class BERTSentimentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_token = outputs.last_hidden_state[:, 0, :]
        dropped = self.dropout(cls_token)
        return self.sigmoid(self.classifier(dropped))

print("\nLoading pre-trained BERT...")
model = BERTSentimentClassifier()

# ── 5. Freeze most of BERT, unfreeze last 2 layers ────
for param in model.bert.parameters():
    param.requires_grad = False

for param in model.bert.encoder.layer[-2:].parameters():
    param.requires_grad = True

for param in model.bert.pooler.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Total parameters:     {total:,}")
print(f"Trainable parameters: {trainable:,}")
print(f"Frozen parameters:    {total - trainable:,}")

# ── 6. Train ───────────────────────────────────────────
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

print("\n--- Training (last 2 BERT layers + classifier) ---")
for epoch in range(20):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels_batch = batch['label']

        pred = model(input_ids, attention_mask).squeeze(-1)
        loss = loss_fn(pred, labels_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 4 == 0:
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in test_loader:
                pred = model(
                    batch['input_ids'],
                    batch['attention_mask']
                ).squeeze(-1)
                correct += (
                    (pred > 0.5).float() == batch['label']
                ).sum().item()
        print(f"Epoch {epoch:2d} | Loss: {total_loss:.4f} | "
              f"Test Acc: {correct/test_size:.2f}")

# ── 7. Test on new sentences ──────────────────────────
print("\n--- Testing on new reviews ---")
new_reviews = [
    "This film is a masterpiece of modern cinema",
    "Absolutely dreadful and painfully slow movie",
    "Not bad but could have been much better",
    "The acting was superb and story was gripping",
]

model.eval()
with torch.no_grad():
    for review in new_reviews:
        encoding = tokenizer(
            review,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        pred = model(
            encoding['input_ids'],
            encoding['attention_mask']
        ).item()
        sentiment = "Positive 😊" if pred > 0.5 else "Negative 😞"
        print(f"'{review[:45]}' → {sentiment} ({pred:.3f})")
