import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)

# ── 1. Generate data ──────────────────────────────────
X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=3, n_redundant=1,
                           random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# ── 2. Define the model ───────────────────────────────
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

model = SimpleNN()
print(f"\nModel architecture:\n{model}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal trainable parameters: {total_params}")

# ── 3. Loss and optimizer ─────────────────────────────
loss_fn = nn.BCELoss()  # Binary Cross Entropy — for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ── 4. Training loop ──────────────────────────────────
epochs = 100
print("\n--- Training ---")
for epoch in range(epochs):
    # Forward pass
    predictions = model(X_train)
    loss = loss_fn(predictions, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

# ── 5. Evaluation ──────────────────────────────────────
model.eval()  # switch to evaluation mode
with torch.no_grad():  # don't track gradients for evaluation
    test_predictions = model(X_test)
    test_predictions_binary = (test_predictions > 0.5).float()
    accuracy = (test_predictions_binary == y_test).float().mean()

print(f"\nFinal Test Accuracy: {accuracy.item():.3f}")

# ── 6. Predict on a single sample ─────────────────────
sample = X_test[0:1]
with torch.no_grad():
    pred = model(sample)
print(f"\nSample prediction probability: {pred.item():.3f}")
print(f"Actual label: {y_test[0].item()}")
