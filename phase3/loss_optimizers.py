import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)

# ── Data ───────────────────────────────────────────────
X, y = make_classification(n_samples=1000, n_features=6, n_classes=3,
                           n_informative=4,n_redundant=0, n_clusters_per_class=1,
                           random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)  # LongTensor for CrossEntropyLoss
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

print(f"Number of classes: {len(torch.unique(y_train))}")
print(f"y_train sample: {y_train[:5]}")  # class labels: 0, 1, 2

# ── Model for multi-class classification ──────────────
class MultiClassNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(6, 16)
        self.layer2 = nn.Linear(16, 3)  # 3 outputs for 3 classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)  # raw logits — NO softmax here
        return x

# ── Compare SGD vs Adam ────────────────────────────────
print("\n--- Comparing Optimizers (same model, same epochs) ---")

for opt_name, opt_class, lr in [("SGD", optim.SGD, 0.01), ("Adam", optim.Adam, 0.01)]:
    torch.manual_seed(42)  # same starting weights for fair comparison
    model = MultiClassNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = opt_class(model.parameters(), lr=lr)

    losses = []
    for epoch in range(50):
        predictions = model(X_train)
        loss = loss_fn(predictions, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Final test accuracy
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test)
        predicted_classes = torch.argmax(test_preds, dim=1)
        accuracy = (predicted_classes == y_test).float().mean()

    print(f"\n{opt_name}:")
    print(f"  Loss at epoch 0:  {losses[0]:.4f}")
    print(f"  Loss at epoch 25: {losses[25]:.4f}")
    print(f"  Loss at epoch 49: {losses[49]:.4f}")
    print(f"  Test Accuracy:    {accuracy.item():.3f}")
