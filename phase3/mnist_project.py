import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.manual_seed(42)

# ── 1. Load MNIST dataset ──────────────────────────────
transform = transforms.Compose([transforms.ToTensor()])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")
print(f"Image shape: {train_data[0][0].shape}")  # [1, 28, 28]

# ── 2. Model ────────────────────────────────────────────
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # 28x28 image -> 784 vector
        self.layer1 = nn.Linear(28*28, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 10)  # 10 digits (0-9)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

model = MNISTNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ── 3. Training loop ────────────────────────────────────
print("\n--- Training ---")
epochs = 3
for epoch in range(epochs):
    total_loss = 0
    for images, labels in train_loader:
        predictions = model(images)
        loss = loss_fn(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss/len(train_loader):.4f}")

# ── 4. Evaluation ────────────────────────────────────────
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        predictions = model(images)
        predicted_classes = torch.argmax(predictions, dim=1)
        correct += (predicted_classes == labels).sum().item()
        total += labels.size(0)

print(f"\nTest Accuracy: {correct/total:.4f}")

# ── 5. Save the model ─────────────────────────────────────
torch.save(model.state_dict(), 'mnist_model.pth')
print("\nModel saved as mnist_model.pth")
