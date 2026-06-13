import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.manual_seed(42)

# ── 1. Data ──────────────────────────────────────────
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# ── 2. CNN Model ─────────────────────────────────────
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv layer 1: 1 input channel (grayscale) -> 16 output channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # Conv layer 2: 16 -> 32 channels
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # After 2 conv+pool: 28 -> 14 -> 7, so 32 channels * 7 * 7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(self.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters (CNN): {total_params:,}")
print(f"(Phase 3 Linear network had ~109,000 parameters for comparison)")

# ── 3. Training ───────────────────────────────────────
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n--- Training CNN ---")
for epoch in range(3):
    total_loss = 0
    for images, labels in train_loader:
        predictions = model(images)
        loss = loss_fn(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/3 | Avg Loss: {total_loss/len(train_loader):.4f}")

# ── 4. Evaluation ───────────────────────────────────────
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        predictions = model(images)
        predicted_classes = torch.argmax(predictions, dim=1)
        correct += (predicted_classes == labels).sum().item()
        total += labels.size(0)

print(f"\nCNN Test Accuracy: {correct/total:.4f}")
print(f"(Phase 3 Linear network got 0.9696)")
