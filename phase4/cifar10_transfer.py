import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

torch.manual_seed(42)

# ── 1. Transforms — resize to 224x224 for ResNet ──────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ── 2. Load CIFAR-10 — using small subset for speed ───
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Use a SUBSET for faster training on CPU (2000 train, 500 test)
train_subset = torch.utils.data.Subset(train_data, range(2000))
test_subset = torch.utils.data.Subset(test_data, range(500))

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

classes = train_data.classes
print(f"Classes: {classes}")
print(f"Training subset: {len(train_subset)} images")
print(f"Test subset: {len(test_subset)} images")

# ── 3. Load pre-trained ResNet18, freeze, replace final layer ──
model = models.resnet18(weights='IMAGENET1K_V1')

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 10)  # 10 CIFAR classes

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTrainable parameters: {trainable:,}")

# ── 4. Train only the final layer ──────────────────────
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # only optimize fc layer

print("\n--- Training (this will take a few minutes on CPU) ---")
model.train()
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

# ── 5. Evaluate ──────────────────────────────────────────
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        predictions = model(images)
        predicted_classes = torch.argmax(predictions, dim=1)
        correct += (predicted_classes == labels).sum().item()
        total += labels.size(0)

accuracy = correct/total
print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"(Random guessing baseline for 10 classes: 0.10)")

# ── 6. Test on a few individual images ────────────────
print("\n--- Sample Predictions ---")
model.eval()
with torch.no_grad():
    images, labels = next(iter(test_loader))
    predictions = model(images[:5])
    predicted_classes = torch.argmax(predictions, dim=1)

    for i in range(5):
        actual = classes[labels[i]]
        predicted = classes[predicted_classes[i]]
        status = "✓" if actual == predicted else "✗"
        print(f"{status} Actual: {actual:12s} | Predicted: {predicted}")

# ── 7. Save the model ────────────────────────────────────
torch.save(model.state_dict(), 'cifar10_resnet.pth')
print("\nModel saved as cifar10_resnet.pth")
