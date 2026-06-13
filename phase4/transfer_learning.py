import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

torch.manual_seed(42)

# ── 1. Load pre-trained ResNet18 ──────────────────────
model = models.resnet18(weights='IMAGENET1K_V1')
print("ResNet18 loaded with ImageNet weights")
print(f"\nOriginal final layer: {model.fc}")

# ── 2. Freeze all layers ───────────────────────────────
for param in model.parameters():
    param.requires_grad = False

frozen_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters (all frozen): {frozen_params:,}")

# ── 3. Replace the final layer ─────────────────────────
# ResNet18's original final layer outputs 1000 classes (ImageNet)
# We replace it for OUR task — let's say 3 classes (e.g. healthy/diseased/severe)
num_classes = 3
model.fc = nn.Linear(model.fc.in_features, num_classes)

print(f"\nNew final layer: {model.fc}")

# Count trainable parameters now
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTrainable parameters (only new layer): {trainable_params:,}")
print(f"As percentage of total: {trainable_params/frozen_params*100:.4f}%")

# ── 4. This is the model ready for fine-tuning ────────
print("\n--- Model is ready ---")
print("Only the new final layer will be trained.")
print("All other 'knowledge' from ImageNet is preserved and reused.")

# ── 5. Demonstrate forward pass with dummy data ───────
dummy_image = torch.randn(1, 3, 224, 224)  # batch=1, RGB, 224x224
output = model(dummy_image)
print(f"\nDummy input shape: {dummy_image.shape}")
print(f"Output shape: {output.shape}")  # should be [1, 3] for 3 classes
print(f"Output values (raw logits): {output}")
