import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ── 1. Generate fake data ─────────────────────────────
# True relationship: y = 3x + 5 (we want model to learn this)
X = np.random.randn(100)
y = 3 * X + 5 + np.random.randn(100) * 0.5  # add noise

# ── 2. Initialize weights randomly ───────────────────
w = 0.0  # slope
b = 0.0  # intercept
lr = 0.01  # learning rate
epochs = 100  # how many times we loop through data

losses = []

# ── 3. Gradient Descent loop ──────────────────────────
for epoch in range(epochs):
    # Forward pass — make predictions
    y_pred = w * X + b

    # Compute loss (Mean Squared Error)
    loss = np.mean((y_pred - y) ** 2)
    losses.append(loss)

    # Compute gradients
    dw = np.mean(2 * (y_pred - y) * X)  # derivative w.r.t w
    db = np.mean(2 * (y_pred - y))      # derivative w.r.t b

    # Update weights — the core gradient descent step
    w = w - lr * dw
    b = b - lr * db

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | w: {w:.3f} | b: {b:.3f}")

print(f"\nFinal learned: w={w:.3f}, b={b:.3f}")
print(f"True values:   w=3.000, b=5.000")

# ── 4. Learning rate comparison ──────────────────────
def train(lr, epochs=50):
    w, b = 0.0, 0.0
    losses = []
    for _ in range(epochs):
        y_pred = w * X + b
        loss = np.mean((y_pred - y) ** 2)
        losses.append(loss)
        dw = np.mean(2 * (y_pred - y) * X)
        db = np.mean(2 * (y_pred - y))
        w -= lr * dw
        b -= lr * db
    return losses

losses_small = train(lr=0.001)
losses_good  = train(lr=0.01)
losses_large = train(lr=0.9)

print("\nFinal loss comparison:")
print(f"lr=0.001 (too small): {losses_small[-1]:.4f}")
print(f"lr=0.01  (just right): {losses_good[-1]:.4f}")
print(f"lr=0.9   (too large): {losses_large[-1]:.4f}")