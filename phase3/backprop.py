import torch
import torch.nn as nn

torch.manual_seed(42)

# ── Demonstrate gradient flow through layers ──────────
class DeepNet(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(10)])
        self.activation = activation

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x

x = torch.randn(1, 10)
y_target = torch.randn(1, 10)
loss_fn = nn.MSELoss()

print("--- Gradient magnitude at each layer ---\n")

for name, activation in [("Sigmoid", nn.Sigmoid()), ("ReLU", nn.ReLU())]:
    print(f"Activation: {name}")
    model = DeepNet(activation)

    output = model(x)
    loss = loss_fn(output, y_target)
    loss.backward()

    # Check gradient magnitude at first and last layer
    first_layer_grad = model.layers[0].weight.grad.abs().mean().item()
    last_layer_grad = model.layers[-1].weight.grad.abs().mean().item()

    print(f"  First layer gradient magnitude: {first_layer_grad:.8f}")
    print(f"  Last layer gradient magnitude:  {last_layer_grad:.8f}")
    print(f"  Ratio (first/last): {first_layer_grad/last_layer_grad:.6f}\n")
