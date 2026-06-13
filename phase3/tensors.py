import torch
import numpy as np

print("--- 1. Creating Tensors ---")
t1 = torch.tensor([1, 2, 3])
print("1D tensor:", t1)

t2 = torch.tensor([[1, 2], [3, 4]])
print("2D tensor:\n", t2)

t3 = torch.zeros(2, 3)
print("Zeros tensor:\n", t3)

t4 = torch.rand(2, 3)
print("Random tensor:\n", t4)

print("\n--- 2. Tensor properties ---")
print("Shape:", t2.shape)
print("Dtype:", t2.dtype)
print("Device:", t2.device)

print("\n--- 3. Operations (same as numpy) ---")
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print("Add:", a + b)
print("Multiply:", a * b)
print("Dot product:", torch.dot(a, b))

print("\n--- 4. Matrix multiplication ---")
M1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
M2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
print("Matmul:\n", torch.matmul(M1, M2))

print("\n--- 5. The MOST IMPORTANT feature: autograd ---")
# requires_grad=True tells PyTorch to track operations for gradient computation
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1  # y = x^2 + 2x + 1

print(f"x = {x.item()}")
print(f"y = x^2 + 2x + 1 = {y.item()}")

# Compute gradient dy/dx automatically
y.backward()
print(f"dy/dx (PyTorch computed) = {x.grad.item()}")
print(f"dy/dx (manual: 2x+2)     = {2*x.item() + 2}")

print("\n--- 6. NumPy <-> PyTorch conversion ---")
np_array = np.array([1, 2, 3])
torch_tensor = torch.from_numpy(np_array)
back_to_numpy = torch_tensor.numpy()
print("NumPy → Tensor:", torch_tensor)
print("Tensor → NumPy:", back_to_numpy)
