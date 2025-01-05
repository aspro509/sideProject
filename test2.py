import torch
from torchviz import make_dot

# Set random seed for reproducibility
torch.manual_seed(0)

# Define matrix sizes
d1 = 4
d2 = 3
N = 5

# Initialize matrices
C1 = torch.randn(d2, d1, requires_grad=True)
A2 = torch.randn(d1, d1, requires_grad=True)
C2 = torch.randn(d2, d1, requires_grad=True)
D1 = torch.randn(d2, d2, requires_grad=True)

X = torch.randn(d1, N)
W = torch.randn(d2, d1)
x = torch.randn(d1, 1)


# Compute H
M = (C1 + D1 @ W) @ X @ X.T @ (A2 + W.T @ C2) - W

loss = x.T @ M.T @ M @ x

loss.backward()

manual_aL_aC2 = 2 * W @ X @ X.T @ (C1 + D1 @ W).T @ M @ x @ x.T
manual_aL_aA2 = 2 * X @ X.T @ (C1 + D1 @ W).T @ M @ x @ x.T

manual_aL_aC1 = 2 * M @ x @ x.T @ (A2 + W.T @ C2).T @ X @ X.T
manual_aL_aD1 = 2 * M @ x @ x.T @ (A2 + W.T @ C2).T @ X @ X.T @ W.T
print("auto aL/aC2\n", C2.grad)
print("manual aL/aC2\n", manual_aL_aC2)

print("auto aL/aC1\n", C1.grad)
print("manual aL/aC1\n", manual_aL_aC1)

print("auto aL/aA2\n", A2.grad)
print("manual aL/aA2\n", manual_aL_aA2)

print("auto aL/aD1\n", D1.grad)
print("manual aL/aD1\n", manual_aL_aD1)
