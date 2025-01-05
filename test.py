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
M = (C1+D1@W)@X@X.T@(A2+W.T@C2)-W

loss = x.T @ M.T @ M @ x

graph = make_dot(loss, params={"C1": C1, "A2":A2, "C2": C2, "D1":D1, "X":X, "W":W, "x":x})
graph.render("./computation_graph", format="png")  # PNG 파일로 저장
graph.view()  # 그래프 시각화