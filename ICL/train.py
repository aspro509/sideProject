import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from linear_self_attention import LSALayer
from linear_self_attention_GD import LSA_GD
import generate_data

import matplotlib.pyplot as plt


def custom_mse_loss(predictions, targets):
    return torch.mean((predictions - targets) ** 2)


d_in = 10
d_out = 1
d_token = d_in + d_out

B = 2048
N = 10
alpha = 1.0
val_B = 10**4

max_grad_norm = 10

model = LSALayer(d_token, d_token, d_token, d_token)

lr = 0.001
optimizer = optim.Adam(
    model.parameters(), lr=lr
)  # SGD with defort setting fails to decrease loss

num_steps = 1500


lsa_gd = LSA_GD()

train_tokens, batch_W = generate_data.generate_linear_regression_batch(
    B=val_B,
    N=N,
    d_in=d_in,
    d_out=d_out,
    noise_std=0.0,
    x_low=-alpha,
    x_high=alpha,
    random_seed=0,
)
test_tokens = generate_data.generate_test_token(
    B=val_B, d_in=d_in, d_out=d_out, x_low=-alpha, x_high=alpha
)

val_data = torch.concatenate([train_tokens, test_tokens], axis=1)
val_target = torch.einsum("bnd, bdy -> bny", test_tokens[:, :, :d_in], batch_W)


losses = []
val_losses = []
for step in range(num_steps):

    model.train()
    train_tokens, batch_W = generate_data.generate_linear_regression_batch(
        B=B, N=N, d_in=d_in, d_out=d_out, noise_std=0.0, x_low=-alpha, x_high=alpha
    )
    test_tokens = generate_data.generate_test_token(
        B=B, d_in=d_in, d_out=d_out, x_low=-alpha, x_high=alpha
    )

    batch_data = torch.concatenate([train_tokens, test_tokens], axis=1)

    target = torch.einsum("bnd, bdy -> bny", test_tokens[:, :, :d_in], batch_W)

    predictions = model(batch_data)[:, -1:, d_in:]

    loss = custom_mse_loss(predictions, target)

    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()

    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    # 검증 손실 계산
    model.eval()  # 모델을 평가 모드로 설정

    with torch.no_grad():

        predictions = model(val_data)[:, -1:, d_in:]
        val_loss = custom_mse_loss(predictions, val_target)

        val_losses.append(val_loss.item())

    if step % 100 == 0:
        print(f"step {step} loss: {loss.item():.4f}, val_loss: {val_loss.item():.4f}")


def moving_average(data, window_size):
    """단순 이동 평균 계산"""
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


# 이동 평균 적용
window_size = 30
ma_losses = moving_average(losses, window_size)


gd_predictions = lsa_gd(val_data)[:, -1:, d_in:]
gd_loss = custom_mse_loss(gd_predictions, val_target).item()


# 시각화
plt.figure(figsize=(10, 5))
plt.plot(
    range(window_size - 1, len(losses)),
    ma_losses,
    label="Moving Average training Loss",
    color="orange",
    alpha=0.3,
)

plt.plot(
    [0, len(val_losses)],
    [gd_loss, gd_loss],
    label="GD",
    color="green",
)
plt.plot(
    val_losses,
    label="validataion Loss",
    color="purple",
)


plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss with Moving Average Smoothing")
plt.legend()
plt.show()
