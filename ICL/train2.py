import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from linear_self_attention import LSALayer
from linear_self_attention_GD import LSA_GD
from self_attention import SALayer
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

model1 = LSALayer(d_token, d_token, d_token, d_token)
model2 = SALayer(d_token, d_token, d_token, d_token)

lr = 0.001
optimizer1 = optim.Adam(model1.parameters(), lr=lr)  # SGD with defort setting fails to decrease loss
optimizer2 = optim.Adam(model2.parameters(), lr=lr)
num_steps = 2000


lsa_gd = LSA_GD()

train_tokens, batch_W = generate_data.generate_linear_regression_batch(
        B=val_B, N=N, d_in=d_in, d_out=d_out, noise_std=0.0, x_low=-alpha, x_high=alpha, random_seed=0
    )
test_tokens = generate_data.generate_test_token(
    B=val_B, d_in=d_in, d_out=d_out, x_low=-alpha, x_high=alpha
)

val_data = torch.concatenate([train_tokens, test_tokens], axis=1)
val_target = torch.einsum("bnd, bdy -> bny", test_tokens[:, :, :d_in], batch_W)


losses1 = []
losses2 = []
val_losses1 = []
val_losses2 = []
for step in range(num_steps):

    model1.train()
    model2.train()

    train_tokens, batch_W = generate_data.generate_linear_regression_batch(
        B=B, N=N, d_in=d_in, d_out=d_out, noise_std=0.0, x_low=-alpha, x_high=alpha
    )
    test_tokens = generate_data.generate_test_token(
        B=B, d_in=d_in, d_out=d_out, x_low=-alpha, x_high=alpha
    )

    batch_data = torch.concatenate([train_tokens, test_tokens], axis=1)

    target = torch.einsum("bnd, bdy -> bny", test_tokens[:, :, :d_in], batch_W)

    predictions1 = model1(batch_data)[:, -1:, d_in:]
    predictions2 = model2(batch_data)[:, -1:, d_in:]

    loss1 = custom_mse_loss(predictions1, target)
    loss2 = custom_mse_loss(predictions2, target)

    losses1.append(loss1.item())
    losses2.append(loss2.item())

    optimizer1.zero_grad()
    loss1.backward()
    #torch.nn.utils.clip_grad_norm_(model1.parameters(), max_grad_norm)
    optimizer1.step()

    optimizer2.zero_grad()
    loss2.backward()
    #torch.nn.utils.clip_grad_norm_(model2.parameters(), max_grad_norm)
    optimizer2.step()

    # 검증 손실 계산
    model1.eval()  # 모델을 평가 모드로 설정
    model2.eval()


    with torch.no_grad():

        predictions1 = model1(val_data)[:, -1:, d_in:]
        predictions2 = model2(val_data)[:, -1:, d_in:]

        val_loss1 = custom_mse_loss(predictions1, val_target)
        val_loss2 = custom_mse_loss(predictions2, val_target)
        
        val_losses1.append(val_loss1.item())
        val_losses2.append(val_loss2.item())

    if step % 100 == 0:
        print(f"step {step:4d} loss: {loss1.item():.4f}, {loss2.item():.4f} val_loss: {val_loss1.item():.4f}, {val_loss2.item():.4f}")

def moving_average(data, window_size):
    """단순 이동 평균 계산"""
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

# 이동 평균 적용
window_size = 30
ma_losses = moving_average(losses1, window_size)


gd_predictions = lsa_gd(val_data)[:, -1:, d_in:]
gd_loss = custom_mse_loss(gd_predictions, val_target).item()


# 시각화
plt.figure(figsize=(10, 5))
plt.plot(
    range(window_size - 1, len(losses1)),
    ma_losses,
    label="Moving Average training Loss",
    color="orange",
    alpha=0.3,
)

plt.plot(
    [0, len(val_losses1)],
    [gd_loss, gd_loss],
    label = "GD",
    color="green",
)

plt.plot(
    val_losses1,
    label="validataion LSA Loss",
    color="purple",

)

plt.plot(
    moving_average(losses2, window_size),
    label="train loss2",
    color="yellow",
    alpha=0.5

)

plt.plot(
    val_losses2,
    label="validataion SA Loss",
    color="blue",

)


plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss with Moving Average Smoothing")
plt.legend()
plt.show()
