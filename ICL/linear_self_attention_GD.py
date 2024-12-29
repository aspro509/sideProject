import torch
import numpy as np
from linear_self_attention import LSALayer
import generate_data

import matplotlib.pyplot as plt


def custom_mse_loss(predictions, targets):
    return torch.mean((predictions - targets) ** 2)


d_in = 10
d_out = 1
d_token = d_in + d_out

B = 10**4
N = 10
alpha = 1.0
eta = 0.001

def LSA_GD(eta=1.52339):
    lsa_gd = LSALayer(d_token, d_token, d_token, d_token)
    W_K = np.identity(d_token, dtype=np.float32)
    W_K[-d_out:, -d_out:] = 0

    W_V = -np.identity(d_token, dtype=np.float32)
    W_V[:d_in, :d_in] = 0


    P = np.identity(d_token, dtype=np.float32)*eta /N

    W_Q = torch.tensor(W_K.copy())
    W_K = torch.tensor(W_K.copy())
    W_V = torch.tensor(W_V.copy())
    P   = torch.tensor(P)
    with torch.no_grad():
        lsa_gd.W_k.weight.copy_(W_K)
        lsa_gd.W_q.weight.copy_(W_Q)
        lsa_gd.W_v.weight.copy_(W_V)
        lsa_gd.P.weight.copy_(P)
    
    return lsa_gd


train_tokens, batch_W = generate_data.generate_linear_regression_batch(
        B=B, N=N, d_in=d_in, d_out=d_out, noise_std=0.0, x_low=-alpha, x_high=alpha, random_seed=0
    )
test_tokens = generate_data.generate_test_token(
    B=B, d_in=d_in, d_out=d_out, x_low=-alpha, x_high=alpha
)

val_data = torch.concatenate([train_tokens, test_tokens], axis=1)
val_target = torch.einsum("bnd, bdy -> bny", test_tokens[:, :, :d_in], batch_W)

def main():
    def loss(eta):
        lsa_gd = LSA_GD(eta=eta)

        val_predictions = lsa_gd(val_data)[:, -1:, d_in:]

        loss = custom_mse_loss(val_predictions, val_target)
        return loss.item()

    etas = [i/100 for i in range(200)]
    losses_eta = []


    for eta in etas:
        losses_eta.append(loss(eta))

    index = np.argmin(losses_eta)
    eta_ = etas[index]

    print(eta_)

    plt.plot(etas, losses_eta, "b-")
    plt.scatter([eta_], [losses_eta[index]], c="r")
    plt.show()

    optimal_eta = 1.52339
if __name__ == "__main__":
    main()