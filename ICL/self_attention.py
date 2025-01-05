import torch
import torch.nn as nn
import numpy as np
import generate_data


class SALayer(nn.Module):
    """
    1-head Linear Self Attention layer
    """

    def __init__(self, input_dim, output_dim, key_dim, value_dim):
        super(SALayer, self).__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim

        # W_q, W_k, W_v, W_o 선형 변환을 위한 가중치 행렬
        self.W_q = nn.Linear(input_dim, key_dim, bias=False)
        self.W_k = nn.Linear(input_dim, key_dim, bias=False)
        self.W_v = nn.Linear(input_dim, value_dim, bias=False)
        self.P = nn.Linear(value_dim, output_dim, bias=False)
        # Initialize weights using truncated normal
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize weights using Haiku-style truncated normal with stddev=0.002.
        """
        std = 0.002
        for module in [self.W_q, self.W_k, self.W_v, self.P]:
            self.truncated_normal_(module.weight, mean=0.0, std=std)

    @staticmethod
    def truncated_normal_(tensor, mean=0.0, std=0.002):
        """
        Apply truncated normal initialization to a tensor.
        """
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_(mean=mean, std=std)
            valid = (tmp < mean + 2 * std) & (tmp > mean - 2 * std)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))

    def forward(self, x):
        """
        Args:
            x: 입력 텐서, shape: (batch_size, sequence_length, input_dim)
            mask: 어텐션 마스크 텐서, shape: (batch_size, sequence_length, sequence_length)

        Returns:
            output: 출력 텐서, shape: (batch_size, sequence_length, output_dim)
        """
        # 쿼리, 키, 값 계산
        Q = self.W_q(x)  # (batch_size, sequence_length, key_dim)
        K = self.W_k(x)  # (batch_size, sequence_length, key_dim)
        V = self.W_v(x)  # (batch_size, sequence_length, value_dim)

        # 어텐션 스코어 계산
        attn_scores = torch.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) / (self.key_dim**0.5), dim=2
        )  # (batch_size, sequence_length, sequence_length)

        # 가중합 계산
        context = torch.matmul(
            attn_scores, V
        )  # (batch_size, sequence_length, value_dim)

        # 출력 계산
        output = torch.add(
            x, self.P(context)
        )  # (batch_size, sequence_length, output_dim)

        return output


def main():
    d_in = 10
    d_out = 1
    d_token = d_in + d_out

    B = 10**4
    N = 10
    alpha = 1.0

    train_tokens, batch_W = generate_data.generate_linear_regression_batch(
        B=B,
        N=N,
        d_in=d_in,
        d_out=d_out,
        noise_std=0.0,
        x_low=-alpha,
        x_high=alpha,
        random_seed=0,
    )
    test_tokens = generate_data.generate_test_token(
        B=B, d_in=d_in, d_out=d_out, x_low=-alpha, x_high=alpha
    )

    val_data = torch.concatenate([train_tokens, test_tokens], axis=1)
    val_target = torch.einsum("bnd, bdy -> bny", test_tokens[:, :, :d_in], batch_W)

    model = SALayer(d_token, d_token, d_token, d_token)

    prediction = model(val_data)

    print(prediction.shape)


if __name__ == "__main__":
    main()
