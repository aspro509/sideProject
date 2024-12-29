import torch
import torch.nn as nn
import numpy as np


class LSALayer(nn.Module):
    """
    1-head Linear Self Attention layer
    """

    def __init__(self, input_dim, output_dim, key_dim, value_dim):
        super(LSALayer, self).__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim

        # W_q, W_k, W_v, W_o 선형 변환을 위한 가중치 행렬
        self.W_q = nn.Linear(input_dim, key_dim, bias=False)
        self.W_k = nn.Linear(input_dim, key_dim, bias=False)
        self.W_v = nn.Linear(input_dim, value_dim, bias=False)
        self.P = nn.Linear(value_dim, output_dim, bias=False)

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
        attn_scores = torch.matmul(
            Q, K.transpose(-2, -1)
        )  # / (self.key_dim**0.5)  # (batch_size, sequence_length, sequence_length)

        # 가중합 계산
        context = torch.matmul(
            attn_scores, V
        )  # (batch_size, sequence_length, value_dim)

        # 출력 계산
        output = torch.add(x, self.P(context))  # (batch_size, sequence_length, output_dim)

        return output


class LSA_Model(nn.Module):
    def __init__(
        self, num_layers, input_dim, output_dim, key_dim, value_dim, hidden_dim=None
    ):
        super(LSA_Model, self).__init__()
        self.num_layers = num_layers

        # 단일 레이어 모델
        if num_layers == 1:
            self.lsa = LSALayer(input_dim, output_dim, key_dim, value_dim)
        # 다층 레이어 모델
        else:
            # 첫 번째 레이어: input_dim -> hidden_dim
            layers = [LSALayer(input_dim, hidden_dim, key_dim, value_dim)]
            # 중간 레이어: hidden_dim -> hidden_dim
            for _ in range(num_layers - 2):
                layers.append(LSALayer(hidden_dim, hidden_dim, key_dim, value_dim))
            # 마지막 레이어: hidden_dim -> output_dim
            layers.append(LSALayer(hidden_dim, output_dim, key_dim, value_dim))
            self.lsa_layers = nn.ModuleList(layers)

    def forward(self, x, mask=None):
        """
        Args:
            x: 입력 텐서, shape: (batch_size, sequence_length, input_dim)
            mask: 어텐션 마스크 텐서, shape: (batch_size, sequence_length, sequence_length)

        Returns:
            output: 출력 텐서, shape: (batch_size, sequence_length, output_dim)
        """
        if self.num_layers == 1:
            output = self.lsa(x, mask)
        else:
            output = x
            for layer in self.lsa_layers:
                output = layer(output, mask)
        return output


def main():
    """
    test 코드!
    """
    d = 10
    N = 100

    lsa_layer = LSALayer(d, d, d, d)

    E = np.random.normal(loc=0, scale=1, size=(10, N, d))
    E = torch.tensor(E)
    E = E.to(torch.float32)

    output = lsa_layer(E)
    print(output.shape)
    print(lsa_layer)


if __name__ == "__main__":
    main()
