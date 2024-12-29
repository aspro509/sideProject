import numpy as np
import torch


def generate_linear_regression_batch(
    B: int = 1,
    N: int = 10,
    d_in: int = 2,
    d_out: int = 1,
    noise_std: float = 0.0,
    x_low: float = -1.0,
    x_high: float = 1.0,
    random_seed=None,
):
    """
    벡터화된 배치 단위 선형 회귀 데이터셋을 생성하는 함수.

    Parameters:
    - B (int): 배치 사이즈, 생성할 W의 개수. 기본값은 1.
    - N (int): 각 배치당 생성할 데이터 포인트의 수. 기본값은 10.
    - d (int): 입력 벡터 x의 차원. 기본값은 2.
    - dy (int): 출력 벡터 y의 차원. 기본값은 1.
    - noise_std (float): y 값에 추가할 가우시안 노이즈의 표준편차. 기본값은 0.0. 0으로 설정 시 노이즈 없음.
    - x_low (float): x_i의 각 성분의 최소값. 기본값은 -1.0.
    - x_high (float): x_i의 각 성분의 최대값. 기본값은 1.0.
    - random_seed (int, optional): 랜덤 시드 설정을 위한 정수. 재현 가능성을 위해 사용.

    Returns:
    - batch_data (numpy.ndarray): shape (B, N, d + dy), 각 배치의 결합된 입력 및 출력 데이터.
    - batch_W (numpy.ndarray): shape (B, d, dy), 생성된 W의 배치.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # 배치별로 W를 샘플링 (B, d, dy)
    batch_W = np.random.normal(loc=0.0, scale=1.0, size=(B, d_in, d_out)).astype(
        np.float32
    )

    # 배치별로 X를 샘플링 (B, N, d)
    batch_X = np.random.uniform(low=x_low, high=x_high, size=(B, N, d_in)).astype(
        np.float32
    )

    # y를 계산 (B, N, dy) = (B, N, d) @ (B, d, dy)
    # np.einsum을 사용하여 배치별 행렬 곱을 수행
    batch_y = np.einsum("bnd, bdy -> bny", batch_X, batch_W)

    # 노이즈 추가 (선택 사항)
    if noise_std > 0.0:
        noise = np.random.normal(loc=0.0, scale=noise_std, size=(B, N, d_out)).astype(
            np.float32
        )
        batch_y += noise

    # X와 y를 결합하여 (B, N, d + dy) 형태로 만듦
    batch_data = np.concatenate([batch_X, batch_y], axis=2)  # (B, N, d + dy)

    return torch.tensor(batch_data), torch.tensor(batch_W)


def generate_test_token(
    B: int = 1024,
    d_in: int = 10,
    d_out: int = 1,
    x_low: float = -1.0,
    x_high: float = 1.0,
):

    test_X = np.random.uniform(low=x_low, high=x_high, size=(B, 1, d_in)).astype(
        np.float32
    )
    test_y = np.zeros((B, 1, d_out)).astype(np.float32)

    test_token = np.concatenate([test_X, test_y], axis=2)

    return torch.tensor(test_token)


# 사용 예제
if __name__ == "__main__":
    B = 8  # 배치 사이즈
    N = 5  # 각 배치당 데이터 포인트 수
    d_in = 4  # 입력 벡터 차원
    d_out = 2  # 출력 벡터 차원
    noise_std = 0.05
    random_seed = 42

    batch_data, batch_W = generate_linear_regression_batch(
        B=B,
        N=N,
        d_in=d_in,
        d_out=d_out,
        noise_std=noise_std,
        x_low=-1.0,
        x_high=1.0,
        random_seed=random_seed,
    )

    batch_test_tokens = generate_test_token(
        B=B,
        d_in=d_in,
        d_out=d_out,
        x_low=-1.0,
        x_high=1.0,
    )

    print(batch_test_tokens.shape)

    print("생성된 W (배치별):\n", batch_W)
    print("\n배치별 데이터 (X와 y가 결합된 형태):")
    for b in range(B):
        print(f"\n배치 {b+1}:")
        for i in range(N):
            data_point = batch_data[b, i]
            x = data_point[:d_in]
            y = data_point[d_in:]
            print(f"  데이터 {i+1}: X = {x}, y = {y}")
