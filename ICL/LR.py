import numpy as np


def generate_linear_regression_data(
    n: int = 10,
    d: int = 2,
    dy: int = 1,
    noise_std: float = 0.1,
    x_low: float = -1.0,
    x_high: float = 1.0,
    random_seed=None,
):
    """
    선형 회귀 데이터셋을 생성하는 함수.

    Parameters:
    - n (int): 생성할 데이터 포인트의 수. 기본값은 10.
    - d (int): 입력 벡터 x의 차원. 기본값은 2.
    - dy (int): 출력 벡터 y의 차원. 기본값 1.
    - noise_std (float): y 값에 추가할 가우시안 노이즈의 표준편차. 기본값은 0.1. 0으로 설정시 노이즈 없음.
    - x_low (float): x_i의 각 성분의 최소값. 기본값은 0.0.
    - x_high (float): x_i의 각 성분의 최대값. 기본값은 1.0.
    - random_seed (int, optional): 랜덤 시드 설정을 위한 정수. 재현 가능성을 위해 사용.

    Returns:
    - X (numpy.ndarray): shape (n, d), 입력 데이터.
    - y (numpy.ndarray): shape (n,), 타겟 값.
    - W (numpy.ndarray): shape (d,dy), 가중치 벡터.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # 가중치 W를 isotropic 가우시안 분포에서 샘플링
    W = np.random.normal(loc=0.0, scale=1.0, size=(d, dy))

    # x_i를 균등 분포에서 샘플링
    X = np.random.uniform(low=x_low, high=x_high, size=(n, d))

    # 타겟 값 y_i = W^T x_i + 노이즈

    y = X.dot(W)
    if noise_std > 0.0:
        noise = np.random.normal(loc=0.0, scale=noise_std, size=(n, dy))
        y += noise
    return X, y, W
