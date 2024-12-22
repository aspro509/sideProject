import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

##############################################################################
# 1) 모델 정의: MLP
##############################################################################
class PINNModel(tf.keras.Model):
    def __init__(self, hidden_units=20, hidden_layers=3, activation='tanh'):
        super(PINNModel, self).__init__()
        self.hidden_layers = []
        for _ in range(hidden_layers):
            self.hidden_layers.append(tf.keras.layers.Dense(hidden_units, activation=activation))
        self.out_layer = tf.keras.layers.Dense(1, activation=None)  # 최종 출력: y(x)

    def call(self, x):
        """
        x: shape (batch_size, 1)
        return: y_pred (batch_size, 1)
        """
        z = x
        for layer in self.hidden_layers:
            z = layer(z)
        return self.out_layer(z)

##############################################################################
# 2) ODE 잔차(Residual) 및 초기조건(BC/IC) 정의
##############################################################################
@tf.function
def compute_residual(model, x):
    """
    2차 미분방정식: y''(x) + y(x) = 0
    residual = y''(x) + y(x)
    """
    with tf.GradientTape() as tape2:
        # 1차 미분을 위해 한 번 감싸고,
        # 그 안에서 또 GradientTape를 써서 2차 미분을 구한다.
        tape2.watch(x)
        with tf.GradientTape() as tape1:
            tape1.watch(x)
            y_pred = model(x)  # shape (batch_size, 1)
        dy_dx = tape1.gradient(y_pred, x)  # 1차 미분
    d2y_dx2 = tape2.gradient(dy_dx, x)     # 2차 미분

    residual = d2y_dx2 + y_pred  # (batch_size, 1)
    return residual

@tf.function
def boundary_loss(model):
    """
    초기조건:
      y(0) = 0  --> residual_1 = y(0) - 0
      y'(0)= 1  --> residual_2 = dy/dx(0) - 1
    """
    x0 = tf.constant([[0.0]], dtype=tf.float32)  # (1,1)
    with tf.GradientTape() as tape:
        tape.watch(x0)
        y0 = model(x0)              # y(0)
    dy0_dx = tape.gradient(y0, x0)  # y'(0)

    # 각각 0에 가깝도록
    r1 = y0 - 0.0      # y(0) - 0
    r2 = dy0_dx - 1.0  # y'(0) - 1
    return r1, r2

##############################################################################
# 3) 전체 손실함수(loss) 및 학습 스텝
##############################################################################
@tf.function
def loss_fn(model, x_in):
    """
    전체 손실:
      1) ODE 잔차의 MSE
      2) 초기조건(BC/IC) 위배 정도의 MSE
    """
    # ODE residual
    residual = compute_residual(model, x_in)       # shape (N, 1)
    loss_ode = tf.reduce_mean(tf.square(residual)) # MSE
    
    # Boundary condition residual
    r_bc = boundary_loss(model)  # (r1, r2)
    loss_bc = 0.0
    for rb in r_bc:
        loss_bc += tf.reduce_mean(tf.square(rb))
    
    return loss_ode + loss_bc

@tf.function
def train_step(model, x_in, optimizer):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(model, x_in)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value

##############################################################################
# 4) 학습 (Training)
##############################################################################
# (A) 학습용 x 데이터: [-pi, pi] 구간에서 200개
x_train = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1).astype(np.float32)

# 모델 및 옵티마이저 설정
model = PINNModel(hidden_units=20, hidden_layers=3, activation='tanh')
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# 학습 루프
EPOCHS = 5000
losses = []
for epoch in range(1, EPOCHS+1):
    loss_val = train_step(model, x_train, optimizer)
    losses.append(loss_val)
    if epoch % 500 == 0:
        print(f"Epoch {epoch}/{EPOCHS}, loss = {loss_val.numpy():.6f}")

##############################################################################
# 5) 예측 & 시각화 (학습 구간 vs. 확장 구간)
##############################################################################
# (B) 예측용 x 데이터: [-2pi, 2pi]로 더 넓게
x_test = np.linspace(-2*np.pi, 2*np.pi, 400).reshape(-1, 1).astype(np.float32)
y_pred = model(x_test).numpy().flatten()
y_true = np.sin(x_test).flatten()  # 실제 해

# 시각화
plt.figure(figsize=(9,5))
plt.plot(x_test, y_true, 'b-', label='True sin(x)')
plt.plot(x_test, y_pred, 'r--', label='NN Prediction')
plt.axvspan(-2*np.pi, -np.pi, color='gray', alpha=0.1, label='Outside Training Range')
plt.axvspan(np.pi, 2*np.pi, color='gray', alpha=0.1)
plt.title("Solve ODE: y'' + y = 0 with y(0)=0, y'(0)=1  (PINN)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# 훈련 구간에 대한 예측 오차 정량화 (선택)
x_train_torch = tf.constant(x_train, dtype=tf.float32)
y_train_pred = model(x_train_torch).numpy().flatten()
y_train_true = np.sin(x_train).flatten()
mse_train = np.mean((y_train_pred - y_train_true)**2)
print(f"[Train Range] MSE = {mse_train:.8f}")

# 확장 구간(예: [-2π, -π) 와 (π, 2π])에 대한 MSE도 간단히 보기
idx_left = x_test.flatten() < -np.pi
idx_right = x_test.flatten() > np.pi
mse_left = np.mean((y_pred[idx_left] - y_true[idx_left])**2)
mse_right = np.mean((y_pred[idx_right] - y_true[idx_right])**2)
print(f"[Outside Range Left]  MSE = {mse_left:.8f}")
print(f"[Outside Range Right] MSE = {mse_right:.8f}")

plt.figure(figsize=(9,5))
plt.plot(losses[1000:])
plt.grid(True)
plt.show()
