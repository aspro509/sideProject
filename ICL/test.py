import generate_data
import numpy as np
import matplotlib.pyplot as plt


E, W = generate_data.generate_linear_regression_batch(100000, d_in=10)

print(W.shape)
print(W.flatten().std())
print(W.mean())

plt.hist(W.flatten(), bins = 100)
plt.show()
