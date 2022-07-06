import numpy as np
import matplotlib.pyplot as plt

a = np.random.randn(100, 100) *0.01
plt.hist(a.flatten(), 50, range=(-3,3))
plt.show()