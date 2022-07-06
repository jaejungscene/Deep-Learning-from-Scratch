import numpy as np

x = np.random.rand(1,2,3)
print(x)
print(x.shape)

print(np.max(x, axis=0))
print(np.max(x, axis=0).shape)

print(np.max(x, axis=1))
print(np.max(x, axis=1).shape)

print(np.max(x, axis=2))
print(np.max(x, axis=2).shape)

# print(np.max(x, axis=3))
# print(np.max(x, axis=3).shape)