import numpy as np
import matplotlib.pyplot as plt

# 함수들은 모두 확률로써 계산하기 위해 
# output 값이 0~1사이 이다!!

# 계단 함수
def step_function(x):
  temp = x>0 
  return temp.astype(np.int64) # astype은 numpy의 array의 함수다.

# 시그모이드 함수
def sigmoid(x):
  return 1/(1+np.exp(-x))

# ReLU 함수
def ReLU(x):
  return np.maximum(0, x)

#softmax function
def softmax(x):
  c = np.max(x)
  exp_x = np.exp(x-c) # 오버플로우 대책
  return exp_x/np.sum(exp_x)

# x = np.arange(-10, 10, 0.1)
# y = softmax(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()

