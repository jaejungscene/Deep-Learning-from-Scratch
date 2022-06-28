import numpy as np

# activation function으로 작용
def sigmoid(x):
  return 1/(1+np.exp(-x))

# 마지막 출력층에서의 활성화 함수로 작용
def identity_function(x):
  return x

# Dictionary 변수에 각 층에 필요한 매개변수(가중치와 편향)을 저장
def init_network():
  network = {}
  network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
  network["B1"] = np.array([0.1, 0.2, 0.3])
  network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
  network["B2"] = np.array([0.1, 0.2])
  network["W3"] = np.array([[0.1, 0.3],[0.2, 0.4]])
  network["B3"] = np.array([0.1, 0.2])

  return network

# 입력 신호를 출력으로 변환하는 처리 과정을 모두 구현
def forward(network, x):
  W1, W2, W3 = network["W1"], network["W2"], network["W3"]
  B1, B2, B3 = network["B1"], network["B2"], network["B3"]
  a1 = np.dot(x, W1) + B1
  z1 = sigmoid(a1)
  a2 = np.dot(z1, W2) + B2
  z2 = sigmoid(a2)
  a3 = z2@W3 + B3
  y = identity_function(a3)

  return y

x = np.array([1.0,0.5])
y = forward(init_network(), x)
print(y)
