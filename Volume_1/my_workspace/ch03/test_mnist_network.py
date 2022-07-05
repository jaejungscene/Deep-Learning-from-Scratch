import sys                                                                                           
sys.path.append("/Users/jaejungscene/Projects/AI/Deep_Learning_from_Scratch/Volume_1")
# parent directory부터 시작하기 위해 sys.path에 parent directory를 추가
import numpy as np
from Activation_Functions import sigmoid, softmax
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
from PIL import Image
import pickle

def get_data():
  (x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True,  normalize=True, one_hot_label=False)
  return x_test, t_test

def init_network():
  with open(sys.path[-1]+"/dataset/sample_weight.pkl", "rb") as f:
    network = pickle.load(f)
  return network

def predict(network, x):
  W1, W2, W3 = network["W1"], network["W2"], network["W3"]
  b1, b2, b3 = network["b1"], network["b2"], network["b3"]

  a1 = np.dot(x, W1) + b1     # (1,784)X(784,50) + (1,50) = (1,50)  <- 사진의 총 픽셀 28x28 = 784
  z1 = sigmoid(a1)            # (1,50)
  a2 = np.dot(z1, W2) + b2    # (1,50)X(50,100) + (1,100) = (1,100)
  z2 = sigmoid(a2)            # (1,100)
  a3 = np.dot(z2, W3) + b3    # (1,100)X(100,10) + (1,10) = (1,10)
  y = softmax(a3)             # (1,10)
  return y # (1,10) 0~9중에 어떤 숫자인지 판별하기 위해 결과 값은 10개가 나왔야 한다. 

def singleResultAt(network, x, n):
  y = predict(network, x[n])
  print("predict is", np.argmax(y))
  print("answer is", t[n])
  img = (x[n]).reshape(28,28)
  plt.imshow(img,cmap='gray')
  plt.show()

# for i in range(4):
#   singleResultAt(network, x, i)

x, t = get_data() # 10000개의 숫자 사진 -> x.shape = (10000, 784)
network = init_network()
accuracy_cnt = 0

W1, W2, W3 = network["W1"], network["W2"], network["W3"]
b1, b2, b3 = network["b1"], network["b2"], network["b3"]

print(type(W1))
print(W1.shape)
print(W1.max())
print(W1.min())
print(W1.mean())
print()

print(type(W2))
print(W2.shape)
print(W2.max())
print(W2.min())
print(W2.mean())
print()

print(type(W3))
print(W3.shape)
print(W3.max())
print(W3.min())
print(W3.mean())
print()

print(type(b1))
print(b1.shape)
print(b1.max())
print(b1.min())
print(b1.mean())
print()