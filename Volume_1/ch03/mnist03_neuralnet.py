import sys                                                               # 부모 디렉터리로 부터
sys.path.append("/Users/jaejungscene/Projects/Machine_Learning/Deep_Learning_from_scratch") # 시작하기 위해
import numpy as np
from Functions import sigmoid, softmax
from dataset.mnist import load_mnist
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

  a1 = np.dot(x, W1) + b1
  z1 = sigmoid(a1)
  a2 = np.dot(z1, W2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3) + b3
  y = softmax(a3)
  return y

x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i  in range(len(x)):
  y = predict(network, x[i])
  p = np.argmax(y)  # 확률이 가장 높은 원소의 인덱스를 얻는다.(array에서 가장 높은 값을 가지는 인덱스를 반환한다.)
  if p == t[i]:
    accuracy_cnt += 1

print("Accuracy:",str(float(accuracy_cnt)/len(x))) # Accuracy: 0.9352

