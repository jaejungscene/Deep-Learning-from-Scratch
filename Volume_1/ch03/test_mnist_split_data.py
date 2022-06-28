import sys                                                                                           
sys.path.append("/Users/jaejungscene/Projects/Machine_Learning/Deep_Learning_from_scratch")
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

x, t = get_data() # 10000개의 숫자 사진 -> x.shape = (10000, 784), t.shape = (10000,)
from sklearn.model_selection import train_test_split
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.2)
network = init_network()
accuracy_cnt = 0

size_of_predict = len(x_train)
print('the number of train data :',size_of_predict)

## size_of_predict만큼의 사진을 순차적으로 처리
for i  in range(size_of_predict):
  y = predict(network, x_train[i])
  p = np.argmax(y)  # 확률이 가장 높은 원소의 인덱스를 얻는다.(array에서 가장 높은 값을 가지는 인덱스를 반환한다.)
  if p == t[i]:
    accuracy_cnt += 1
print("Accuracy:", str(float(accuracy_cnt)/len(x[:size_of_predict]))) # Accuracy: 0.9352  <-- 확률
