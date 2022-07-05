import sys                                                                                           
sys.path.append("/Users/jaejungscene/Projects/AI/Deep_Learning_from_Scratch/Volume_1")
from dataset.mnist import load_mnist
import numpy as np

def printArray(x):
  for i in np.arange(len(x)):
    if i%28==0:
      print()
    print("%4d" %(x[i]), end=" ")

temp = load_mnist(flatten=True,  normalize=False, one_hot_label=False)
print(type(temp))
print(len(temp))
print(len(temp[0]))
print(len(temp[0][0]))

(x_train, t_train), (x_test, t_test) = temp

print("type(x_train) is",type(x_train))
print(x_train.shape)   #(60000, 784)
print(t_train.shape)   #(60000,)
print()

print("type(x_train[0]) is",type(x_train[0]))   #numpy.ndarray
print(x_test.shape)    #(10000, 784)
print(t_test.shape)    #(10000,)

print("-"*30)

printArray(x_train[0]) # img 5에 대한 픽셀값들
print("\n",t_train[0]) # 5
printArray(x_test[0])  # img 7에 대한 픽셀값들
print("\n",t_test[0])  # 7

print("-"*30)

temp = load_mnist(flatten=True,  normalize=False, one_hot_label=True)
(x_train, t_train), (x_test, t_test) = temp
printArray(x_train[0]) # img 5에 대한 픽셀값들
print("\n",t_train[0]) # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
printArray(x_test[0])  # img 7에 대한 픽셀값들
print("\n",t_test[0])  # [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
print(type(x_test[0][0]))