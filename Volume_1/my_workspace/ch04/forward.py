import sys
sys.path.append("/Users/jaejungscene/Projects/AI/Deep_Learning_from_Scratch/Volume_1")

import numpy as np
from Activation_Functions import softmax
from Loss_Functions import cross_entropy_error
from ch04.numerical_grandient import numerical_gradient

t = np.array([0, 0, 1]) #정답 레이블
W = np.array([ [0.47355232, 0.9977393,  0.84668094],
               [0.85557411, 0.03563661, 0.69422093] ])
x = np.array([0.6, 0.9])
result = x@W
print(result) # W 곱셈
print()

result = softmax(result) # activation function 통과
print(result)
print()

def loss(x, W, t):
   z = x@W
   y = softmax(z)
   loss_value = cross_entropy_error(y, t)
   return loss_value
print(loss(x, W, t))
print()

def f(W):
   return loss(x, W, t)

dW = numerical_gradient(f, W)
print(dW)