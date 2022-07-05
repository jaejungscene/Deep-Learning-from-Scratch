import numpy as np

#######################################

def AND1(x1, x2):
  w1, w2, theta = 0.5, 0.5, 0.7
  total = x1*w1 + x2*w2
  if total <= theta:
    return 0
  else:
    return 1

print(AND1(0,0))
print(AND1(0,1))
print(AND1(1,0))
print(AND1(1,1))

#######################################
#theta를 bias로 변형
print('-'*40)

def AND2(x1, x2):
  x = np.array([x1,x2])
  w = np.array([0.5,0.5])
  bias = -0.7
  y = np.sum(x*w) + bias
  if y<=0: return 0
  else: return 1

print(AND2(0,0))
print(AND2(0,1))
print(AND2(1,0))
print(AND2(1,1))
