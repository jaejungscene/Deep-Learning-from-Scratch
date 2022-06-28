import numpy as np

#################### 내가 만든 XOR ######################
def AND1(x1, x2):
  x = np.array([x1, x2])
  w = np.array([1, -1])
  b = 0.5
  tmp = np.sum(x*w) + b
  if tmp < 0:
    return 1
  else:
    return 0

def AND2(x1, x2):
  x = np.array([x1, x2])
  w = np.array([1, -1])
  b = -0.5
  tmp = np.sum(x*w) + b
  if tmp > 0:
    return 1
  else:
    return 0

def OR(x1, x2):
  x = np.array([x1,x2])
  w = np.array([0.5,0.5])
  b = -0.1
  tmp = np.sum(x*w) + b
  if tmp > 0:
    return 1
  else:
    return 0

def XOR1(x1, x2):
  return OR(AND1(x1,x2), AND2(x1,x2))

print('0 0 |',XOR1(0,0))
print('0 1 |',XOR1(0,1))
print('1 0 |',XOR1(1,0))
print('1 1 |',XOR1(1,1))
print('-'*40)
##################################################

################### 책에 있는 XOR ##################
def NAND(x1, x2):
  x = np.array([x1,x2])
  w = np.array([-0.5,-0.5])
  bias = 0.7
  y = np.sum(x*w) + bias
  if y<=0: return 0
  else: return 1

def AND(x1, x2):
  x = np.array([x1,x2])
  w = np.array([0.5,0.5])
  bias = -0.7
  y = np.sum(x*w) + bias
  if y<=0: return 0
  else: return 1

def XOR2(x1,x2):
  s1 = NAND(x1,x2)
  s2 = OR(x1,x2)
  return AND(s1,s2)

print('0 0 |',XOR2(0,0))
print('0 1 |',XOR2(0,1))
print('1 0 |',XOR2(1,0))
print('1 1 |',XOR2(1,1))