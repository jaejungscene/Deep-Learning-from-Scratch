import numpy as np

def NAND(x1, x2):
  x = np.array([x1,x2])
  w = np.array([-0.5,-0.5])
  bias = 0.7
  y = np.sum(x*w) + bias
  if y<=0: return 0
  else: return 1

def NOT(x):
  return NAND(x,x)

def OR(x1, x2):
  s1 = NAND(x1,x1)
  s2 = NAND(x2,x2)
  return NAND(s1, s2)

def AND(x1, x2):
  s1 = NAND(x1,x2)
  return NAND(s1, s1)

print('0 |',NOT(0))
print('1 |',NOT(1))
print('-'*40)

print('0 0 |',OR(0,0))
print('0 1 |',OR(0,1))
print('1 0 |',OR(1,0))
print('1 1 |',OR(1,1))
print('-'*40)

print('0 0 |',AND(0,0))
print('0 1 |',AND(0,1))
print('1 0 |',AND(1,0))
print('1 1 |',AND(1,1))