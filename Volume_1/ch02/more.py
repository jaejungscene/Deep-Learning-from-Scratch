import numpy as np

def NAND(x1, x2):
  x = np.array([x1,x2])
  w = np.array([1,-0.5])
  bias = -0.16666
  y = np.sum(x*w) + bias
  if y<=0: return 0
  else: return 1

print('0 0 |',NAND(0,0))
print('0 1 |',NAND(0,1))
print('1 0 |',NAND(1,0))
print('1 1 |',NAND(1,1))