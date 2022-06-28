import numpy as np

def NOR(x1, x2):
  x = np.array([x1,x2])
  w = np.array([-0.5,-0.5])
  bias = 0.4
  y = np.sum(x*w) + bias
  if y<=0: return 0
  else: return 1

print('0 0 |',NOR(0,0))
print('0 1 |',NOR(0,1))
print('1 0 |',NOR(1,0))
print('1 1 |',NOR(1,1))