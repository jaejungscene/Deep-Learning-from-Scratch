import numpy as np


def function_2(x):
   return x[0,0]**2 + x[0,1]**2


# partial differentiation(편미분)
# ======================================
#                f(x+h, y) - f(x, y)
# fx(x,y)   =    -------------------
#                        h
# ======================================

def numerical_gradient(f, x):
   if len(x.shape) == 1:
      x = x.reshape(1, x.shape[0])
   
   h = 1e-4
   grad = np.zeros_like(x) # x와 형상이 같은 배열 생성

   for i in range(x.shape[0]):
      for j in range(x.shape[1]):
         temp_x = x[i,j]

         x[i,j] = temp_x + h
         fx1 = f(x)

         x[i,j] = temp_x - h
         fx2 = f(x)

         grad[i,j] = (fx1 - fx2) / (2*h)
         x[i,j] = temp_x
   
   return grad

print( numerical_gradient(function_2, np.array([3.0, 4.0])) )
print( numerical_gradient(function_2, np.array([0.0, 2.0])) )
print( numerical_gradient(function_2, np.array([5.0, 0.0])) )
