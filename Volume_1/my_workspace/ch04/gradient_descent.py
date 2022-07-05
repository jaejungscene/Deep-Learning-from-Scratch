import numpy as np
import matplotlib.pyplot as plt


def function_2(x):
   return x[0]**2 + x[1]**2


# partial differentiation(편미분)
# ======================================
#                f(x+h, y) - f(x, y)
# fx(x,y)   =    -------------------
#                        h
# ======================================

def numerical_gradient(f, x):
   h = 1e-4
   grad = np.zeros_like(x) # x와 형상이 같은 배열 생성

   for i in range(x.size):
      temp_x = x[i]

      x[i] = temp_x + h
      fx1 = f(x)

      x[i] = temp_x - h
      fx2 = f(x)

      grad[i] = (fx1 - fx2) / (2*h)
      x[i] = temp_x
   
   return grad

point_x0 = []
point_x1 = []

def gradient_descent(f, init_x, learning_rate=0.01, step_num=100):
   x = init_x
   global point_x0
   global point_x1

   for i in range(step_num):
      point_x0.append(x[0])
      point_x1.append(x[1])
      grad = numerical_gradient(f, x)
      x = x - (learning_rate * grad)
   
   return x


init_x = np.array([-3.0, 4.0]) # 시작 점
result = gradient_descent(function_2, init_x=init_x, learning_rate=0.1, step_num=100)
print(result)

figure, axes = plt.subplots()
for i in np.arange(0.3, 10, 1):
   draw_circle = plt.Circle((0, 0), i, fill=False, linestyle='--')
   plt.gcf().gca().add_artist(draw_circle)
axes.set_aspect(1)
plt.scatter(point_x0, point_x1, marker='o', s=10)
plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)
plt.show()