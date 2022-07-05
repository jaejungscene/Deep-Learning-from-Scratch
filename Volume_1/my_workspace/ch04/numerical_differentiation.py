'''
numerical differentiation(수치 미분) 구현
부동소수점 표현의 한계로 컴퓨터는 진정한 미분 값을 구할 수 없다.
따라서 밑과 같이 최대한 실제 미분 값에 근접하도록 수치 미분의 값을 구현한다.
'''

# =============== 일반 미분 =================
# df(x)        f(x+h) - f(x)
# -----   =    -------------
#  dx                h
# ======================================

# =============== 중심차분을 통한 수치미분 =================
# df(x)        f(x+h) - f(x-h)
# -----   =    -------------
#  dx                2*h
# ======================================

def numerical_diff01(f, x):
   h = 1e-50
   return (f(x+h) - f(x)) / h



# 위 numberical_diff01 함수를 계산한 함수
# 1e-4 정도의 값은 성능과 표현 적절하게 유지해주는 값이라고 알려져있음.
# x에서의 실제 미분값을 최대한 근사하기 위해 x를 중심으로
# 전후의 차분을 계산하는 중심차분(or 중앙차분)활용하여 수치미분 구현
def numerical_diff02(f, x):
   h = 1e-4
   return (f(x+h)- f(x-h)) / (2*h)


###########################################################


import numpy as np
import matplotlib.pylab as plt


def function_1(x):
   return 0.01*(x**2) + 0.1*x

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

print(numerical_diff01(function_1, 5)) # 0.0
print(numerical_diff02(function_1, 5)) # 0.1999999999990898


