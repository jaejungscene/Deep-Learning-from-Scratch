# The function we want to minimize or maximize is called the objective function or criterion.
# When we are minimizing it, we may also call it the cost function, loss function, or error function.
import numpy as np

# y = 추정 값, t = 정답 레이블

# Sum of Squares for Error, SSE
def sum_squares_error(y, t):
  return 0.5 * np.sum((y-t)**2)

# Cross Entropy Error, CEE
def cross_entropy_error(y, t):
  delta = 1e-7 # delta를 통해 log의 값이 -무한대가 되지 않도록 한다.
  return -np.sum(t * np.log(y+delta))