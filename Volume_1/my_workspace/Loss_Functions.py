# The function we want to minimize or maximize is called the objective function or criterion.
# When we are minimizing it, we may also call it the cost function, loss function, or error function.

# y = 추정 값, t = 정답 레이블
import numpy as np


# Sum of Squares for Error, SSE
def sum_squares_error(y, t):
  return 0.5 * np.sum((y-t)**2)


# Cross Entropy Error, CEE
def cross_entropy_error(y, t, one_hot=True):
  # dimension이 1일 때 dimension을 하나 증가시키기 위해
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)

  batch_size = y.shape[0]
  # np.log()함수에 0이 입력되면 마이너스 무한대를 뜻하는
  # -inf가 출력되어 계산을 진행할 수 없게 되기 때문에
  # 아주 작은 값(1e-7=10^-7)통해 log의 값이 -inf가 되지 않도록 한다.
  if one_hot: # label이 one hot encoding일 때
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
  else: # label이 숫자로만 표기되어 있을 때
    return -np.sum(t * np.log(y[np.arange(batch_size, t)] + 1e-7)) / batch_size


