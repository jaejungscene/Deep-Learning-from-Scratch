import sys                                                                                           
sys.path.append("/Users/jaejungscene/Projects/Machine_Learning/Deep_Learning_from_scratch")
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) =\
  load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000, 10)

# 훈련 데이타가 너무 많을 때 무작위로 적절한 크기의 데이타들을 뽑아 학습 = mini batch 학습
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) # 지정한 점위의 수 중에서 random으로 원하는 개수만 꺼냄 
print(batch_mask.shape) # (10,)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
print(x_batch.shape)
print(t_batch.shape)