import sys                                                                                           
sys.path.append("/Users/jaejungscene/Projects/AI/Deep_Learning_from_scratch/Volume_1")
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) =\
  load_mnist(normalize=True, one_hot_label=True)

print(type(x_train))
print(x_train.shape) # (60000, 784)
print()
print(type(x_test))
print(t_train.shape) # (60000, 10)
print()

# 훈련 데이타가 너무 많을 때 무작위로 적절한 크기의 데이타들을 뽑아 학습 = mini batch 학습
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) # 0~599999 사이의 값을 꺼냄 --> index 값

print(batch_mask)
print(batch_mask.shape) # (10,)
print()

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
print(x_batch.shape)
print(t_batch.shape)