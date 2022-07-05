import sys                                                                                           
sys.path.append("/Users/jaejungscene/Projects/AI/Deep_Learning_from_Scratch/Volume_1/my_workspace")
from dataset.mnist import load_mnist
import numpy as np

temp = load_mnist(flatten=True,  normalize=False, one_hot_label=False)
print(type(temp)) # tuple
print(len(temp)) # 2
print()

# tuple of train data
print(type(temp[0])) # tuple
print(len(temp[0]))
print()

# tuple of test data
print(type(temp[0])) # tuple
print(len(temp[1]))
print()

# train data 개수
print(type(temp[0][0])) # numpy.ndarray
print('x_train :',temp[0][0].shape) # x_train 
print('t_train :',temp[0][1].shape) # t_train
print()

# test data 개수
print(type(temp[1][0])) # numpy.ndarray
print('x_test :',temp[1][0].shape) # x_test
print('t_test :',temp[1][1].shape) # t_test
print()

print('===================================================')

temp = load_mnist(flatten=True,  normalize=False, one_hot_label=True)
print(type(temp)) # tuple
print(len(temp)) # 2
print()

# tuple of train data
print(type(temp[0])) # tuple
print(len(temp[0]))
print()

# tuple of test data
print(type(temp[0])) # tuple
print(len(temp[1]))
print()

# train data 개수
print(type(temp[0][0])) # numpy.ndarray
print('x_train :',temp[0][0].shape) # x_train 
print('t_train :',temp[0][1].shape) # t_train
print()

# test data 개수
print(type(temp[1][0])) # numpy.ndarray
print('x_test :',temp[1][0].shape) # x_test
print('t_test :',temp[1][1].shape) # t_test
print()