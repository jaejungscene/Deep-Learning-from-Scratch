import sys, os
from Volume_1.common.layers import SoftmaxWithLoss
sys.path.append("/Users/jaejungscene/Projects/AI/Deep_Learning_from_Scratch/Volume_1")
from dataset.mnist import load_mnist
import numpy as np

(x_train, t_train), (x_test, t_test)  = load_mnist(flatten=True,  normalize=False, one_hot_label=False)

layer = SoftmaxWithLoss()

