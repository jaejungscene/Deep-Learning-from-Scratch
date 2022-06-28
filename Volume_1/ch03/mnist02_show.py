import sys                                                            
sys.path.append("/Users/jaejungscene/Projects/Machine_Learning/Deep_Learning_from_scratch")
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt



(x_train, t_train), (x_test, t_test) = \
  load_mnist(flatten=True,  normalize=False)

img1 = x_test[0]
label = t_test[0]

print(label)
print(img1.shape)
img2 = img1.reshape(28,28)
print(img2.shape)

testImg = np.array([[0,127],[127,255]])
# grayscale
pil_img = Image.fromarray(np.uint8(testImg))
pil_img.show()

# RGB
testImg = np.array([[[255, 0, 0],[0, 255, 0]],[[0, 255, 0], [0, 0, 255]]])
plt.imshow(testImg)
plt.show()