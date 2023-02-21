# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import cv2
# a = np.array([  [1, 2, 1, 1, 1, 2], 
#                 [1, 1, 1, 1, 1, 1],
#                 [1, 1, 1, 1, 2, 1],
#                 [1, 1, 2, 1, 1, 1],
#                 [2, 1, 1, 1, 1, 1]])
b = np.array([  [2, 3, 4, 5, 6],
                [1, 2, 1, 6, 7],
                [1, 1, 1, 7, 8]])
# f = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

# d = scipy.signal.convolve2d(b, f, mode='valid')
# print(d.shape)

for d in range(5):
    img = b.copy()
    # translate the image by d
    img = np.roll(img, -d, axis=1)
    if d != 0:
        img[:, -d:] = np.zeros(img[:, -d:].shape)
    print(img)