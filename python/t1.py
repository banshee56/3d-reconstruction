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
b = np.array([  [2, 3, 4],
                [1, 2, 1],
                [1, 1, 1]])
f = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

d = scipy.signal.convolve2d(b, f, mode='valid')
print(d.shape)