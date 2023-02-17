import numpy as np
from submission import eight_point
import cv2

im1 = np.random.randint(0, 5, (8, 8))
print(im1)
print('-----------')
w = 2
for i in range(im1.shape[1]):
        x = im1[:, i]    # one point in im1                   3x1 point
        w1 = im1[max(0, x[1] - w): min(im1.shape[1], x[1] + w) + 1, 
                 max(0, x[0] - w): min(im1.shape[0], x[0] + w) + 1]
        print(w1)
        print('-----------')

# a = np.array([[2, 7],
#             [2, 3],
#             [4, 2],
#             [4, 3],
#             [6, 7],
#             [6, 2],
#             [3, 3],
#             [5, 6],
#             [7, 6]])
# b = np.array([[4, 3],
#             [5, 6],
#             [6, 4],
#             [4, 7],
#             [2, 5],
#             [5, 4],
#             [2, 6],
#             [2, 3],
#             [6, 7]])

# # a = np.array([1, 2]).reshape((1, 2))
# # b = np.array([3, 4]).reshape((1, 2))
# print(cv2.findFundamentalMat(a, b, cv2.FM_8POINT)[0])
# f = eight_point(a, b, 8)
# f = f/f[2,2]
# print(f)