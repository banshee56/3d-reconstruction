import numpy as np
from submission import eight_point
import cv2


import matplotlib.pyplot as plt
import numpy as np

# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


# Creating dataset
z = np.random.randint(100, size =(50))
x = np.random.randint(80, size =(50))
y = np.random.randint(60, size =(50))

print(x.shape)
# Creating figure
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

# Creating plot
ax.scatter3D(x, y, z, color = "green")
plt.title("simple 3D scatter plot")

# show plot
plt.show()


# w = 2
# im1 = np.random.randint(0, 5, (5, 5, 3))
# # print(im1)
# print(im1[-2:2, 1:2, :])       # returns empty array if out of bounds
# print(im1)
# im1_p = np.pad(im1, pad_width=[(w, w),(w, w),(0, 0)], mode='constant')
# print(im1_p)
# point = [0,2]

# # print(im1_p[:, 1:4, 1:4])

# w1 = im1_p[:,
#         int(point[1]): int(point[1] + 2*w) + 1,  # 3x3 shape with 3 rgb channels
#         int(point[0]): int(point[0] + 2*w) + 1]
# print('window:')
# print(w1)
# print(w1.shape)



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