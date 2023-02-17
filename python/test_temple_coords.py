import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt
import cv2 as cv

# 1. Load the two temple images and the points from data/some_corresp.npz
datadir = '../data/'
resultsdir = '.'
im1 = cv.imread(datadir + 'im1.png')
im2 = cv.imread(datadir + 'im2.png')
data = np.load("../data/some_corresp.npz") 
pts1 = data['pts1']
pts2 = data['pts2']

# 2. Run eight_point to compute F
M = max(im1.shape[0], im1.shape[1])
F = sub.eight_point(pts1, pts2, M)

###### to visualize results ######
# hlp.displayEpipolarF(im1, im2, F)

# 3. Load points in image 1 from data/temple_coords.npz
data = np.load("../data/temple_coords.npz") 
pts11 = data['pts1']

# 4. Run epipolar_correspondences to get points in image 2
pts22 = sub.epipolar_correspondences(im1, im2, F, pts11)
T = sub.eight_point(pts11, pts22, M)
print(T)

###### to visualize results ######
hlp.epipolarMatchGUI(im1, im2, T)

# 5. Compute the camera projection matrix P1

# 6. Use camera2 to get 4 camera projection matrices P2

# 7. Run triangulate using the projection matrices

# 8. Figure out the correct P2

# 9. Scatter plot the correct 3D points

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
