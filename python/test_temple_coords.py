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
data = np.load(datadir + "some_corresp.npz") 
pts1 = data['pts1']
pts2 = data['pts2']

# 2. Run eight_point to compute F
M = max(im1.shape[0], im1.shape[1])
F = sub.eight_point(pts1, pts2, M)
print("Fundamental matrix, F:")
print(F)

###### to visualize results ######
# print(F)
# hlp.displayEpipolarF(im1, im2, F)

# 3. Load points in image 1 from data/temple_coords.npz
data = np.load(datadir + "temple_coords.npz") 
pts1 = data['pts1']

# 4. Run epipolar_correspondences to get points in image 2
pts2 = sub.epipolar_correspondences(im1, im2, F, pts1)

###### to visualize results ######
# hlp.epipolarMatchGUI(im1, im2, F)

# 5. Compute the camera projection matrix P1
# get intrinsic matrices
intrinsics = np.load(datadir + "intrinsics.npz")
K1 = intrinsics['K1']
K2 = intrinsics['K2']

# compute essential matrix
E = sub.essential_matrix(F, K1, K2)
print("Essential matrix, E:")
print(E)

# 6. Use camera2 to get 4 camera projection matrices P2
# compute P using P = K * R * [I|-C]
P1 = np.hstack((K1, np.zeros((3, 1))))    # P1 = K1 * [I|0]
extrinsics = hlp.camera2(E)
P2_0 = np.dot(K2, extrinsics[:, :, 0])
P2_1 = np.dot(K2, extrinsics[:, :, 1])
P2_2 = np.dot(K2, extrinsics[:, :, 2])
P2_3 = np.dot(K2, extrinsics[:, :, 3])

# 7. Run triangulate using the projection matrices
# use computed pts2
pts3d_0 = sub.triangulate(P1, pts1, P2_0, pts2)
pts3d_1 = sub.triangulate(P1, pts1, P2_1, pts2)
pts3d_2 = sub.triangulate(P1, pts1, P2_2, pts2)
pts3d_3 = sub.triangulate(P1, pts1, P2_3, pts2)

# 8. Figure out the correct P2
P2 = None           # correct P2
maxValidPts = 0     # number of valid points in correct P2      
pts3d = None        # correct 3d points

P2s = [P2_0, P2_1, P2_2, P2_3]
pts3ds = [pts3d_0, pts3d_1, pts3d_2, pts3d_3]
ex = [extrinsics[:, :, 0], extrinsics[:, :, 1], extrinsics[:, :, 2], extrinsics[:, :, 3]]

index = 0
for p in range(len(P2s)):
    a = pts3ds[p][:, 2]         # get column of Z values
    validPts = len(a[a>0])      # count # of positive Z values

    # comapre number of valid points to max
    if validPts > maxValidPts:
        maxValidPts = validPts  # update max count
        index = p               # the index containing the correct P2

P2 = P2s[index]                 # update correct P2
pts3d = pts3ds[index]           # the correct 3d points
ex2 = ex[index]                 # the corresponding extrinsic matrix

### calculate reprojection error
# get heterogenous reprojected points
pts3d_homogenous = np.hstack((pts3d, np.ones((pts3d.shape[0], 1)))).T   # 4xN matrix of homogenous coordinates on im1 
reprojected_pts1 = np.dot(P1, pts3d_homogenous).T                       # Nx3 homogenous pts1
reprojected_pts1 = reprojected_pts1/reprojected_pts1[:, -1].reshape((reprojected_pts1.shape[0], 1)) # turning 'z' value of each homogenous coordinate into 1
reprojected_pts1 = reprojected_pts1[:, 0:2]                             # turning into heterogenous coordinate, Nx2
reprojected_pts1 = reprojected_pts1.astype(int)                         # turning to pixel coordinate

# calculate the mean euclidean error
# print(np.linalg.norm(reprojected_pts1 - pts1, axis=1))
reprojection_error = np.mean(np.linalg.norm(reprojected_pts1 - pts1, axis=1))
print("Reprojection Error: " + str(reprojection_error))

# 9. Scatter plot the correct 3D points
ax = plt.axes(projection ="3d")
ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2])
plt.title("3D Reconstruction")
plt.show()

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
R1 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3)
t1 = np.array([0, 0, 0]).reshape(3, 1)
R2 = ex2[0:3, 0:3]
t2 = ex2[:, -1].reshape(3, 1)
np.savez((datadir + "extrinsics.npz"), R1=R1, R2=R2, t1=t1, t2=t2)
