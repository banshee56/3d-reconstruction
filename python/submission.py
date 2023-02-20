"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
import helper as hlp

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):
    # normalize points
    n = pts1.shape[0]

    # scale coordinates by M
    s = 1/M                         # scale factor
    pts1 = s * pts1                 # normalizing coordinates
    pts2 = s * pts2
    T = np.array(  [[s, 0, 0],      # the transformation matrix
                    [0, s, 0], 
                    [0, 0, 1]])     
    
    # compute A
    A = np.zeros(shape=(n, 9))
    for p in range(n):
        x0, y0 = pts1[p, 0], pts1[p, 1]
        x1, y1 = pts2[p, 0], pts2[p, 1]
        A[p] = [x0*x1, x0*y1, x0, y0*x1, y0*y1, y0, x1, y1, 1]

    # commpute F
    u, d, vt = np.linalg.svd(A)             # svd of A
    f = vt[-1]                              # entries of F are the elements of column of V corresponding to the least singular value
    F = np.reshape(f, newshape=(3, 3))      # compute F

    # Enforce rank 2 constraint on F
    U, D, Vt = np.linalg.svd(F)             # svd of F
    D[-1] = 0                               # set third singular value to 0
    F = np.dot(np.dot(U, np.diag(D)), Vt)   # recompute F

    # refine the solution by using local minimization
    F = hlp.refineF(F, pts1, pts2)

    # unnormalize/unscale F
    F_unnorm = np.dot(np.dot(T.T, F), T)
    return F_unnorm


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):
    # use fundamental matrix to estimate the corresponding epipolar line l'
    pts1_homogenous = np.hstack((pts1, np.ones((pts1.shape[0], 1)))).T    # 3xN matrix of homogenous coordinates on im1
    l2 = np.dot(F, pts1_homogenous)                                       # 3xN matrix of corresponding epipolar lines
    pts2 = np.zeros_like(pts1)

    w = 5      # window/patch 'radius', padding width
    im1_p = np.pad(im1, pad_width=[(w, w),(w, w),(0, 0)], mode='constant')
    im2_p = np.pad(im2, pad_width=[(w, w),(w, w),(0, 0)], mode='constant')

    # for each point in im1, generate a set of candidate points in the second image
    # x values shifted due to padding
    x_cand = np.arange(0, im2.shape[1], 1)                  # x values in im2
    x_cand = np.reshape(x_cand, (x_cand.shape[0], 1))       # Mx1

    # go through each point in pts1, an Nx2 array
    for i in range(pts1.shape[0]):
        point = pts1[i]                                     # one point in pts1, shape 1x2
        epipolar_correspondence = np.array([0, 0])          # the corresponding point of 'point'
        correspondence_score = float('inf')                 # the score of epipolar_correspondence

        # window in im1
        # indices shifted by +w both ways due to padding
        w1 = im1_p[int(point[1]): int(point[1] + 2*w) + 1,  # (2w+1)x(2w+1) window with 3 rgb channels
                int(point[0]): int(point[0] + 2*w) + 1, :]

        l = l2[:, i]                                        # corresponding line in im2

        y_cand = (- l[0]*x_cand - l[2])/l[1]                # corresponding y val for each x val on line in im2, shifted due to padding
        y_cand = np.reshape(y_cand, (y_cand.shape[0], 1))   # Mx1
        cand = np.hstack((x_cand, y_cand))                  # Mx2

        # go through each candidate point
        for j in range(cand.shape[0]):
            cand_point = cand[j]

            # window in im2
            # points are shifted by w due to padding
            w2 = im2_p[int(cand_point[1]): int(cand_point[1] + 2*w) + 1,    # (2w+1)x(2w+1) window with 3 rgb channels
                    int(cand_point[0]): int(cand_point[0] + 2*w) + 1, :]
            
            # # ignore patches that don't align
            # if w2.shape != w1.shape:
            #     continue

            # compute similarity
            # get the manhattan distance between the points
            score = np.sum((w1-w2)**2)

            # if score is new max
            if score < correspondence_score:
                epipolar_correspondence[0] = cand_point[0]
                epipolar_correspondence[1] = cand_point[1]
                correspondence_score = score

        # set up corresponding point in pts2
        pts2[i] = np.reshape(epipolar_correspondence, (1, 2))

    # point with highest score is treated as epipolar correspondence
    return pts2


"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    # F = K2^(-T) * E * K1^(-1)
    # E = K2^(T) * E * K1
    E = np.dot(np.dot(K2.T, F), K1)
    return E


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    pts3d = np.zeros((pts1.shape[0], 3))

    for p in range(pts1.shape[0]):
        # camera 1 variables
        x1 = pts1[p][0]
        y1 = pts1[p][1]
        p1_1 = P1[0].reshape(1,4)
        p1_2 = P1[1].reshape(1,4)
        p1_3 = P1[2].reshape(1,4)

        # camera 2 variables
        x2 = pts2[p][0]
        y2 = pts2[p][1]
        p2_1 = P2[0].reshape(1,4)
        p2_2 = P2[1].reshape(1,4)
        p2_3 = P2[2].reshape(1,4)

        # compute A
        A = np.array([y1 * p1_3 - p1_2,
                      p1_1 - x1 * p1_3,

                      y2 * p2_3 - p2_2,
                      p2_1 - x2 * p2_3]).reshape((4, 4))

        # compute SVD of A
        U, D, Vt = np.linalg.svd(A)
        X = Vt[-1]      # ROW of Vt corresponding to smallest singular value
        X = X/X[-1]
        pts3d[p] = X[0:3]

    return pts3d


"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    pass


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    # replace pass by your implementation
    pass


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    pass


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation
    pass


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # replace pass by your implementation
    pass
