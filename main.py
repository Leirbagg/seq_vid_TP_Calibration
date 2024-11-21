import numpy as np
import cv2
import os
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')

# img = cv2.imread('1.jpg')
# height, width = img.shape[:2]
# scale_percent = 10  # Réduire à 50% de la taille originale
# new_width = int(width * scale_percent / 100)
# new_height = int(height * scale_percent / 100)
# resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
# cv2.imshow('Image redimensionnée', resized_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

for fname in images:
    print(fname)
    img = cv2.imread(fname)

    height, width = img.shape[:2]
    scale_percent = 10 
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
    print(ret)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(resized_img, (7,6), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)
        cv2.imwrite(fname+'_calib.png', img)


cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
fx = mtx[0, 0]
fy = mtx[1, 1]
focale_m_x = fx * 1.4e-6
focale_m_y = fy * 1.4e-6

print(f"ret : {ret}\nmtx : {mtx}\ndist : {dist}\nrvecs : {rvecs}")

# Afficher la focale
print("Focale en pixels :")
print(f"fx = {fx}")
print(f"fy = {fy}")

print("\nFocale en mètres :")
print(f"Focale x = {focale_m_x} m")
print(f"Focale y = {focale_m_y} m")

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

mean_error /= len(objpoints)
print(f"Erreur moyenne de reprojection : {mean_error}")


### BASICS ###

# def draw(img, corners, imgpts):
#     corner = tuple(corners[0].ravel())
#     img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
#     img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
#     img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
#     return img

# axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

### END BASICS ###

### CUBE

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img

axis = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0],
                   [0,0,-1],[0,1,-1],[1,1,-1],[1,0,-1] ])

### END CUBE ###

for fname in glob.glob('*.jpg'):
    img = cv2.imread(fname)

    height, width = img.shape[:2]
    scale_percent = 10 
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
    print(ret)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(resized_img,corners2,imgpts)
        cv2.imshow('img',img)
        cv2.waitKey(500)
        cv2.imwrite(fname+'_cube.png', img)

cv2.destroyAllWindows()