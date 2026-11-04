"""
Calibración offline de la cámara
Sigue los mismos pasos que el Apartado A de la Práctica 2
"""
import cv2
from typing import List
import numpy as np
import copy
import glob
import os

"""
FUNCIONES DE LA PRACTICA 2
"""
def load_images(filenames: List) -> List:
    imgs = []
    for filename in filenames:
        img = cv2.imread(filename)
        imgs.append(img)
    return imgs

def show_image(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def write_image(filename, img):
    cv2.imwrite(filename, img)

def get_chessboard_points(chessboard_shape, dx, dy):
    cols, rows = chessboard_shape

    return np.array(
        [[dx * col, dy * row, 0]
         for row in range(rows)
         for col in range(cols)],
        dtype=np.float32)


"""
PARÁMETROS DEL TABLERO
"""
chessboard_shape = (7, 10)   #esquinas interiores
square_size = 20           

"""
MISMOS PASOS QUE LA PRÁCTICA 2
"""
#carga de imágenes
imgs_path = sorted(glob.glob("data/calibration_images/*.jpg"))
imgs = load_images(imgs_path)
print("Imágenes cargadas:", len(imgs))


#detección de esquinas
corners = [cv2.findChessboardCorners(img, chessboard_shape) for img in imgs]

corners_copy = copy.deepcopy(corners)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

imgs_gray = []
for img in imgs:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgs_gray.append(gray)

corners_refined = []
for i in range(len(imgs_gray)):
    if corners_copy[i][0]:
        refined = cv2.cornerSubPix(imgs_gray[i], corners_copy[i][1], (6, 6), (-1, -1), criteria)
        corners_refined.append((corners_copy[i][0], refined))

print("Imágenes con el tablero detectado:", len(corners_refined))


#comprobar las detecciones
imgs_copy = copy.deepcopy(imgs)
imgs_copy = [cv2.drawChessboardCorners(img, chessboard_shape, corner[1], corner[0])
             for img, corner in zip(imgs_copy, corners_copy) if corner[0]]

for img in imgs_copy:
    show_image("Corners", img)  


#puntos 3D del tablero
chessboard_points = [get_chessboard_points(chessboard_shape, square_size, square_size) for _ in corners_refined]


#calibración
valid_corners = []
for valid, refined in corners_refined:
    valid_corners.append(refined)
valid_corners = np.asarray(valid_corners, dtype=np.float32)

imageSize = imgs_gray[0].shape[::-1]
rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(chessboard_points, valid_corners, imageSize, None, None)

extrinsics = list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs))

print("Intrinsics:\n", intrinsics)
print("Distortion coefficients:\n", dist_coeffs)
print("Root mean squared reprojection error:\n", rms)


#guardar los parámetros (crea un archivo)
#para que el resto del proyecto cargará este archivo en vez de recalibrar cada vez
os.makedirs("data/calibration", exist_ok=True)
np.savez("data/calibration/calib.npz", intrinsics=intrinsics, dist_coeffs=dist_coeffs, rms=rms)
print("Parámetros guardados en data/calibration/calib.npz")


"""
#corrección de distorsión
# Imagen original y corregida lado a lado, para el informe
undistorted = cv2.undistort(imgs[0], intrinsics, dist_coeffs)
comparison = np.hstack((imgs[0], undistorted))
write_image("data/calibration/undistort_demo.jpg", comparison)
show_image("Original vs corregida", comparison)
"""