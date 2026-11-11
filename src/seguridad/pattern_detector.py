"""
Detector de patrones del sistema de seguridad.
Reconoce 4 tarjetas en negro sobre fondo blanco: "circulo", "triangulo", "cuadrado" y "lineas"
"""
import cv2
import numpy as np
import os
from typing import List, Optional


"""
Parametros
"""
GAUSS_SIGMA = 1
MIN_AREA = 1500    #area mínima (px) para que un contorno cuente como figura
CIRCULARITY_MIN = 0.75      #4*pi*A/P^2 -> 1 en un círculo perfecto, 0.785 en un cuadrado
MIN_LINES = 4               #nº mínimo de segmentos de Hough para decir que son lineas


"""
FUNCION NECESARIA DE LA PRACTICA 1
"""
def write_image(output_folder: str, img_name: str, img: np.array):
    img_path = os.path.join(output_folder, img_name)
    cv2.imwrite(img_path, img)

"""
FUNCIONES NECESARIAS DE LA PRACTICA 3
"""
def show_image(img: np.array, img_name: str = "Image"):
    # Show the image
    cv2.imshow(img_name, img)
    # Wait until any key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gaussian_blur(img: np.array, sigma: float, filter_shape: Optional[List] = None, verbose: bool = False) -> np.array:
    # TODO If not given, compute the filter shape 
    if filter_shape == None:
        filter_l = 2 * int(np.ceil(4 * sigma)) + 1
    else:
        filter_l = filter_shape[0]

    medio = filter_l//2
    
    # TODO Create the filter coordinates matrices
    y, x = np.mgrid[-medio:medio + 1, -medio:medio + 1]
    
    # TODO Define the formula that goberns the filter
    formula = 1/(2*np.pi*sigma**2)*np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian_filter = formula / formula.sum() #normalizar 
    
    # TODO Process the image
    gb_img = cv2.filter2D(img, -1, gaussian_filter)
    
    if verbose:
        show_image(img=gb_img, img_name=f"Gaussian Blur: Sigma = {sigma}")
    
    return gaussian_filter, gb_img.astype(np.uint8)

def binarize(gray_img):
    """
    En esta funcion la implementamos parecida en la practica 3, pero esta invertida
    - Lo negro pasa a ser blanco (para findContours)
    - usamos cv2.threshold para que sea mas optima
    - se adapta a la luz de la habitacion
    """
    _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary_img

def clean_mask(binary_img):
    """
    apertura (erosión + dilatación), la misma idea que custom_erode() y custom_dilate()
    de la práctica 3, pero con la función de OpenCV para que vaya en tiempo real
    tambien quita los puntitos de ruido sin deformar las figuras
    """
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

"""
FUNCION NECESARIA DE LA PRACTICA 4
"""
def draw_lines(img, lines, color, thickness):
    if lines is None:
        return img
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


def detect_pattern(roi):
    """
    detecta qué tarjeta hay dentro del recuadro
    devuelve el patrón ("circulo", "triangulo", "cuadrado", "lineas" o None) y un diccionario
    """
    pasos = {}
    h, w = roi.shape[:2]

    #1 paso: escala de grises: las tarjetas son en blanco y negro (prctica 1)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    pasos["1_gris"] = gray

    #2 paso: filtro gaussiano para quitar el ruido de la cámara (practica 3)
    _, blurred = gaussian_blur(gray, GAUSS_SIGMA)
    pasos["2_gauss"] = blurred

    #3 oaso: sinarización invertida: la figura negra pasa a blanco (practica 3)
    binary = binarize(blurred)
    pasos["3_binaria"] = binary

    #4 paso: apertura (erosión + dilatación) para borrar puntitos de ruido (practica 3)
    mask = clean_mask(binary)
    pasos["4_morfologia"] = mask

    #5 paso: contornos de las zonas blancas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    figuras = []
    for c in contours:
        #se descarta el ruido
        if cv2.contourArea(c) < MIN_AREA:
            continue
        #se descartan los que tocan el borde del recuadro
        x, y, cw, ch = cv2.boundingRect(c)
        if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1:
            continue
        figuras.append(c)

    dibujo = roi.copy()
    cv2.drawContours(dibujo, figuras, -1, (0, 255, 0), 2)
    pasos["5_contornos"] = dibujo

    patron = None

    #6 paso: varias figuras: comprobar si son líneas con Canny + Hough (practica 4)
    if len(figuras) >= 2:
        edges = cv2.Canny(blurred, 50, 150)   # umbrales de la Práctica 4
        pasos["6_canny"] = edges

        #parámetros que usamos en la practica 4
        rho = 1
        theta = np.pi / 180
        threshold = 50
        min_line_length = 50
        max_line_gap = 10
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

        hough = draw_lines(roi.copy(), lines, (0, 0, 255), 2)
        pasos["7_hough"] = hough

        #cada franja tiene 2 bordes largos, asi que deben salir varios segmentos
        if lines is not None and len(lines) >= MIN_LINES:
            patron = "lineas"

    #7 paso: una sola figura: clasificar por su forma
    elif len(figuras) == 1:
        c = figuras[0]
        area = cv2.contourArea(c)
        perimetro = cv2.arcLength(c, True)

        #simplificar el contorno a un polígono y contar sus vértices
        approx = cv2.approxPolyDP(c, 0.04 * perimetro, True)
        vertices = len(approx)

        #circularidad: 1 en un círculo, ~0.785 en un cuadrado, ~0.6 en un triángulo
        circularidad = 4 * np.pi * area / (perimetro ** 2)

        if vertices == 3:
            patron = "triangulo"
        elif vertices == 4:
            patron = "cuadrado"
        elif circularidad > CIRCULARITY_MIN:
            patron = "circulo"

    return patron, pasos