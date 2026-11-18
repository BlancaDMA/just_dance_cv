"""
Sistema de seguridad: une los dos codigos (detector de patrones + decodificador de secuencias)
Controles:
    r -> resetear la secuencia
    s -> guardar las imágenes intermedias del frame actual (para el informe)
    q -> salir
"""
import cv2
import numpy as np
import os
import time

from pattern_detector import detect_pattern, write_image
from sequence_decoder import SequenceDecoder


"""
CONFIGURACIÓN
"""
PASSWORD = ["circulo", "triangulo", "circulo", "lineas"]
ROI_SIZE = 260                         #lado del recuadro central
CALIB_PATH = "data/calibration/calib.npz"
RESULTS_FOLDER = "results/security"

"""
PARAMETROS DE CLAIBRACIÓN
"""
calib = np.load(CALIB_PATH)
intrinsics = calib["intrinsics"]
dist_coeffs = calib["dist_coeffs"]

#abrir la cámara (con los datos obtenidos de la calibración)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

decoder = SequenceDecoder(PASSWORD) #creo el decoder con la clase
tiempo_denegado = None
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #espejo y luego corrección de distorsión
    frame = cv2.flip(frame, 1)
    frame = cv2.undistort(frame, intrinsics, dist_coeffs)

    #recuadro central donde se pone la tarjeta
    h, w = frame.shape[:2]
    x0 = w // 2 - ROI_SIZE // 2
    y0 = h // 2 - ROI_SIZE // 2
    roi = frame[y0:y0 + ROI_SIZE, x0:x0 + ROI_SIZE]

    #detección y decodificación
    patron, pasos = detect_pattern(roi)
    estado = decoder.update(patron)

    #si se deniega, se muestra el mensaje 2 segundos y se resetea solo
    if estado == "DENEGADO":
        if tiempo_denegado is None:
            tiempo_denegado = time.time()
        elif time.time() - tiempo_denegado > 2:
            decoder.reset()
            tiempo_denegado = None

    """
    DIBUJO
    """
    color = (0, 255, 0) if patron is not None else (255, 255, 255)
    cv2.rectangle(frame, (x0, y0), (x0 + ROI_SIZE, y0 + ROI_SIZE), color, 2)
    cv2.putText(frame, f"Patron: {patron}", (x0, y0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    #barra de progreso del patrón que se está viendo
    if decoder.contador > 0:
        progreso = int(ROI_SIZE * decoder.contador / decoder.frames_estable)
        cv2.rectangle(frame, (x0, y0 + ROI_SIZE + 5), (x0 + progreso, y0 + ROI_SIZE + 12), (0, 255, 0), -1)

    #un círculo por cada patrón de la contraseña
    for i in range(len(PASSWORD)):
        centro = (20 + i * 30, h - 20)
        if i < len(decoder.secuencia):
            cv2.circle(frame, centro, 10, (0, 255, 255), -1)
        else:
            cv2.circle(frame, centro, 10, (255, 255, 255), 2)

    if estado == "CONCEDIDO":
        cv2.putText(frame, "ACCESO CONCEDIDO", (130, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    elif estado == "DENEGADO":
        cv2.putText(frame, "ACCESO DENEGADO", (140, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    #FPS
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Sistema de seguridad", frame)

    """
    TECLAS
    """
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        decoder.reset()
        tiempo_denegado = None
    elif key == ord("s"):
        #secuencia de transformaciones de la imagen
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        write_image(RESULTS_FOLDER, "0_original.jpg", roi)
        for nombre, img in pasos.items():
            write_image(RESULTS_FOLDER, f"{nombre}.jpg", img)
        print("Pasos guardados en", RESULTS_FOLDER)

cap.release()
cv2.destroyAllWindows()
