"""
Extrae la coreografía de referencia de un vídeo: la posición del guante amarillo
en cada frame. 
Se ejecuta UNA vez por vídeo (offline), antes de jugar

Genera: t, x, y, detectado, fps

Controles de la ventana de previsualización:
    s -> guardar las imágenes intermedias del frame actual
    q -> parar
"""
import cv2
import numpy as np
import os


"""
CONFIGURACIÓN
"""
VIDEO_NAME = "coreografia.mp4"     #el vídeo tiene que estar en data/choreographies/

#rango HSV del amarillo del guante (búscalo con get_hsv_color_ranges de la Práctica 1)
LIGHT_YELLOW = (18, 60, 180)
DARK_YELLOW = (40, 255, 255)

#zona del vídeo donde se busca el guante, en fracción del ancho/alto.
#deja fuera los laterales (destellos), arriba (puntuación) y abajo (letras, pictogramas).
ROI = (0.25, 0.15, 0.75, 0.85)     #(x_ini, y_ini, x_fin, y_fin)

MIN_AREA = 30                      #area mínima de la mancha para contar como guante
SHOW = True                        #False para no ver el proyecto

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIDEO_PATH = os.path.join(ROOT, "data", "choreographies", VIDEO_NAME)
OUT_PATH = os.path.join(ROOT, "data", "choreographies", os.path.splitext(VIDEO_NAME)[0] + ".npz")
RESULTS_FOLDER = os.path.join(ROOT, "results", "choreography")


"""
FUNCION DE LA PRÁCTICA 1
"""
def write_image(output_folder: str, img_name: str, img: np.array):
    img_path = os.path.join(output_folder, img_name)
    cv2.imwrite(img_path, img)


def detect_glove(frame):
    """
    Busca el guante amarillo dentro de la ROI
    """
    pasos = {}
    h, w = frame.shape[:2]
    x0, y0 = int(ROI[0] * w), int(ROI[1] * h)
    x1, y1 = int(ROI[2] * w), int(ROI[3] * h)
    roi = frame[y0:y1, x0:x1]
    pasos["0_roi"] = roi

    #1 espacio HSV: Práctica 1
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 2 máscara del amarillo Práctica 1
    mask = cv2.inRange(hsv, LIGHT_YELLOW, DARK_YELLOW)
    pasos["1_mascara"] = mask

    #3 apertura para quitar puntitos sueltos (idea de custom_erode/custom_dilate Práctica 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    pasos["2_morfologia"] = mask

    #4 nos quedamos con la mancha amarilla más grande
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, pasos
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA:
        return None, pasos

    #5 centro de la mancha con los momentos
    M = cv2.moments(c)
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    dibujo = roi.copy()
    cv2.drawContours(dibujo, [c], -1, (0, 255, 0), 2)
    cv2.circle(dibujo, (int(cx), int(cy)), 6, (0, 0, 255), -1)
    pasos["3_guante"] = dibujo

    #normalizado entre 0 y 1 para poder compararlo con la cámara de la jugadora
    return (cx / roi.shape[1], cy / roi.shape[0]), pasos


"""
LECTURA DEL VÍDEO
"""
#basado en read_video() de la Práctica 5, pero frame a frame
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open the video file", VIDEO_PATH)
    exit()

frame_rate = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Vídeo: {n_frames} frames a {frame_rate:.1f} FPS")

xs, ys = [], []
i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    pos, pasos = detect_glove(frame)
    if pos is None:
        xs.append(np.nan)   #sin detección: se rellena al final
        ys.append(np.nan)
    else:
        xs.append(pos[0])
        ys.append(pos[1])

    if SHOW:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (int(ROI[0] * w), int(ROI[1] * h)), (int(ROI[2] * w), int(ROI[3] * h)), (255, 255, 255), 1)
        if pos is not None:
            gx = int(ROI[0] * w + pos[0] * (ROI[2] - ROI[0]) * w)
            gy = int(ROI[1] * h + pos[1] * (ROI[3] - ROI[1]) * h)
            cv2.circle(frame, (gx, gy), 15, (0, 0, 255), 3)
        cv2.putText(frame, f"Frame {i}/{n_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Coreografia", frame)
        cv2.imshow("Mascara", pasos["2_morfologia"])

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            os.makedirs(RESULTS_FOLDER, exist_ok=True)
            for nombre, img in pasos.items():
                write_image(RESULTS_FOLDER, f"{nombre}.jpg", img)
            print("Pasos guardados en", RESULTS_FOLDER)
    i += 1

cap.release()
if SHOW:
    cv2.destroyAllWindows()


# ---------- Rellenar los frames sin guante ----------
xs = np.array(xs)
ys = np.array(ys)
detectado = ~np.isnan(xs)
print(f"Guante detectado en {detectado.sum()} de {len(xs)} frames ({100 * detectado.mean():.1f} %)")

# Interpolación lineal entre la detección anterior y la siguiente
frames = np.arange(len(xs))
if detectado.sum() >= 2:
    xs = np.interp(frames, frames[detectado], xs[detectado])
    ys = np.interp(frames, frames[detectado], ys[detectado])

t = frames / frame_rate   # tiempo en segundos de cada frame


# ---------- Guardar ----------
np.savez(OUT_PATH, t=t, x=xs, y=ys, detectado=detectado, fps=frame_rate)
print("Coreografía guardada en", OUT_PATH)
