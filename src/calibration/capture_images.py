"""
captura de imágenes del tablero para la calibración offline
"""
import cv2
import os


#función de la Práctica 2
def write_image(filename, img):
    cv2.imwrite(filename, img)


#carpeta donde se guardan las imágenes
output_folder = "data/calibration_images"
os.makedirs(output_folder, exist_ok=True)

#abrir la cámara (igual que en test.py)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 0 #contador para los nombres de las imágenes
print("ESPACIO para guardar, q para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  #si no ponemos esto sale la camara invertida y queremos efecto espejo

    cv2.imshow("Captura calibracion", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        filename = f"{output_folder}/calibration_{count:03d}.jpg"
        write_image(filename, frame)
        print("Guardada:", filename)
        count += 1
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
