import cv2
import numpy as np

def seleccionar_color_guante():
    

    camara = cv2.VideoCapture(0) 
    #Usamos el video la camara hasta que veamos el guante y damos al espacio
    while True:
        ret, frame = camara.read()  #hasta que paremos va guardando el frame y fotos
        if not ret:  #si no consigue hacer las fotos para
            break
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 32:  #termina si pulsas el espacio
            break

    if not ret:
        print("No se pudo encontrar la imagen")
        return 

    # de la practica 5 apartado d sacamos el como seleccionar una parte de la imagen
    x, y, w, h = cv2.selectROI('Frame', frame, False)   
    crop = frame[y:y+h, x:x+w].copy()     #recorta la parte seleccionada
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)   #convertimos a HSV

    #Ahora vamos a calcular el color del rectangulo seleccionado
    H, S, V = np.median(hsv_crop, axis=(0, 1))
    #calculamos el max y min para que dependiendo de la iluminacion lo siga captando
    hsv_min = (max(0, int(H)-10), max(0, int(S)-60), max(0, int(V)-60)) 
    hsv_max = (min(179, int(H)+10), 255, 255)

    camara.release()
    cv2.destroyAllWindows()
    return hsv_min, hsv_max


def seguir_guante(hsv_min, hsv_max):
    """
    Ahora que tenemos en max y min que puede tomar el color tenemos que hacer el rectangulo que seguira al guante
    """
    camara = cv2.VideoCapture(0) 
    while True:
        ret, frame = camara.read()  #hasta que paremos va guardando el frame y fotos
        if not ret:  #si no consigue hacer las fotos para
            break

        #Usando la practica 1 parte b, hacemos la segmentacion por color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, hsv_min, hsv_max)  #Nos quedamos solo con el color del guante, nos sale blanco lo que tiene el color naranja y negro el resto 
        
        #buscamos los contornos de las zonas que tienen el color del guante
        contornos, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

        if len(contornos) > 0:  #si hemos encontrado contorno

            #si detecta varios contornos, es decir varias manchas del color del guante se queda con la mayor asumiendo que sera el guante
            contorno = max(contornos, key=cv2.contourArea)

            # ponemos limite 300 para asegurarnos que la mancha detectada es el guante
            if cv2.contourArea(contorno) > 300:

                x, y, w, h = cv2.boundingRect(contorno) #rectangulo que rodea el guante
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  #dibujamos el rectangulo

        cv2.imshow("Tracker", frame)  #mostramos la imagen (en este caso video)
 
        #Pulsamos enter para salir
        if cv2.waitKey(1) & 0xFF == 13:
            break

    camara.release()
    cv2.destroyAllWindows()






    

if __name__ == "__main__":
    hsv_min , hsv_max = seleccionar_color_guante()
    seguir_guante(hsv_min , hsv_max)
