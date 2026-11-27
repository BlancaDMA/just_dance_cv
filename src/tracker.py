import cv2
import numpy as np


def seleccionar_color_guante(frame, caja):  
    x, y, w, h = caja  

    if w == 0 or h == 0:  #comprobamos que se ha seleccionado una zona
        return  

    # De la practica 5 apartado d sacamos el como recortar una parte de la imagen
    crop = frame[y:y+h, x:x+w].copy()     #recorta la parte seleccionada
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)   #convertimos a HSV

    #Ahora vamos a calcular el color del rectangulo seleccionado
    H, S, V = np.median(hsv_crop, axis=(0, 1))
    #calculamos el max y min para que dependiendo de la iluminacion lo siga captando
    hsv_min = (max(0, int(H)-10), max(0, int(S)-60), max(0, int(V)-60))
    hsv_max = (min(179, int(H)+10), 255, 255)

    return hsv_min, hsv_max



def detectar_guante(frame, hsv_min, hsv_max):  

    #Usando la practica 1 parte b, hacemos la segmentacion por color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, hsv_min, hsv_max)  #Nos quedamos solo con el color del guante, nos sale blanco lo que tiene el color naranja y negro el resto

    #Usando la apertura de la practica 3 quitamos las manchas pequeñas de ruido
    kernel = np.ones((3, 3), np.uint8)  
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)  

    #buscamos los contornos de las zonas que tienen el color del guante
    contornos, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

    if len(contornos) > 0:  #si hemos encontrado contorno

        #si detecta varios contornos, es decir varias manchas del color del guante se queda con la mayor asumiendo que sera el guante
        contorno = max(contornos, key=cv2.contourArea)

        # ponemos limite 50 para asegurarnos que la mancha detectada es el guante, un valor menor para que al alejarnos nos identifique el guante
        if cv2.contourArea(contorno) > 50:  

            x, y, w, h = cv2.boundingRect(contorno) #rectangulo que rodea el guante
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  #dibujamos el rectangulo

            #Calculamos el centro del guante con momentos
            M = cv2.moments(contorno)  
            if M["m00"] != 0:  #Necsitamos que el area no sea 0
                #Coordenadas del centro:
                cx = int(M["m10"] / M["m00"])  
                cy = int(M["m01"] / M["m00"])  
                return np.array([cx, cy])  

    return None  



