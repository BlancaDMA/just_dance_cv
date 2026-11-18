"""
decodificador de secuencias
memoriza hasta 4 patrones consecutivos y concede o deniega el acceso
según si coinciden, en orden, con la contraseña
"""


class SequenceDecoder:

    def __init__(self, password, frames_estable=15):
        """
        password: ["circulo", "triangulo", "circulo", "lineas"]
        frames_estable: frames seguidos que un patrón tiene que verse para contarlo
                        (evita que un frame mal detectado meta un patrón erróneo)
        """
        self.password = password
        self.frames_estable = frames_estable
        self.reset()

    def reset(self):
        """vuelve a empezar"""
        self.secuencia = []           #ya memorizados
        self.candidato = None         # lo que se está viendo ahora
        self.contador = 0             # contador de frames seguidos
        self.esperando_retirar = False
        self.estado = "INTRODUCIENDO" # "INTRODUCIENDO"/"CONCEDIDO"/"DENEGADO"

    def update(self, patron):
        """
        se llama una vez por frame con el patrón detectado
        devuelve el estado actual.
        """
        if self.estado != "INTRODUCIENDO":
            return self.estado

        #si no se ve ninguna tarjeta: se puede enseñar la siguiente
        if patron is None:
            self.esperando_retirar = False
            self.candidato = None
            self.contador = 0
            return self.estado

        #ya se ha guardado la tarjeta que se está enseñando: hay que retirarla
        #antes de que cuente la siguiente
        if self.esperando_retirar:
            return self.estado

        #contar cuántos frames seguidos se ve el mismo patrón
        if patron == self.candidato:
            self.contador += 1
        else:
            self.candidato = patron
            self.contador = 1

        #si se ha visto el tiempo suficiente se guarda
        if self.contador >= self.frames_estable:
            self.secuencia.append(patron)
            self.esperando_retirar = True
            self.candidato = None
            self.contador = 0

            #con la secuencia completa -> comprobar la contraseña
            if len(self.secuencia) == len(self.password):
                if self.secuencia == self.password:
                    self.estado = "CONCEDIDO"
                else:
                    self.estado = "DENEGADO"

        return self.estado