import numpy as np

class Perceptron(object):

    def __init__(self, num_entradas, epocas=100, razon_aprendizaje=0.01):
        self.epocas = epocas
        self.razon_aprendizaje = razon_aprendizaje
        self.pesos = np.zeros(num_entradas + 1)
        self.resultados = []
           
    def pensar(self, entradas):
        self.suma = np.dot(entradas, self.pesos[1:]) + self.pesos[0]
        if self.suma > 0:
          return 1
        return 0            

    def entrenar(self, entradas_entrenamiento, salidas):
        for _ in range(self.epocas):
            self.pesos_ant = self.pesos.tolist()
            for entrada, salida in zip(entradas_entrenamiento, salidas):
                resultado = self.pensar(entrada)
                self.pesos[1:] += self.razon_aprendizaje * (salida - resultado) * entrada
                self.pesos[0] += self.razon_aprendizaje * (salida - resultado)
            self.pesos_act = self.pesos.tolist()
            if self.calcular_estabilidad() == True:
                break
            self.resultados.append({"epoca":_+1,"pesos anteriores":self.pesos_ant,"pesos actuales":self.pesos_act})
    
    def calcular_estabilidad(self):
        if self.pesos_ant == self.pesos_act:
            return True
        else:
            return False