import json
from numpy import random, dot, heaviside, exp, array
class Neurona(object):
    def __init__(self,entradas=3,epocas=10):
        random.seed(1)
        self.entradas = entradas
        self.pesos = 1 * random.random((self.entradas,1)) -1 # pesos aleatorios
        self.epocas = epocas
        self.resultados = []

    def escalon(self,x):
        return heaviside(x,0)
    
    def entrenar(self,entrada,salidas_esperadas):
        for epoca in range(self.epocas):
            salida = self.pensar(entrada)
            error = salidas_esperadas - salida
            ajuste = dot(entrada.T,error*.2)
            pesos_ant = self.pesos
            self.pesos += (ajuste - 1)
            self.resultados.append({"epoca_"+str(epoca):{"entrada":entrada.tolist(),"salida esperada":salidas_esperadas.tolist(),"pesos":self.pesos.tolist(),"salidas":salida.tolist(),"error":error.tolist(),"pesos nuevos":self.pesos.tolist()}})
    
    def pensar(self,entrada):
        return self.escalon(dot(entrada,self.pesos))