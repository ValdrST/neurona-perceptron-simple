import json,numpy
from Neurona import Perceptron
from sys import argv
from numpy import array
import pprint

def parse_datos(entrada_csv):
    f = open(entrada_csv,"r")
    r = f.read()
    r = r.split("\n")
    entrada = []
    salida = []
    for l in r:
        l = l.split(",")
        entrada.append(l[:len(l)-1])
        salida.append(int(l[len(l)-1]))
    return array(entrada).astype(int) , array(salida).astype(int)


if __name__ == "__main__":
    num_entradas = int(argv[3])
    neurona = Perceptron(num_entradas,epocas=10)
    entrada, salida = parse_datos(argv[1])
    neurona.train(entrada,salida)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(neurona.resultados)
    f = open(argv[2],"w")
    f.write(json.dumps(neurona.resultados))
    print("Entrada: {0} Salida: {1}".format(numpy.array([0,0,0,0]).tolist(),neurona.pensar(numpy.array([0,0,0,0]))))
    print("Entrada: {0} Salida: {1}".format(numpy.array([1,1,1,1]).tolist(),neurona.pensar(numpy.array([1,1,1,1]))))