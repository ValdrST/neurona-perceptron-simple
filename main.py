import json,numpy
from Neurona import Neurona
from sys import argv

def parse_datos(entrada_csv):
    f = open(entrada_csv,"r")
    r = f.read()
    r = r.split("\n")
    entrada = []
    salida = [[]]
    for l in r:
        l = l.split(",")
        entrada.append(l[:4])
        salida[0].append(int(l[4]))
    return numpy.array(entrada).astype(int) , numpy.array(salida).T


if __name__ == "__main__":
    neurona = Neurona(entradas=4,epocas=10)
    entrada, salida = parse_datos(argv[1])
    neurona.entrenar(entrada,salida)
    print(neurona.resultados)
    f = open(argv[2],"w")
    f.write(json.dumps(neurona.resultados))
    print(neurona.pensar(numpy.array([1,1,1,1])))
    print(neurona.pensar(numpy.array([1,0,1,1])))
    