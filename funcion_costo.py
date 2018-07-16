import numpy as np
import matplotlib.pyplot as plt

def graficar2d(i):
    return i+X

def minimizar_costo(t1,t2,X, Y,tam,iteraciones):
    resultados=np.zeros[100]
    for  i in range(iteraciones):
        resultados[i]
        print(resultados[i])

def costo(t1,t2,X, Y,tam):
    aux=np.zeros((tam, 1))
    aux=(t1+(t2*X)-Y)
    aux=np.power(aux,2)
    costo=(np.sum(aux))/(2*tam)
    return costo

datos = np.loadtxt('ex1data1.txt', delimiter=',')
X = np.zeros((datos.shape[0], 1))
Y = np.zeros((datos.shape[0], 1))
print("datos",datos.shape)
X=datos[:,0]
Y=datos[:,1]
#configuracion
aprendizaje=0.2
t1=0.1
t2=0.2

plt.scatter(X,Y)
plt.show()

minimizar_costo(0,1,X,Y,datos.shape[0])
costo(0,1,X,Y,datos.shape[0],100)
#print(X)
#print((Y*2)+10)
#px=np.mean(X)
#py=np.mean(Y)