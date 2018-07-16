import  numpy as np

datos=np.loadtxt('ex1data1.txt',delimiter=',')
X=np.zeros((datos.shape[0],1))
Y=np.zeros((datos.shape[0],1))
X=datos[:,0]
Y=datos[:,1]

px=np.mean(X)
py=np.mean(Y)

print(px,py)