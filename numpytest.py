import  numpy as  np

A=np.array([[1,2,3],[4,5,6]])
D=np.array([[4,5],[6,7],[3,5]])
B=np.zeros((A.shape[0],A.shape[1]))
C=np.zeros((A.shape[0],A.shape[1]))

B=A*A

AD=A.dot(D)



#print(B)
print (AD)