from __future__ import division, print_function
import numpytest as np
import random
import sklearn
from sklearn.datasets.samples_generator import make_regression
import pylab
from scipy import stats

def gradientDescent2(Lambda, x, y, max_iter = 10000):
    """
    Esta función permite calcular el parametro «theta» para la aplicación afín que mejor se ajusta
    a los datos (x, y).
    :param Lambda: Tasa de aprendizaje.
    :param x: Valores independientes de entrada.
    :param y: Valores dependientes de salida.
    :param epsilon: Umbral de parada.
    :param
    """
    k = x.shape[0] # número de muestras.
    theta = np.ones(x.shape[1]+1)
    X = np.c_[ np.ones(x.shape[0]), x]
    Y = y
    for iter in range(0, max_iter):
        hypothesis = np.dot(X, theta)
        loss = hypothesis - Y
        gradient = np.dot(X.T, loss) / k
        theta = theta - alpha * gradient  # update
    return theta
    pass

if __name__ == '__main__':
    x, y = make_regression(n_samples = 200, n_features = 1, n_informative = 1, random_state = 0, noise = 35)
    m, n = np.shape(x)
    alpha = 0.001 # learning rate
    theta = gradientDescent2(alpha, x, y, 1000)
    # plot
    for i in range(x.shape[1]):
        y_predict = theta[0] + theta[1]*x
    pylab.plot(x, y, 'o')
    pylab.plot(x, y_predict,'k-')
    pylab.show()
    print(theta)
    print('¡Listo!')