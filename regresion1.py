from __future__ import division, print_function
import numpytest as np
import random
import sklearn
from sklearn.datasets.samples_generator import make_regression
import pylab
from scipy import stats


def gradientDescent(Lambda, x, y, epsilon, max_iter=10000):
    """
    Esta función permite calcular el parametro «theta» para la aplicación afín que mejor se ajusta
    a los datos (x, y).
    :param Lambda: Tasa de aprendizaje.
    :param x: Valores independientes de entrada.
    :param y: Valores dependientes de salida.
    :param epsilon: Umbral de parada.
    :param max_iter: Numero de veces que se repite la iteración.
    """

    converged = False
    iter = 0
    k = x.shape[0]  # tamaño de la muestra.

    # Iniciamos el parametro theta.
    alpha = np.random.random(x.shape[1])
    beta = np.random.random(x.shape[1])

    # Error total, e(theta_t)
    temp_et = sum([(y[i] - alpha - beta * x[i]) ** 2 for i in range(k)])

    # Ciclo de iteraciones
    while not converged:
        # Para el conjunto muestral se calcula el gradiente de la funcion e(theta)
        grad0 = 1.0 / k * sum([(y[i] - alpha - beta * x[i]) for i in range(k)])
        grad1 = 1.0 / k * sum([(y[i] - alpha - beta * x[i]) * x[i] for i in range(k)])

        # Guardamos temporamente el parametro theta
        temp_alpha = alpha + Lambda * grad0
        temp_beta = beta + Lambda * grad1

        # Actualizamos el parametro theta
        alpha = temp_alpha
        beta = temp_beta

        # Error cuadrático medio.
        et = sum([(y[i] - alpha - beta * x[i]) ** 2 for i in range(k)])

        if abs(temp_et - et) <= epsilon:
            print('La iteración converge: ', iter)
            converged = True

        temp_et = et  # Actualización del error
        iter += 1  # Actualización del numero de iteraciones

        if iter == max_iter:
            print('¡Máximo de iteraciones excedido!')
            converged = True

    return alpha, beta
    pass


if __name__ == '__main__':
    x, y = make_regression(n_samples=200, n_features=1, n_informative=1, random_state=0, noise=35)
    print('Tamaño del conjunto de prueba: ', x.shape, y.shape)

    Lambda = 0.001  # Tasa de aprendizaje
    epsilon = 0.01  # Umbra de convergencia

    # Llamamos la función gradientDescent para obtener el parametro theta.
    alpha, beta = gradientDescent(Lambda, x, y, epsilon, max_iter=10000)
    print('alpha = ', alpha, ', beta = ', beta)

    # Comprobando los resultados de nuestro algoritmo con scipy linear regression
    #slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x[:, 0], y)
    #print('Intercepto = ', intercept, ', Pendiente = ', slope)

    # plot
   # for i in range(x.shape[0]):
   #     y_predict = alpha + beta * x

    #pylab.plot(x, y, 'o')
    #pylab.plot(x, y_predict, 'k-')
    #pylab.show()
    print('¡Listo!')