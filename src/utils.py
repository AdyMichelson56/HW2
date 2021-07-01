import numpy as np
import math
import matplotlib.pyplot as plt

def report(iterationNumber, currentX, currentFX, currentStepLength, currentobjectiveF):
    print("the number of inertion is: "+iterationNumber + "current xi is :"+ currentX+ "current f(x1) is "+currentFX +"current step length is:"+currentStepLength + "current objective function is : " + currentobjectiveF)

def plotObjToIterNumber(iteration,f):
    plt.scatter(iteration, f)
    plt.title('obj by iteration number Rosenbrock -bfgs')
    plt.show()

def plotContour(array,x_list,func):
    x_min = y_min = -2
    x_max = y_max = 2
    delta = 0.0025

    x = np.arange(x_min, x_max, delta)
    y = np.arange(y_min, y_max, delta)
    vec = np.array(list(zip(x, y)))
    X, Y = np.meshgrid(x, y)

    Z = np.empty_like(X)
    w, h = Z.shape
    for i in range(w):
        for j in range(h):
            arr = np.array([X[i, j], Y[i, j]])
            val = func(arr)[0]
            Z[i, j] = val

    fig, ax = plt.subplots()
    plt.contour(X, Y, Z)
    ax.set_title('Contour X^TQ3X -bfgs')

    scatter_x, scatter_y = zip(*x_list)
    plt.scatter(scatter_x, scatter_y)
    plt.show()