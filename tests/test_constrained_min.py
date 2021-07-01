import numpy as np
import matplotlib.pyplot as plt

from src.constrained_min import interior_pt

def main():
    test_lp()
    test_qp()

def test_qp():
    xstar= interior_pt(logQP, np.array([x,y,z]), np.ones(3), np.ones(3), np.array(([0.2,0.1,0.7])))
    print(xstar.shape)
    print(xstar)

def test_lp():
    xstar = interior_pt(logLP, np.array([x, y]), np.ones(2), np.ones(2), np.array([0.5, 0.75]))
    print(xstar.shape)
    print(xstar)


def x(x):
    return -x
def y(y):
    return -y
def z(z):
    return -z
def logQP(x,t):
    f=t*(x[0]**2)+t*(x[1]**2)+t*((x[2]+1)**2) -np.log(x[0]) -np.log(x[1])-np.log(x[2])
    df = np.array([2*t*x[0]-1/x[0],2*x[1]*t-1/x[1],2*t*x[2] +2*t-1/x[2]])
    hess = 2*t*np.eye(3)+np.diag(1/x**2)
    return f,df,hess


def logLP(x,t):
    f= -t*x[0] -t*x[1]-np.log(x[0]+x[1]-1)-np.log(1-x[1])-np.log(2-x[0])-np.log(x[1])
    df= np.array([-t -1/(x[0]-2) -1/(x[0]+x[1]-1), -t -1/(x[1]-1) -1/(x[0]+x[1]-1) -1/x[1]])
    hess =np.array([[1/((x[0]-2)**2)+1/((x[0]+x[1]-1)**2), 1/((x[0]+x[1]-1)**2)],[1/((x[0]+x[1]-1)**2), 1/((x[0]+x[1]-1)**2) +1/x[1]**2 + 1/((x[1]-1)**1)]])
    return f,df,hess



if __name__ == '__main__':
    main()