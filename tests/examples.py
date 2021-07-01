import numpy as np;

def Qfun2(x):
    Q = np.array([[5, 0], [0, 1]])
    f = np.dot(np.dot(x.T, Q), x)
    df = np.dot((Q + Q.T), x)
    hess = Q + Q.T
    return f, df, hess

def Qfun1(x):
    Q = np.array([[1, 0], [0, 1]])
    f = np.dot(np.dot(x.T, Q), x)
    df = np.dot((Q + Q.T), x)
    hess = Q + Q.T
    return f, df, hess

def Qfun3(x):
    Q1 = np.array([[5, 0], [0, 1]])
    Q2 = np.array([[np.sqrt(3) / 2, -0.5], [0.5, np.sqrt(3) / 2]])
    Q = Q2.T @ Q1 @ Q2
    f = np.dot(np.dot(x.T, Q), x)
    df = np.dot((Q + Q.T), x)
    hess = Q + Q.T
    return f, df, hess

def linFunc(x):
    a=np.array([5.2,3.8])
    f= np.dot(x,a)
    df= a
    return f, df,0

def funcRosenbrock(x):
    df = np.array([400 * x[0] ** 3 - 400 * x[0]*x[1] + 2 * x[0] - 2, 200 * x[1] - 200*x[0] * x[0]])
    f = 100 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]) + (1 - x[0]) * (1 - x[0])
    hess = np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2 ,-400 * x[0]], [-400 * x[0], 200]])
    return f, df, hess

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
