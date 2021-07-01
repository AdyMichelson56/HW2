import numpy as np



def interior_pt(func, ineq_constraints, eq_constraints_mat,eq_constraints_rhs, x0):
        n = len(ineq_constraints)
        x = x0
        t = 1
        u = 10
        epsilon = 10 ** -13
        xStar = BarrierMethod(x, epsilon, n, t, u, func, eq_constraints_mat,eq_constraints_rhs)
        return xStar[0]

def BarrierMethod(x, epsilon, m, t, u, func, eq_constraints_mat,eq_constraints_rhs):
        isTerminate = False
        i = 0
        values = []
        iteration = []
        while (not isTerminate):
            if(eq_constraints_mat.shape[0] > 0):
                x = NewtonMethod(x, epsilon, t,func, eq_constraints_mat,eq_constraints_rhs)
            else:
                x=line_search1(func,t, x,10**-2,10**-8,1000)
            iteration.append(i)
            #values.append(f0(x, mu, G, k))
            if (m / t < epsilon):
                isTerminate = True
            else:
                t = t * u
                i = i + 1

        return x, values, iteration


def NewtonMethod(x, epsilon,t,func, A, b):
        isTerminate = False
        i = 0
        max_itr = 10
        n = len(x)
        while (not isTerminate):
            f, grad_f,hess_f = func(x,t)
            rhs = np.append(-grad_f, 0)
            kkt_mat = np.concatenate((hess_f, np.ones(n).reshape((1, n))))
            c = np.append(np.ones(n), 0)
            c = c.reshape(((n + 1, 1)))
            kkt_mat = np.concatenate((kkt_mat, c), axis=1)
            pnt = np.linalg.solve(kkt_mat, rhs)
            pnt = pnt[:n]
            newtonDecrment = np.power(NewtonDecrment(pnt, hess_f), 2)
            if (0.5 * newtonDecrment < epsilon):
                isTerminate = True
            else:
                #ak = find_wolfe_step_size(t,func,x,pnt,grad_f)
                x = x +1* pnt
                i= i+1
        return x

def NewtonDecrment(pnt, dfdfx):
        return np.sqrt(pnt.T @ dfdfx @ pnt)

def find_wolfe_step_size(t,f, xk, pk, grad0, init_step_len=1.0, slope_ratio=1e-4, back_track_factor=0.1):
    alpha = init_step_len
    i=0
    while i<10 or f(xk + alpha * pk,t)[0] > f(xk,t)[0] + slope_ratio * alpha * grad0.T @ pk :
        alpha = back_track_factor * alpha
        print(alpha)
        i=i+1
    return alpha


def line_search1(f,t, x0, obj_tol, param_tol, max_iter):
        x_list = [x0]
        x_prev = x0
        f_prev, df_prev, hess = f(x0,t);
        f_list = [f_prev]
        i = 0
        iteration = [i]
        step_size = 0.1
        max_iter = 1000
        obj_tol = 10 ** -2
        param_tol = 10 ** -8
        success = -1
        while i <= max_iter and success < 1:
            pnt = Newthon_dir(x_prev, f,t)
            pnt = -df_prev
            ak = 1
            x_next = x_prev + ak * pnt
            x_list.append(x_next)
            print(x_next)

            x_list.append(x_next)
            f_next, df_next, hess_next = f(x_next,t)
            f_list.append(f_next)
            iteration.append(i + 1)
            # utils.report(i, x_prev, f_prev, x_next-x_prev, f_next-f_prev)
            i = i + 1
            success = test_converage(x_next, x_prev, f_next, f_prev, df_next, obj_tol, param_tol)
            x_prev = x_next
            df_prev = df_next

        print("test")
        result = ""
        if success == -1:
            result = "Max iteration reached"
        elif success == 1:
            result = "numeric tolerance for successful termination in terms of small enough change in objective function values, between two consecutive iterations(𝑓(𝑥𝑖+1)and 𝑓(𝑥𝑖)). "
        else:
            result = "the numeric tolerance for successful termination in terms of small enough distance between two consecutive iterations iteration locations (𝑥𝑖+1 and 𝑥𝑖)."
        print("Result:{res}".format(res=result))
        return x_next, success, x_list, f_list, iteration


def test_converage(x_next, x_prev, f_next, f_prev, df_next, obj_tol, param_tol):
    resCode = -1
    if (f_next - f_prev <= obj_tol).all():
        resCode = 1
    elif (x_next - x_prev<= param_tol).all():
        resCode = 2
    return resCode

def Newthon_dir(x, func,t):
    f, df, hess = func(x,t)
    dir = np.linalg.solve(hess, -df)
    return  dir