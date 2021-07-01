import numpy as np
import math
import matplotlib.pyplot as plt

def line_search(f, x0, obj_tol, param_tol, max_iter,dir_selection_method, init_step_len=1.0, slope_ratio=1e-4, back_track_factor=0.2):
        x_list = [x0]
        x_prev = x0
        f_prev, df_prev, hess = f(x0);
        f_list = [f_prev]
        i = 0
        iteration = [i]
        step_size = 0.1
        max_iter = 1000
        obj_tol = 10 ** -2
        param_tol = 10 ** -8
        success = -1
        bk = hess
        bk1 = bk
        while i <= max_iter and success < 1:

            if dir_selection_method == "bfgs":
                pnt= Bfgs_dir(bk, df_prev)

            if (dir_selection_method == "nt"):
                pnt = Newthon_dir(x_prev, f)
            if (dir_selection_method == "gd"):
                pnt=-df_prev
            ak= find_wolfe_step_size(f,x_prev,pnt,df_prev)
            x_next=x_prev+ak*pnt
            x_list.append(x_next)
            print(x_next)

            x_list.append(x_next)
            f_next, df_next,hess_next = f(x_next)
            f_list.append(f_next)
            iteration.append(i+1)
            if dir_selection_method == "bfgs":
                bk = Bfgs_next(bk, df_next, df_prev, x_next, x_prev)
            # utils.report(i, x_prev, f_prev, x_next-x_prev, f_next-f_prev)
            i = i + 1
            success = test_converage(x_next, x_prev, f_next, f_prev, df_next, obj_tol, param_tol)
            x_prev = x_next
            df_prev = df_next

        result = ""
        #success = -1
        if success == -1:
            result = "Max iteration reached"
        elif success == 1:
            result = "numeric tolerance for successful termination in terms of small enough change in objective function values, between two consecutive iterations(𝑓(𝑥𝑖+1)and 𝑓(𝑥𝑖)). "
        else:
            result = "the numeric tolerance for successful termination in terms of small enough distance between two consecutive iterations iteration locations (𝑥𝑖+1 and 𝑥𝑖)."
        print("Result:{res}".format(res=result))
        return x_next, success, x_list,f_list,iteration


def Newthon_dir(x, func):
    f, df, hess = func(x)
    dir = np.linalg.solve(hess, -df)
    return  dir

def Bfgs_dir(bk,df):
    return np.linalg.solve(bk, -df)




def Bfgs_next(bk, df, df_prev, x, x_prev):
    sk = x - x_prev
    sk=sk.reshape(-1, 1)
    yk = df - df_prev
    yk=yk.reshape(-1, 1)
    bk1 = bk - (bk @ sk @ sk.T @ bk) / (sk.T @ bk @ sk) + (yk @ yk.T) / (yk.T @ sk)
    return  bk1


def test_converage(x_next, x_prev, f_next, f_prev, df_next, obj_tol, param_tol):
    resCode = -1
    if (f_next - f_prev <= obj_tol).all():
        resCode = 1
    elif (x_next - x_prev<= param_tol).all():
        resCode = 2
    return resCode

def find_wolfe_step_size(f, xk, pk, grad0, init_step_len=1.0, slope_ratio=1e-4, back_track_factor=0.2):
    alpha = init_step_len
    while np.isnan(f(xk + alpha * pk)[0]) or f(xk + alpha * pk)[0] > f(xk)[0] + slope_ratio * alpha * grad0.T @ pk:
        alpha = back_track_factor * alpha
    return alpha
