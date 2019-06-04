import numpy as np

from cuml.ts.batched_lbfgs import batched_fmin_bfgs

import scipy.optimize as optimize

from IPython.core.debugger import set_trace

def rosenbrock(x, a=1, b=100):

    return (a-x[0])**2 + b*(x[1] - x[0]**2)**2

def g_rosenbrock(x, a=1, b=100):

    g = np.array([-2*a - 4*b*x[0]*(-x[0]**2 + x[1]) + 2*x[0],
                  b*(-2*x[0]**2 + 2*x[1])])
    
    return g


def batched_rosenbrock(x: np.ndarray,
                       num_batches: int,
                       a: np.ndarray,
                       b: np.ndarray) -> np.ndarray:

    fall = np.zeros(num_batches)
    for i in range(num_batches):
        fall[i] = rosenbrock(x[i*2:(i+1)*2], a[i], b[i])

    return fall

def g_batched_rosenbrock(x: np.ndarray,
                         num_batches: int,
                         a: np.ndarray,
                         b: np.ndarray) -> np.ndarray:

    gall = np.zeros(2*num_batches)
    for i in range(num_batches):
        gall[i*2:(i+1)*2] = g_rosenbrock(x[i*2:(i+1)*2], a[i], b[i])

    return gall

def test_batched_bfgs():

    num_batches = 100
    np.random.seed(42)
    a = np.random.normal(1, scale=0.1, size=num_batches)
    b = np.random.normal(100, scale=1, size=num_batches)

    def f(x, n=None):
        nonlocal a
        nonlocal b
        nonlocal num_batches

        if n is not None:
            return rosenbrock(x, a[n], b[n])

        fb = batched_rosenbrock(x, num_batches, a, b)
        return fb.sum()/num_batches

    def gf(x, n=None):
        nonlocal a
        nonlocal b
        nonlocal num_batches

        if n is not None:
            return g_rosenbrock(x, a[n], b[n])
        
        g = g_batched_rosenbrock(x, num_batches, a, b)
        return g

    x0 = np.zeros(2*num_batches)
    x0[0] = 0.9
    x0[1] = 0.9
    
    # global optimizer
    options = {"disp": 10}
    res_ref = optimize.minimize(f, x0, jac=gf, method="BFGS", options=options)

    # problem-at-a-time optimizer
    # for ib in range(num_batches):
    #     res_ref_batch = optimize.minimize(f, x0[2*ib:2*(ib+1)],
    #                                       jac=gf,
    #                                       method="BFGS",
    #                                       options=options, args=(ib,))

    # print("res_ref=", res_ref)

    res_true = np.zeros(num_batches*2)
    for i in range(num_batches):
        res_true[i*2:(i+1)*2] = np.array([a[i], a[i]**2])

    # print("|res_diff_ref|_max", np.max(res_ref.x-res_true))

    # our new batch-aware bfgs optimizer
    res_xk, _ = batched_fmin_bfgs(f, x0, num_batches, g=gf, disp=1, max_steps=100)
    
    print("batched res_xk:", res_xk)
    print("|res_diff_my_batched|_max", np.max(np.abs(res_xk-res_true)))
