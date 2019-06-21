import numpy as np

from cuml.ts.batched_bfgs import batched_fmin_bfgs
from cuml.ts.batched_lbfgs import batched_fmin_lbfgs

import scipy.optimize as optimize

from IPython.core.debugger import set_trace

import cuml.ts.batched_arima as batched_arima

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

def test_batched_bfgs_rosenbrock():

    num_batches = 1
    np.random.seed(42)
    a = np.random.normal(1, scale=0.1, size=num_batches)
    b = np.random.normal(100, scale=1, size=num_batches)
    a = np.array([1])
    b = np.array([100])

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
    x0[0] = 0.0
    x0[1] = 0.0
    
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
    res_xk, _, _ = batched_fmin_bfgs(f, x0, num_batches, g=gf, disp=1, max_steps=100)
    
    # print("batched res_xk:", res_xk)
    # print("|res_diff_my_batched|_max", np.max(np.abs(res_xk-res_true)))
    np.testing.assert_almost_equal(np.max(np.abs(res_xk-res_true)), 0.0)
    

def test_batched_lbfgs_rosenbrock():

    num_batches = 5
    np.random.seed(42)
    a = np.random.normal(1, scale=0.1, size=num_batches)
    b = np.random.normal(100, scale=10, size=num_batches)

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
    x0[0] = 0.0
    x0[1] = 0.0
    
    # global optimizer
    options = {"disp": 10}
    # res_ref = optimize.minimize(f, x0, jac=gf, method="L-BFGS-B", options=options)

    # problem-at-a-time optimizer
    # for ib in range(num_batches):
    #     res_ref_batch = optimize.minimize(f, x0[2*ib:2*(ib+1)],
    #                                       jac=gf,
    #                                       method="L-BFGS-B",
    #                                       options=options, args=(ib,))

    # print("res_ref=", res_ref)

    res_true = np.zeros(num_batches*2)
    for i in range(num_batches):
        res_true[i*2:(i+1)*2] = np.array([a[i], a[i]**2])

    # print("|res_diff_ref|_max", np.max(res_ref.x-res_true))

    # our new batch-aware l-bfgs optimizer
    res_xk, niter = batched_fmin_lbfgs(f, x0, num_batches, gf, iprint=-1, factr=100)
    
    # print("batched res_xk:", res_xk)
    print("|res_diff_my_batched|_max", np.max(np.abs(res_xk-res_true)))
    np.testing.assert_almost_equal(np.max(np.abs(res_xk-res_true)), 0.0)
    


def test_batch_arima_fit():
    num_samples = 100
    xs = np.linspace(0, 1, num_samples)
    np.random.seed(12)
    num_batches = 30

    noise = np.random.normal(scale=0.1, size=num_samples)
    ys = noise + 0.5*xs

    ys_df = np.zeros((num_samples, num_batches), order="F")
    mu0 = []
    ar0 = []
    ma0 = []
    for ib in range(num_batches):
        noise = np.random.normal(scale=0.1, size=num_samples)
        ys = noise + 0.5*xs
        ys_df[:,ib] = ys[:]
        mu0.append(0.02 + noise[0]/10)
        ar0.append(np.array([-0.04+noise[1]/100]))
        ma0.append(np.array([-0.8+noise[2]/10]))

    mu0 = np.array(mu0)

    order = (1, 1, 1)

    b_model_fit_all = batched_arima.BatchedARIMAModel.fit(ys_df, order, mu0, ar0, ma0, opt_disp=1, h=1e-8)
    print("b_model:", b_model_fit_all)


def test_single_arima_fit():
    num_samples = 100
    xs = np.linspace(0, 1, num_samples)
    np.random.seed(12)
    num_batches = 30

    order = (1, 1, 1)

    for ib in range(num_batches):
        noise = np.random.normal(scale=0.1, size=num_samples)
        ys = np.reshape(noise + 0.5*xs, (num_samples, 1), order="F")
        b_model_fit_single = batched_arima.BatchedARIMAModel.fit(ys, order,
                                                                 np.array([0.02 + noise[0]/10]),
                                                                 [np.array([-0.04+noise[1]/100])],
                                                                 [np.array([-0.8+noise[2]/10])],
                                                                 opt_disp=1, h=1e-8)

    


def test_batched_lbfgs():

    x0 = np.array([0, 0])

    xs = batched_fmin_lbfgs(rosenbrock, x0, g_rosenbrock, iprint=1, maxiter=100, m=10)
    xs2 = optimize.fmin_l_bfgs_b(rosenbrock, x0, g_rosenbrock, iprint=1)


if __name__ == "__main__":
    test_lbfgs()
