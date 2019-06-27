import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import cuml.ts.arima as arima
import cuml.ts.batched_arima as batched_arima
import statsmodels.tsa.arima_model as sm
# import cudf
import pandas as pd

import pmdarima as pm

from scipy.optimize.optimize import _approx_fprime_helper

from cuml.ts.stationarity import stationarity

def test_arima():

    num_samples = 100
    xs = np.linspace(0, 1, num_samples)
    np.random.seed(12)
    noise = np.random.normal(scale=0.1, size=num_samples)
    ys = noise + 0.5*xs

    num_batches = 2
    # ys_df = pd.DataFrame([ys for i in range(num_batches)]).transpose()
    # ys_df = np.reshape
    ys_df = np.reshape(np.tile(np.reshape(ys, (num_samples, 1)), num_batches), (num_samples, num_batches), order="F")
    order = (0, 1, 1)
    mu = 0.0
    arparams = np.array([])
    maparams = np.array([-1.0])
    b_model = batched_arima.BatchedARIMAModel(num_batches*[order], np.tile(mu, num_batches),
                                              num_batches*[arparams],
                                              num_batches*[maparams], ys_df)
    
    # warm-up (cublas handle needs a warmup)
    ll_b = batched_arima.BatchedARIMAModel.loglike(b_model)
    
    # GPU
    start = timer()
    ll_b = batched_arima.BatchedARIMAModel.loglike(b_model)
    end = timer()
    print("GPU Time ({} batches): {}s".format(num_batches, end-start))
    print("ll_b=", ll_b[0:5])

    # CPU
    start = timer()
    ll_b_cpu = batched_arima.BatchedARIMAModel.loglike(b_model, gpu=False)
    end = timer()
    print("CPU Time ({} batches): {}s".format(num_batches, end-start))
    print("ll_b_cpu=", ll_b_cpu[0:5])

    # print("ll_b_ref=", b_ll_ref)

    model = arima.ARIMAModel(order, mu, arparams, maparams, ys)
    ll = arima.loglike(model)
    print("ll=", ll)

    for lli in ll_b:
        np.testing.assert_approx_equal(ll, lli)
        
    return ys_df


def test_ll_batches():
    num_samples = 100
    xs = np.linspace(0, 1, num_samples)
    np.random.seed(12)
    num_batches = 2

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
        mu0.append(0.2 + noise[0]/10)
        ar0.append(np.array([-0.04+noise[1]/100]))
        ma0.append(np.array([-0.8+noise[1]/10]))

    mu0 = np.array(mu0)
    
    # ys_df = pd.DataFrame([ys for i in range(num_batches)]).transpose()
    # ys_df = np.reshape
    # ys_df = np.reshape(np.tile(np.reshape(ys, (num_samples, 1)), num_batches), (num_samples, num_batches), order="F")
    order = (1, 1, 1)
    # mu = 0.2
    # arparams = np.array([-0.04])
    # maparams = np.array([-0.8])
    b_model_all = batched_arima.BatchedARIMAModel(num_batches*[order], mu0, ar0, ma0, ys_df)

    ll_all, vs_all = batched_arima.BatchedARIMAModel.run_kalman(b_model_all)
    b_model_fit_all = batched_arima.BatchedARIMAModel.fit(ys_df, order, mu0, ar0, ma0, opt_disp=90)

    vs_all_2 = np.zeros(vs_all.shape)
    ll_all_2 = np.zeros(num_batches)

    for ib in range(num_batches):        
        ysi = np.reshape(ys_df[:,ib], (num_samples, 1), order="F")
        b_model_i = batched_arima.BatchedARIMAModel(num_batches*[order], np.tile(mu0[ib], 1),
                                                    1*[ar0[ib]],
                                                    1*[ma0[ib]], ysi)
        ll_i, vs_i = batched_arima.BatchedARIMAModel.run_kalman(b_model_i)
        vs_all_2[:, ib] = vs_i[:, 0]
        ll_all_2[ib] = ll_i[0]
        b_model_fit_i = batched_arima.BatchedARIMAModel.fit(ysi, order, np.array([mu0[ib]]), [ar0[ib]], [ma0[ib]], opt_disp=50)

    np.testing.assert_array_equal(ll_all, ll_all_2)
    np.testing.assert_array_equal(vs_all, vs_all_2)


def test_ma_term():

    np.random.seed(12)
    y = np.random.normal(0.0, 0.4, 10)
    yp = 0.1*y[:-1]
    ma_params = np.array([0.1])

    term = batched_arima.BatchedARIMAModel.fc_ma(3, (1, 1, 1), y, yp, ma_params)
    ma_fc1 = (y[-2]-yp[-1])*ma_params[0]
    ma_fc2 = (y[-1]-ma_fc1)*ma_params[0]
    np.testing.assert_array_almost_equal(term, [ma_fc1, ma_fc2, 0.0])

def test_cpu_vs_gpu():
    """test CPU vs GPU implementation"""
    num_samples = 10
    xs = np.linspace(0, 1, num_samples)
    np.random.seed(12)
    noise = np.random.normal(scale=0.1, size=num_samples)
    ys = noise + 0.5*xs
    for num_batches in range(1, 5):

        ys_df = np.reshape(np.tile(np.reshape(ys, (num_samples, 1)), num_batches), (num_samples, num_batches), order="F")
        order = (1, 1, 1)
        mu = 0.0
        arparams = np.array([-0.01])
        maparams = np.array([-1.0])
        x = np.r_[mu, arparams, maparams]
        x = np.tile(x, num_batches)
        num_samples = ys_df.shape[0]
        num_batches = ys_df.shape[1]

        p, d, q = order
        num_parameters = 1 + p + q
        gpu = False
        print("Batched Gradient")
        ll_gpu = batched_arima.BatchedARIMAModel.ll_f(num_batches, num_parameters, order, ys_df, x, gpu=True)
        ll_cpu = batched_arima.BatchedARIMAModel.ll_f(num_batches, num_parameters, order, ys_df, x, gpu=False)

        print("BS({}) GPU vs CPU={} vs {}".format(num_batches, ll_gpu, ll_cpu))
        np.testing.assert_almost_equal(ll_gpu, ll_cpu)

def test_gradient():
    """test gradient implementation"""
    num_samples = 100
    xs = np.linspace(0, 1, num_samples)
    np.random.seed(12)
    noise = np.random.normal(scale=0.1, size=num_samples)
    ys = noise + 0.5*xs
    for num_batches in range(1, 5):
        ys_df = np.reshape(np.tile(np.reshape(ys, (num_samples, 1)), num_batches), (num_samples, num_batches), order="F")
        order = (1, 1, 1)
        mu = 0.0
        arparams = np.array([-0.01])
        maparams = np.array([-1.0])
        x = np.r_[mu, arparams, maparams]
        x = np.tile(x, num_batches)
        num_samples = ys_df.shape[0]
        num_batches = ys_df.shape[1]

        p, d, q = order
        num_parameters = 1 + p + q
        gpu = True
        print("Batched Gradient")
        g = batched_arima.BatchedARIMAModel.ll_gf(num_batches, num_parameters, order, ys_df, x, gpu=gpu)
        print("One-at-a-time Gradient")
        grad_fd = np.zeros(len(x))
        h = 1e-8
        for i in range(len(x)):
            def fx(xp):
                return batched_arima.BatchedARIMAModel.ll_f(num_batches, num_parameters, order,
                                                            ys_df, xp, gpu=gpu).sum()

            xph = np.copy(x)
            xmh = np.copy(x)
            xph[i] += h
            xmh[i] -= h
            f_ph = fx(xph)
            f_mh = fx(xmh)
            grad_fd[i] = (f_ph-f_mh)/(2*h)

        print("g={}, g_ref={}".format(g, grad_fd))
        # np.testing.assert_array_equal(g, grad_fd)

        def f(xk):
            return batched_arima.BatchedARIMAModel.ll_f(num_batches, num_parameters, order,
                                                        ys_df, xk, gpu=True).sum()

        # from scipy
        g_sp = _approx_fprime_helper(x, f, h)
        np.testing.assert_array_equal(g, g_sp, verbose=True)

def test_grid_search():
    num_samples = 200
    xs = np.linspace(0, 1, num_samples)
    np.random.seed(12)
    noise = np.random.normal(scale=0.1, size=num_samples)
    ys = noise + 0.5*xs
    num_batches = 2
    ys_b = np.reshape(np.tile(np.reshape(ys, (num_samples, 1)), num_batches), (num_samples, num_batches), order="F")
    best_model, ic = batched_arima.grid_search(ys_b, d=1)
    print("ic={}".format(ic))

    pm_model_fit = pm.auto_arima(ys, seasonal=False)

    return ic, best_model, pm_model_fit


def test_stationarity():

    num_samples = 200
    xs = np.linspace(0, 1, num_samples)
    np.random.seed(12)
    noise = np.random.normal(scale=0.1, size=num_samples)
    ys1 = noise + 0.5*xs
    ys2 = noise

    num_batches = 2
    ys_df = np.zeros((num_samples, num_batches), order="F")
    ys_df[:, 0] = ys1
    ys_df[:, 1] = ys2

    d_b = stationarity(ys_df)
    np.testing.assert_array_equal(d_b, [1, 0])


def test_against_statsmodels(plot=True):

    num_samples = 100
    xs = np.linspace(0, 1, num_samples)
    np.random.seed(12)
    noise = np.random.normal(scale=0.1, size=num_samples)
    ys = noise + 0.5*xs
    num_batches = 1
    ys_df = np.reshape(np.tile(np.reshape(ys, (num_samples, 1)), num_batches), (num_samples, num_batches), order="F")
    # order = (0, 1, 1)
    order = (1, 1, 1)
    mu = 0.0
    arparams = np.array([-0.01])
    # arparams = np.array([])
    maparams = np.array([-1.0])

    # x0 = np.array([0.01051398, -0.4953897, -2.09984231])

    # b_model = batched_arima.BatchedARIMAModel(num_batches*[order], [x0[0]],
    #                                           num_batches*[np.array([x0[1]])],
    #                                           # num_batches*[np.array([])],
    #                                           num_batches*[np.array([x0[2]])], ys_df)

    # ll_b = batched_arima.BatchedARIMAModel.loglike(b_model)

    start = timer()
    for i in range(num_batches):
        yi = ys_df[:,i]
        sm_model = sm.ARIMA(yi, order=order)
        sm_model_fit = sm_model.fit(disp=-1)
    end = timer()
    print("sm_model_fit time (1 batches): {}s ({}s / batch / step)".format((end-start),
                                                                           (end-start)/1/num_samples))

    start = timer()
    y_sm_p = sm_model_fit.predict(start=1, end=len(xs), typ="levels")
    fc_steps = 10

    y_sm_fc, _, _ = sm_model_fit.forecast(steps=fc_steps)
    # note: statsmodel forecast includes last in-sample prediction as first step.
    x_fc = np.linspace(xs[-1]+(xs[1]-xs[0]), xs[-1]+(xs[1]-xs[0])*(fc_steps+1), fc_steps)
    end = timer()

    gpu = False

    start = timer()
    b_model = batched_arima.BatchedARIMAModel.fit(ys_df,
                                                  order,
                                                  sm_model_fit.params[0],
                                                  [sm_model_fit.params[1]],
                                                  # [],
                                                  [sm_model_fit.params[1]],
                                                  gpu=gpu, opt_disp=-1)
    end = timer()
    print("Rapids ARIMA fit time: {}s ({}s / batch / step)".format(end-start, (end-start)/num_batches/num_samples))
    print("b_model: ", b_model)
    y_b = batched_arima.BatchedARIMAModel.predict_in_sample(b_model, gpu=gpu)
    y_fc = batched_arima.BatchedARIMAModel.forecast(b_model, nsteps=fc_steps)

    if plot:
        plt.clf()
        plt.plot(xs, ys, "k-")
        plt.plot(xs, y_sm_p,"r-")
        plt.plot(x_fc, y_sm_fc,"g-")
        # plt.plot(x_fc, y_fc[:, 0],"b--")
        plt.plot(x_fc, y_fc,"b--")
        plt.plot(xs, y_b[:, 0],"b--")
        plt.show()

    # print("||y_sm_p - y_b||=", [np.linalg.norm((ys+y_sm_p) - y_b[:, i]) for i in range(num_batches)])

    for i in range(num_batches):
        np.testing.assert_array_almost_equal(y_sm_p, y_b[:, i], 4)
        np.testing.assert_array_almost_equal(y_sm_fc, y_fc[:, i], 4)

if __name__ == "__main__":
    # test_against_statsmodels()
    # test_arima()
    # test_cpu_vs_gpu()
    test_gradient()
    
