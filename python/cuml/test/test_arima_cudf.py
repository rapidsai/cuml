import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import cuml.ts.arima as arima
import cuml.ts.batched_arima as batched_arima
import statsmodels.tsa.arima_model as sm
# import cudf
import pandas as pd

def test_arima():

    num_samples = 1000
    xs = np.linspace(0, 1, num_samples)
    np.random.seed(12)
    noise = np.random.normal(scale=0.1, size=num_samples)
    ys = noise + 0.5*xs

    num_batches = 1000
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


def test_gradient():
    """test gradient implementation"""
    num_samples = 100
    xs = np.linspace(0, 1, num_samples)
    np.random.seed(12)
    noise = np.random.normal(scale=0.1, size=num_samples)
    ys = noise + 0.5*xs
    for num_batches in range(1,4):
        ys_df = np.reshape(np.tile(np.reshape(ys, (num_samples, 1)), num_batches), (num_samples, num_batches), order="F")
        order = (0, 1, 1)
        mu = 0.0
        arparams = np.array([])
        maparams = np.array([-1.0])
        x = np.r_[mu, arparams, maparams]
        x = np.tile(x, num_batches)
        num_samples = ys_df.shape[0]
        num_batches = ys_df.shape[1]

        p, d, q = order
        num_parameters = 1 + p + q
        f = batched_arima.BatchedARIMAModel.ll_f(num_batches, num_parameters, order, ys_df, x)
        
        g = batched_arima.BatchedARIMAModel.ll_gf(num_batches, num_parameters, order, ys_df, x)
        
        grad_fd = np.zeros(len(x))
        h = 1e-8
        for i in range(len(x)):
            def fx(xs):
                return batched_arima.BatchedARIMAModel.ll_f(num_batches, num_parameters, order,
                                                            ys_df, xs).sum()

            xph = np.copy(x)
            xmh = np.copy(x)
            xph[i] += h
            xmh[i] -= h
            f_ph = fx(xph)
            f_mh = fx(xmh)
            grad_fd[i] = (f_ph-f_mh)/(2*h)


        np.testing.assert_array_equal(g, grad_fd)


def test_against_statsmodels():

    num_samples = 100
    xs = np.linspace(0, 1, num_samples)
    np.random.seed(12)
    noise = np.random.normal(scale=0.1, size=num_samples)
    ys = noise + 0.5*xs
    num_batches = 1000
    ys_df = np.reshape(np.tile(np.reshape(ys, (num_samples, 1)), num_batches), (num_samples, num_batches), order="F")
    order = (0, 1, 1)
    mu = 0.0
    arparams = np.array([])
    maparams = np.array([-1.0])
    # b_model = batched_arima.BatchedARIMAModel(num_batches*[order], np.tile(mu, num_batches),
    #                                           num_batches*[arparams],
    #                                           num_batches*[maparams], ys_df)

    
    start = timer()
    sm_model = sm.ARIMA(ys, order=order)
    sm_model_fit = sm_model.fit()
    end = timer()
    print("sm_model_fit time: {}s".format(end-start))

    start = timer()
    b_model = batched_arima.BatchedARIMAModel.fit(ys_df,
                                                  order,
                                                  0.0,
                                                  arparams,
                                                  maparams)
    end = timer()
    print("Rapids ARIMA fit time: {}s".format(end-start))
    
    return sm_model_fit, b_model

if __name__ == "__main__":
    test_against_statsmodels()
    # test_arima()
    # test_gradient()
