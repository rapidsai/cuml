import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import cuml.ts.arima as arima
import cuml.ts.batched_arima as batched_arima
# import cudf
import pandas as pd

def test_arima_cudf():

    num_samples = 1000
    xs = np.linspace(0, 1, num_samples)
    np.random.seed(12)
    noise = np.random.normal(scale=0.1, size=num_samples)
    ys = noise + 0.5*xs

    num_batches = 100000
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
    
    start = timer()
    ll_b = batched_arima.BatchedARIMAModel.loglike(b_model)
    end = timer()
    print("GPU-pd Time ({} batches): {}s".format(num_batches, end-start))
    print("ll_b=", ll_b[0:5])

    models = [arima.ARIMAModel(order, mu, arparams, maparams, ys) for i in range(num_batches)]
    start = timer()
    ll_b2 = batched_arima.batched_loglike(models, gpu=True)
    end = timer()
    print("GPU-np Time ({} batches): {}s".format(num_batches, end-start))

    start = timer()
    b_ll_cpu = batched_arima.batched_loglike(models, gpu=False)
    end = timer()
    print("CPU Time ({} batches): {}s".format(num_batches, end-start))

    # print("ll_b_ref=", b_ll_ref)

    model = arima.ARIMAModel(order, mu, arparams, maparams, ys)
    ll = arima.loglike(model)
    print("ll=", ll)

    for lli in ll_b:
        np.testing.assert_approx_equal(ll, lli)

    return ys_df

if __name__ == "__main__":
    test_arima_cudf()
