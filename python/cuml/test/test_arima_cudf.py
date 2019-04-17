import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import cuml.ts.arima as arima
import cuml.ts.batched_arima as batched_arima
import cudf
import pandas as pd

def test_arima_cudf():

    num_samples = 10
    xs = np.linspace(0, 1, num_samples)
    np.random.seed(12)
    noise = np.random.normal(scale=0.1, size=num_samples)
    ys = noise + 0.5*xs

    num_batches = 2
    ys_df = cudf.DataFrame([(i, ys) for i in range(num_batches)])
    order = (0, 1, 1)
    mu = 0.0
    arparams = np.array([])
    maparams = np.array([-1.0])
    b_model = batched_arima.BatchedARIMAModel(num_batches*[order], np.tile(mu, num_batches),
                                              num_batches*[arparams],
                                              num_batches*[maparams], ys_df)

    ll_b = batched_arima.BatchedARIMAModel.loglike(b_model)
    print("ll_b=", ll_b)

    models = [arima.ARIMAModel(order, mu, arparams, maparams, ys) for i in range(num_batches)]
    b_ll_ref = batched_arima.batched_loglike(models, gpu=True)
    print("ll_b_ref=", b_ll_ref)

    model = arima.ARIMAModel(order, mu, arparams, maparams, ys)
    ll = arima.loglike(model)
    print("ll=", ll)

    for lli in ll_b:
        np.testing.assert_approx_equal(ll, lli)

    return ys_df

if __name__ == "__main__":
    test_arima_cudf()
