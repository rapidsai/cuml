import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import cuml.ts.arima as arima

def test_arima_start_params():

    num_samples = 100
    xs = np.linspace(0, 1, num_samples)
    np.random.seed(12)
    noise = np.random.normal(scale=0.1, size=num_samples)
    ys = noise + 0.5*xs

    ys_diff = np.diff(ys)

    print("test 1,1,1")
    arma = arima.start_params((1, 1, 1), ys_diff)
    print("test 2,1,1")
    arma = arima.start_params((2, 1, 1), ys_diff)

    # plt.plot(xs[0:-3], ys_diff[2:], "k-", xs[0:-4], y_fit, "r--")
    # plt.show()
