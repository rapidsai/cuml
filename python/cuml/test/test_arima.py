import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import cuml.ts.arima as arima

# test data time
t = np.array([1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20,
              21, 24, 25, 26, 28, 39, 40, 41, 42, 43, 45, 46, 48, 50, 51, 52, 53,
              55, 56, 58, 59, 60, 63, 71, 72, 74])

# test dataset 0
data0 = np.array([16454, 12708, 14084, 20929, 11888, 13378, 20503, 17422, 16574,
                  16567, 14222, 14471, 11988, 17122, 15448, 14290, 13679, 10690,
                  17240, 17900, 16673, 1070, 16165, 15832, 18495, 15160, 15638,
                  21688, 18284, 2306, 10159, 8224, 7517, 14363, 11185, 15804,
                  2816, 12217, 7739, 5459, 6241, 171, 11118])

# test dataset 1
data1 = np.array([16492, 12671, 13644, 18120, 11420, 10904, 20723, 17011, 15109,
                  15791, 13014, 14622, 12029, 15932, 14731, 13573, 13229, 11371,
                  16400, 16939, 16672, 2520, 14627, 14035, 14724, 15094, 12812,
                  20126, 16411, 2687, 9582, 8291, 7352, 14313, 10552, 14143,
                  2790, 12960, 7362, 4606, 6155, 158, 11435])

# The ARIMA model of dataset 0. ("smoothed dataset 0")
data_smooth = np.array([16236.380267964598, 14198.707110817017, 13994.129600585984,
                        15705.975404284243, 14455.226246272636, 14114.076675764649,
                        15033.216755054425, 15021.10438408751, 14954.822759706418,
                        14904.042532492134, 14557.421649530697, 14347.41471896904,
                        13877.476483976807, 14059.990544916833, 13888.386639087348,
                        13665.988312305493, 13436.674608089721, 12979.25813798955,
                        13199.416272194492, 13243.849692596767, 13157.053784142185,
                        11904.470827085499, 12356.442250181439, 12279.590418507576,
                        12401.153685335092, 12190.66504090282, 12122.442825730872,
                        12444.119210649873, 12326.524612239178, 11276.55939500802,
                        11278.522346300862, 10911.26233776968, 10575.493222628831,
                        10692.727355175008, 10395.405550019213, 10480.90443078538,
                        9652.114779061498, 9806.45087894164, 9401.00281392505,
                        9019.688213508754, 8766.056499652503, 8158.794074075997,
                        8294.86605488629])

def get_data():
    d = np.zeros((len(t), 2))
    d[:, 0] = data0
    d[:, 1] = data1
    return (t, d)

def test_arima_start_params():
    """
    Tests start_params function for multiple (p,d,q) options
    """
    _, ys = get_data()
    arma = arima.start_params((1, 1, 1), ys[:, 0])
    print("arma=", arma)
    arma_ref = np.array([1.306700000000000e+04, 8.578545193799022e-01, -6.241669663164802e-01])
    np.testing.assert_array_almost_equal(arma, arma_ref)
    arma = arima.start_params((2, 1, 1), ys[:, 0])
    print("arma=", arma)
    arma_ref = [1.3067000000000000e+04, 1.4359734767607857e-01, 1.9335180865645191e-01,
                9.0764356294912391e-02]
    np.testing.assert_array_almost_equal(arma, arma_ref)


def test_log_likelihood():
    """
    Test loglikelihood against reference results using reference parameters
    """
    x0 = [[-220.35376518754148, -0.2617000627224417, -2.1893003751753457],
          [-2.3921544864718811e+02, -1.3525124433776395e-01, -7.5978156540072991e-02,
           -2.4055488944465053e+00]]
    ref_ll = [-415.7117855771454, -415.32341960785186]

    _, y = get_data()

    for p in range(1, 3):
        order = (p, 1, 1)
        y0 = np.zeros((len(t), 1))
        y0[:, 0] = y[:, 0]
        ll = arima.ll_f(1, order, y0, np.copy(x0[p-1]), trans=True)
        # print("ll=", ll)
        np.testing.assert_almost_equal(ll, ref_ll[p-1])


def testBIC():
    """Test "Bayesian Information Criterion" metric. BIC penalizes the
    log-likelihood with the number of parameters.

    """
    np.set_printoptions(precision=16)

    bic_reference = [[851.0904458614862, 842.6620993460326], [854.747970752074, 846.2220267762417]]

    _, y = get_data()

    for p in range(1, 3):
        order = (p, 1, 1)

        nb = 2

        x0 = np.array([])
        for i in range(nb):
            x0i = arima.init_x0(order, y[:,i])
            x0 = np.r_[x0, x0i]

        p, d, q = order
        mu0, ar0, ma0 = arima.unpack(p, d, q, nb, x0)

        batched_model = arima.fit(y, order,
                                  mu0,
                                  ar0,
                                  ma0,
                                  opt_disp=-1, h=1e-9, gpu=True)

        # print("BIC({}, 1, 1): ".format(p), batched_model.bic)
        np.testing.assert_allclose(batched_model.bic, bic_reference[p-1], rtol=1e-4)


def testFit_Predict_Forecast(plot=False):
    """
    Full integration test: Tests fit followed by in-sample prediction and out-of-sample forecast
    """
    np.set_printoptions(precision=16)

    t, y = get_data()

    ns_train = 35
    ns_test = len(t) - ns_train

    y_b_p = []
    y_f_p = []

    for p in range(1, 3):
        order = (p, 1, 1)

        nb = 2

        x0 = np.array([])
        y_train = np.zeros((ns_train, nb))
        for i in range(nb):
            y_train[:, i] = y[:ns_train, i]
            x0i = arima.init_x0(order, y_train[:, i])
            x0 = np.r_[x0, x0i]

        p, d, q = order
        mu0, ar0, ma0 = arima.unpack(p, d, q, nb, x0)

        batched_model = arima.fit(y_train, order,
                                  mu0,
                                  ar0,
                                  ma0,
                                  opt_disp=-1, h=1e-9, gpu=True)

        y_b = arima.predict_in_sample(batched_model)
        y_fc = arima.forecast(batched_model, ns_test)

        y_b_p.append(y_b)
        y_f_p.append(y_fc)

    if plot:
        nb_plot = 2
        fig, axes = plt.subplots(nb_plot, 1)
        axes[0].plot(t, y[:, 0], t[:ns_train], y_b_p[0][:, 0], "r-", t[ns_train-1:-1], y_f_p[0][:, 0], "--")
        axes[0].plot(t[:ns_train], y_b_p[1][:, 0], "g-", t[ns_train-1:-1], y_f_p[1][:, 0], "y--")
        axes[1].plot(t, y[:, 1], t[:ns_train], y_b_p[0][:, 1], "r-", t[ns_train-1:-1], y_f_p[0][:, 1], "--")
        axes[1].plot(t[:ns_train], y_b_p[1][:, 1], "g-", t[ns_train-1:-1], y_f_p[1][:, 1], "y--")

        plt.show()

    l2_error_predict0 = np.sum((y_b_p[0][:, :] - y[:ns_train, :])**2, axis=0)
    l2_error_predict1 = np.sum((y_b_p[1][:, :] - y[:ns_train, :])**2, axis=0)
    # print("l2_error_predict=({},{})".format(l2_error_predict0, l2_error_predict1))

    l2_error_ref0 = [5.182155880809287e+08, 4.431172130950448e+08]
    l2_error_ref1 = [5.401615747413801e+08, 4.649077944193724e+08]

    l2_error_forecast0 = np.sum((y_f_p[0][:, :] - y[ns_train-1:-1, :])**2, axis=0)
    l2_error_forecast1 = np.sum((y_f_p[1][:, :] - y[ns_train-1:-1, :])**2, axis=0)
    # print("l2_error_forecast=({},{})".format(l2_error_forecast0, l2_error_forecast1))

    l2_error_fc_ref0 = [2.783439226866143e+08, 2.400999999394908e+08]
    l2_error_fc_ref1 = [3.7288613986029667e+08, 3.0391497933745754e+08]

    np.testing.assert_allclose(l2_error_predict0, l2_error_ref0, rtol=1e-3)
    np.testing.assert_allclose(l2_error_predict1, l2_error_ref1, rtol=1e-3)
    np.testing.assert_allclose(l2_error_forecast0, l2_error_fc_ref0, rtol=1e-3)
    np.testing.assert_allclose(l2_error_forecast1, l2_error_fc_ref1, rtol=1e-3)
    


def bench_arima(num_batches=240, plot=False):

    ns = len(t)
    y_b = np.zeros((ns, num_batches))

    for i in range(num_batches):
        y_b[:, i] = np.random.normal(size=ns, scale=2000) + data_smooth

    # if plot:
    #     plt.plot(t, data_smooth, "r--", t, data0, "k-", t, y_b[:, 0], t, y_b[:, 1], t, y_b[:, 2])
    #     plt.show()
    # return

    p, d, q = (1, 1, 1)
    order = (p, d, q)

    x0 = np.array([])

    start = timer()

    for i in range(num_batches):
        x0i = arima.init_x0(order, y_b[:, i])
        x0 = np.r_[x0, x0i]

    mu0, ar0, ma0 = arima.unpack(p, d, q, num_batches, x0)

    batched_model = arima.fit(y_b, order,
                              mu0,
                              ar0,
                              ma0,
                              opt_disp=-1, h=1e-9, gpu=True)

    end = timer()

    print("GPU Time ({} batches) = {} s".format(num_batches, end - start))
    print("Solver iterations (max/min/avg): ", np.max(batched_model.niter), np.min(batched_model.niter),
          np.mean(batched_model.niter))

    yt_b = arima.predict_in_sample(batched_model)

    if plot:
        plt.plot(t, y_b[:, 0], "k-", t, yt_b[:, 0], "r--", t, data0, "g--", t, data_smooth, "y--")
        plt.show()


if __name__ == "__main__":
    # testBIC()
    # testFit_Predict_Forecast()

    bench_arima(num_batches=200)
