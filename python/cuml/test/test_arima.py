#
# Copyright (c) 2019, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import numpy as np
from timeit import default_timer as timer
import cuml.tsa.arima as arima
from scipy.optimize.optimize import _approx_fprime_helper

from cuml.utils.input_utils import input_to_host_array


# test data time
t = np.array([1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21,
              24, 25, 26, 28, 39, 40, 41, 42, 43, 45, 46, 48, 50, 51, 52, 53,
              55, 56, 58, 59, 60, 63, 71, 72, 74])

# test dataset 0
data0 = np.array([16454, 12708, 14084, 20929, 11888, 13378, 20503, 17422,
                  16574, 16567, 14222, 14471, 11988, 17122, 15448, 14290,
                  13679, 10690, 17240, 17900, 16673, 1070, 16165, 15832, 18495,
                  15160, 15638, 21688, 18284, 2306, 10159, 8224, 7517, 14363,
                  11185, 15804, 2816, 12217, 7739, 5459, 6241, 171, 11118])

# test dataset 1
data1 = np.array([16492, 12671, 13644, 18120, 11420, 10904, 20723, 17011,
                  15109, 15791, 13014, 14622, 12029, 15932, 14731, 13573,
                  13229, 11371, 16400, 16939, 16672, 2520, 14627, 14035, 14724,
                  15094, 12812, 20126, 16411, 2687, 9582, 8291, 7352, 14313,
                  10552, 14143, 2790, 12960, 7362, 4606, 6155, 158, 11435])

# The ARIMA model of dataset 0. ("smoothed dataset 0")
data_smooth = np.array([16236.380267964598, 14198.707110817017,
                        13994.129600585984, 15705.975404284243,
                        14455.226246272636, 14114.076675764649,
                        15033.216755054425, 15021.10438408751,
                        14954.822759706418, 14904.042532492134,
                        14557.421649530697, 14347.41471896904,
                        13877.476483976807, 14059.990544916833,
                        13888.386639087348, 13665.988312305493,
                        13436.674608089721, 12979.25813798955,
                        13199.416272194492, 13243.849692596767,
                        13157.053784142185, 11904.470827085499,
                        12356.442250181439, 12279.590418507576,
                        12401.153685335092, 12190.66504090282,
                        12122.442825730872, 12444.119210649873,
                        12326.524612239178, 11276.55939500802,
                        11278.522346300862, 10911.26233776968,
                        10575.493222628831, 10692.727355175008,
                        10395.405550019213, 10480.90443078538,
                        9652.114779061498, 9806.45087894164, 9401.00281392505,
                        9019.688213508754, 8766.056499652503,
                        8158.794074075997, 8294.86605488629])


def create_yp_ref():
    """ creates reference prediction """
    (_, y) = get_data()
    model = arima.ARIMA(y, (1, 1, 1), fit_intercept=True)
    model.set_params({
        "mu": np.array([-217.7230173548441, -206.81064091237104]),
        "ar": np.array([[0.0309380078339684, -0.0371740508810001]], order='F'),
        "ma": np.array([[-0.9995474311219695, -0.9995645146854383]], order='F')
    })
    y_p_ref = model.predict_in_sample()
    print("yp_ref=", y_p_ref)


# reference prediction for (1,1,1). recreate with `create_yp_ref()`
yp_ref = [[16236.276982645155, 14199.83635121614, 13993.612504802639,
           15701.001917841138, 14457.318782427961, 14114.436684625534,
           15028.287621746756, 15018.826402730409, 14953.184455915669,
           14902.339385888643, 14557.310116753155, 14346.872075971714,
           13878.49920540047, 14056.977409351373, 13886.543137497267,
           13664.846049477095, 13435.78531068983, 12980.316970030086,
           13195.421878944875, 13239.340147071023, 13153.408671153384,
           11912.022478836143, 12352.451020219527, 12275.76344198953,
           12395.309983436986, 12187.223001418526, 12118.535320809358,
           12435.729542924131, 12320.661934977046, 11282.214722260982,
           11278.023270572445, 10911.916572651637, 10576.37138790725,
           10688.356982664653, 10393.213992661886, 10475.175832966357,
           9655.699137880823, 9802.85623495, 9400.49054615417,
           9020.574181472959, 8766.084012642543, 8162.945169968312,
           8291.973806637427],
          [16285.189359087628, 14338.460909054174,
           13850.63823251114, 14556.56635360983, 13954.695497411303,
           13244.780548562172, 13744.642846463914, 14080.121846941318,
           14051.566389907626, 13986.915237521414, 13786.62136453952,
           13582.380651361393, 13344.11985120289, 13170.332862411682,
           13105.025676475907, 12962.955049014487, 12776.960524427446,
           12553.108594193804, 12354.900642927994, 12350.399980965518,
           12352.327486277976, 12215.518342586416, 11656.131573206087,
           11563.059813979233, 11449.754138979828, 11362.05755263616,
           11286.236966021392, 11116.378254211602, 11223.015986560224,
           11233.463302287848, 10709.250034043267, 10466.998468513524,
           10192.800693817426, 9840.980314287335, 9786.651333552647,
           9559.92129655608, 9584.118472336395, 9080.57441537021,
           9030.024898020312, 8807.168013053131, 8470.279842824808,
           8280.44295003853, 7648.106311322318]]


def get_data():
    """Convenience function to get reference data"""
    d = np.zeros((len(t), 2))
    d[:, 0] = data0
    d[:, 1] = data1
    return (t, d)


def test_transform():
    """Test the parameter transformation code."""
    x0 = np.array([-36.24493319, -0.76159416, -0.76159516, -167.65533746,
                   -0.76159416, -0.76159616])

    # Without corrections to the MA parameters, this inverse transform will
    # return NaN
    Tx0 = arima._batch_invtrans(0, 1, 2, 2, x0)

    assert(not np.isnan(Tx0).any())

    Tx0 = arima._batch_invtrans(2, 1, 0, 2, x0)

    assert(not np.isnan(Tx0).any())

    Tx0 = arima._batch_invtrans(1, 1, 1, 2, np.array([-1.27047619e+02,
                                                      1.90024682e-02,
                                                      -5.88867176e-01,
                                                      -1.20404762e+02,
                                                      5.12333137e-05,
                                                      -6.14485076e-01]))
    np.testing.assert_allclose(Tx0, np.array([-1.27047619e+02,
                                              3.80095119e-02,
                                              -1.35186024e+00,
                                              -1.20404762e+02,
                                              1.02466627e-04,
                                              -1.43219144e+00]))


def test_log_likelihood():
    """
    Test loglikelihood against reference results using reference parameters
    """
    x0 = [[-220.35376518754148,
           -0.2617000627224417,
           -2.1893003751753457],
          [-2.3921544864718811e+02, -1.3525124433776395e-01,
           -7.5978156540072991e-02,
           -2.4055488944465053e+00]]
    ref_ll = [-415.7117855771454, -415.32341960785186]

    _, y = get_data()

    for p in range(1, 3):
        order = (p, 1, 1)
        y0 = np.zeros((len(t), 1), order='F')
        y0[:, 0] = y[:, 0]
        ll = arima.ll_f(1, len(t), order, 1, y0, np.copy(x0[p-1]), trans=True)
        np.testing.assert_almost_equal(ll, ref_ll[p-1])

    x = [-1.2704761899e+02, 3.8009501900e-02, -1.3518602400e+00,
         -1.2040476199e+02, 1.0245662700e-04, -1.4321914400e+00]
    ll = arima.ll_f(2, len(t), (1, 1, 1), 1, y, np.array(x))
    np.set_printoptions(precision=14)
    ll_ref = np.array([-418.2732740315433, -413.7692130741877])
    np.testing.assert_allclose(ll, ll_ref)


def test_gradient_ref():
    """Tests the gradient based on a reference output"""
    x = np.array([-1.2704761899e+02, 3.8009511900e-02, -1.3518602400e+00,
                  -1.2040476199e+02, 1.0246662700e-04, -1.4321914400e+00])

    _, y = get_data()
    np.set_printoptions(precision=14)
    g = arima.ll_gf(2, len(t), 3, (1, 1, 1), 1, y, x)
    g_ref = np.array([-7.16227077646181e-04, -4.09565927839139e+00,
                      -4.10715017551411e+00, -1.02602371043758e-03,
                      -4.46265460141149e+00,
                      -4.18378931499319e+00])
    np.testing.assert_allclose(g, g_ref, rtol=1e-6)


def test_gradient():
    """test gradient implementation using FD"""
    num_samples = 100
    xs = np.linspace(0, 1, num_samples)
    np.random.seed(12)
    noise = np.random.normal(scale=0.1, size=num_samples)
    ys = noise + 0.5*xs
    for num_batches in range(1, 5):
        ys_df = np.reshape(np.tile(np.reshape(ys,
                                              (num_samples, 1)),
                                   num_batches),
                           (num_batches, num_samples), order="C").T
        order = (1, 1, 1)
        mu = 0.0
        arparams = np.array([-0.01])
        maparams = np.array([-1.0])
        x = np.r_[mu, arparams, maparams]
        x = np.tile(x, num_batches)
        num_samples = ys_df.shape[0]
        num_batches = ys_df.shape[1]

        p, d, q = order
        num_parameters = d + p + q
        g = arima.ll_gf(num_batches, num_samples,
                        num_parameters, order, 1, ys_df, x)
        grad_fd = np.zeros(len(x))
        h = 1e-8
        for i in range(len(x)):
            def fx(xp):
                return arima.ll_f(num_batches, num_samples, order, 1,
                                  ys_df, xp).sum()

            xph = np.copy(x)
            xmh = np.copy(x)
            xph[i] += h
            xmh[i] -= h
            f_ph = fx(xph)
            f_mh = fx(xmh)
            grad_fd[i] = (f_ph-f_mh)/(2*h)

        np.testing.assert_allclose(g, grad_fd, rtol=1e-4)

        def f(xk):
            return arima.ll_f(num_batches, num_samples, order, 1,
                              ys_df, xk).sum()

        # from scipy
        g_sp = _approx_fprime_helper(x, f, h)
        np.testing.assert_allclose(g, g_sp, rtol=1e-4)


def test_bic():
    """Test "Bayesian Information Criterion" metric. BIC penalizes the
    log-likelihood with the number of parameters.

    """
    np.set_printoptions(precision=16)

    bic_reference = [[851.0904458614862, 842.6620993460326],
                     [854.747970752074, 846.2220267762417]]

    _, y = get_data()

    for p in range(1, 3):
        order = (p, 1, 1)

        batched_model = arima.ARIMA(y, order, fit_intercept=True)
        batched_model.fit()

        np.testing.assert_allclose(batched_model.bic,
                                   bic_reference[p-1], rtol=1e-4)


def test_fit():
    """Test the `fit()` function against reference parameters."""
    _, y = get_data()

    mu_ref = [np.array([-217.7230173548441, -206.81064091237104]),
              np.array([-217.72325384510506, -206.77224439903458])]
    ar_ref = [
        np.array([[0.0309380078339684, -0.0371740508810001]], order='F'),
        np.array([[0.0309027562133337, -0.0386322768036704],
                  [-0.0191533926207157, -0.0330133336831984]], order='F')]
    ma_ref = [
        np.array([[-0.9995474311219695, -0.9995645146854383]], order='F'),
        np.array([[-0.999629811305126, -0.9997747315789454]], order='F')]

    ll_ref = [[-414.7628631782474, -410.049081775547],
              [-414.7559799310751, -410.0285309839064]]

    for p in range(1, 3):
        order = (p, 1, 1)

        batched_model = arima.ARIMA(y, order, fit_intercept=True)
        batched_model.fit()

        print("num iterations: ", batched_model.niter)

        x = arima.pack(order, (0, 0, 0, 0), 1, 2, batched_model.get_params())

        llx = arima.ll_f(2, len(t), order, 1, y, x, trans=False)

        rtol = 1e-2
        # parameter differences are more difficult to test precisely due to the
        # nonlinear-optimization.
        np.testing.assert_allclose(batched_model.mu, mu_ref[p-1], rtol=rtol)
        np.testing.assert_allclose(batched_model.ar, ar_ref[p-1],
                                   rtol=rtol)
        np.testing.assert_allclose(batched_model.ma, ma_ref[p-1],
                                   rtol=rtol)

        # more important is that the loglikelihood is close to a relatively
        # higher tolerance.
        np.testing.assert_allclose(llx, ll_ref[p-1], rtol=1e-6)


def test_predict(plot=False):
    """Test the `predict_in_sample()` function using provided parameters"""
    _, y = get_data()

    mu = [np.array([-217.7230173548441, -206.81064091237104]),
          np.array([-217.72325384510506, -206.77224439903458])]
    ar = [np.array([[0.0309380078339684, -0.0371740508810001]], order='F'),
          np.array([[0.0309027562133337, -0.0386322768036704],
                    [-0.0191533926207157, -0.0330133336831984]], order='F')]
    ma = [np.array([[-0.9995474311219695, -0.9995645146854383]], order='F'),
          np.array([[-0.999629811305126, -0.9997747315789454]], order='F')]

    l2err_ref = [[7.611525998416604e+08, 7.008862739645946e+08],
                 [7.663156224285843e+08, 6.993847054122686e+08]]

    for p in range(1, 3):
        order = (p, 1, 1)

        model = arima.ARIMA(y, order, fit_intercept=True)
        model.set_params({"mu": mu[p-1], "ar": ar[p-1], "ma": ma[p-1]})

        d_y_b_p = model.predict_in_sample()
        y_b_p = input_to_host_array(d_y_b_p).array

        if plot:
            import matplotlib.pyplot as plt
            nb_plot = 2
            fig, axes = plt.subplots(nb_plot, 1)
            axes[0].plot(t, y[:, 0], t, y_b_p[:, 0], "r-")
            axes[1].plot(t, y[:, 1], t, y_b_p[:, 1], "r-")
            if p == 1:
                axes[0].plot(t, yp_ref[p-1][0], "g--")
                axes[1].plot(t, yp_ref[p-1][1], "g--")
            plt.show()

        l2_error_predict = np.sum((y_b_p - y)**2, axis=0)
        np.testing.assert_allclose(l2err_ref[p-1], l2_error_predict)
        if p == 1:
            np.testing.assert_allclose(y_b_p[:, 0], yp_ref[0])
            np.testing.assert_allclose(y_b_p[:, 1], yp_ref[1])


def test_forecast():
    """Test forecast using provided parameters"""
    _, y = get_data()

    mu = [np.array([-217.7230173548441, -206.81064091237104]),
          np.array([-217.72325384510506, -206.77224439903458])]
    ar = [np.array([[0.0309380078339684, -0.0371740508810001]], order='F'),
          np.array([[0.0309027562133337, -0.0386322768036704],
                    [-0.0191533926207157, -0.0330133336831984]], order='F')]
    ma = [np.array([[-0.9995474311219695, -0.9995645146854383]], order='F'),
          np.array([[-0.999629811305126, -0.9997747315789454]], order='F')]

    y_fc_ref = [np.array([[8291.97380664, 7993.55508519, 7773.33550351],
                          [7648.10631132, 7574.38185979, 7362.6238661]]),
                np.array([[7609.91057747, 7800.22971962, 7473.00968599],
                          [8016.79544837, 7472.39902223, 7400.83781943]])]

    for p in range(1, 3):
        order = (p, 1, 1)

        model = arima.ARIMA(y, order, fit_intercept=True)
        model.set_params({"mu": mu[p-1], "ar": ar[p-1], "ma": ma[p-1]})

        d_y_b_fc = model.forecast(3)
        y_b_fc = input_to_host_array(d_y_b_fc).array

        np.testing.assert_allclose(y_fc_ref[p-1], y_b_fc.T)


def test_fit_predict_forecast(plot=False):
    """Full integration test: Tests fit followed by in-sample prediction and
    out-of-sample forecast

    """
    np.set_printoptions(precision=16)

    _, y = get_data()

    ns_train = 35
    ns_test = len(t) - ns_train

    y_b_p = []
    y_f_p = []

    for p in range(1, 3):
        order = (p, 1, 1)

        nb = 2

        y_train = np.zeros((ns_train, nb))
        for i in range(nb):
            y_train[:, i] = y[:ns_train, i]

        p, _, _ = order

        batched_model = arima.ARIMA(y_train, order, fit_intercept=True)
        batched_model.fit()

        d_y_b = batched_model.predict_in_sample()
        y_b = input_to_host_array(d_y_b).array
        d_y_fc = batched_model.forecast(ns_test)
        y_fc = input_to_host_array(d_y_fc).array

        y_b_p.append(y_b)
        y_f_p.append(y_fc)

    if plot:
        import matplotlib.pyplot as plt
        nb_plot = 2
        _, axes = plt.subplots(nb_plot, 1)
        axes[0].plot(t, y[:, 0], t[:ns_train], y_b_p[0][:, 0], "r-",
                     t[ns_train-1:-1], y_f_p[0][:, 0], "--")
        axes[0].plot(t[:ns_train], y_b_p[1][:, 0], "g-",
                     t[ns_train-1:-1], y_f_p[1][:, 0], "y--")
        axes[0].plot(t, yp_ref[0], "b--")
        axes[1].plot(t, y[:, 1], t[:ns_train], y_b_p[0][:, 1], "r-",
                     t[ns_train-1:-1], y_f_p[0][:, 1], "--")
        axes[1].plot(t[:ns_train], y_b_p[1][:, 1], "g-",
                     t[ns_train-1:-1], y_f_p[1][:, 1], "y--")
        axes[1].plot(t, yp_ref[1], "b--")

        plt.show()

    l2_error_predict0 = np.sum((y_b_p[0][:, :] - y[:ns_train, :])**2, axis=0)
    l2_error_predict1 = np.sum((y_b_p[1][:, :] - y[:ns_train, :])**2, axis=0)

    l2_error_ref0 = [5.1819845778009456e+08, 4.4313075823450834e+08]
    l2_error_ref1 = [5.4015810529295897e+08, 4.6489505018349826e+08]

    l2_error_forecast0 = np.sum((y_f_p[0][:, :] - y[ns_train-1:-1, :])**2,
                                axis=0)
    l2_error_forecast1 = np.sum((y_f_p[1][:, :] - y[ns_train-1:-1, :])**2,
                                axis=0)

    l2_error_fc_ref0 = [2.7841860168252653e+08, 2.4003239604745972e+08]
    l2_error_fc_ref1 = [3.728470033076098e+08, 3.039953059636233e+08]

    rtol = 5e-5
    np.testing.assert_allclose(l2_error_predict0, l2_error_ref0, rtol=rtol)
    np.testing.assert_allclose(l2_error_predict1, l2_error_ref1, rtol=rtol)
    rtol = 1e-3
    np.testing.assert_allclose(l2_error_forecast0, l2_error_fc_ref0, rtol=rtol)
    np.testing.assert_allclose(l2_error_forecast1, l2_error_fc_ref1, rtol=rtol)


def test_grid_search(num_batches=2):
    """Tests grid search using random data over the default range of p,q
    parameters"""
    ns = len(t)
    y_b = np.zeros((ns, num_batches))

    for i in range(num_batches):
        y_b[:, i] = np.random.normal(size=ns, scale=2000) + data_smooth

    best_order, best_mu, best_ar, best_ma, best_ic = arima.grid_search(y_b,
                                                                       d=1)

    if num_batches == 2:
        np.testing.assert_array_equal(best_order, [(0, 1, 1), (0, 1, 1)])


def demo():
    """Demo example from the documentation"""
    import matplotlib.pyplot as plt
    num_samples = 200
    xs = np.linspace(0, 1, num_samples)
    np.random.seed(12)
    noise = np.random.normal(scale=0.05, size=num_samples)
    noise2 = np.random.normal(scale=0.05, size=num_samples)
    ys1 = noise + 0.5*xs + 0.1*np.sin(xs/np.pi)
    ys2 = noise2 + 0.25*xs + 0.15*np.sin(0.8*xs/np.pi)
    ys = np.zeros((num_samples, 2))
    ys[:, 0] = ys1
    ys[:, 1] = ys2

    plt.plot(xs, ys1, xs, ys2)

    model = arima.ARIMA(ys, (1, 1, 1), fit_intercept=True)
    model.fit()

    d_yp = model.predict_in_sample()
    yp = input_to_host_array(d_yp).array
    d_yfc = model.forecast(50)
    yfc = input_to_host_array(d_yfc).array
    dx = xs[1] - xs[0]
    xfc = np.linspace(1, 1+50*dx, 50)
    plt.plot(xs, yp)
    plt.plot(xfc, yfc)


def bench_arima(num_batches=240, plot=False):
    """A parameterized benchmark allowing an arbitrarily large number of similar,
    but not identical batches"""

    ns = len(t)
    y_b = np.zeros((ns, num_batches))

    for i in range(num_batches):
        y_b[:, i] = np.random.normal(size=ns, scale=2000) + data_smooth

    p, d, q = (1, 1, 1)
    order = (p, d, q)

    start = timer()

    batched_model = arima.ARIMA(y_b, order, fit_intercept=True)
    batched_model.fit()

    end = timer()

    print("GPU Time ({} batches) = {} s".format(num_batches, end - start))
    print("Solver iterations (max/min/avg): ",
          np.max(batched_model.niter),
          np.min(batched_model.niter),
          np.mean(batched_model.niter))

    d_yt_b = batched_model.predict_in_sample()
    yt_b = input_to_host_array(d_yt_b).array

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(t, y_b[:, 0], "k-", t, yt_b[:, 0], "r--",
                 t, data0, "g--",
                 t, data_smooth, "y--")
        plt.show()
