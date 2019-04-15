import cuml.ts.arima as arima
import cuml.ts.batched_arima as batched_arima
import matplotlib.pyplot as plt
import numpy as np


def test_arima(plot=False, verbose=False, check_asserts=True):
    num_samples = 100
    xs = np.linspace(0, 1, num_samples)
    np.random.seed(12)
    noise = np.random.normal(scale=0.1, size=num_samples)
    ys = noise + 0.5*xs

    x_train, x_test = xs[0:66], xs[66:]
    y_train, y_test = ys[0:66], ys[66:]
    order = (0, 1, 1)
    ar_params0 = 0.1 * np.ones(order[0])
    ma_params0 = -1 * np.ones(order[2])
    rp_model_fit = arima.fit(y_train, order, 0.0,
                             ar_params0, ma_params0)

    y_rp = arima.predict_in_sample(rp_model_fit)
    y_rp_fc = arima.forecast(rp_model_fit, nsteps=len(y_test))

    l2err_in_sample = np.sqrt(np.sum((y_train - y_rp)**2))
    l2err_forecast = np.sqrt(np.sum((y_test - y_rp_fc)**2))

    if check_asserts:
        # mu=0.005771399176219675, ar=[], ma=[-1.00581135]
        np.testing.assert_approx_equal(rp_model_fit.mu, 0.005771399176219675, 1)
        np.testing.assert_allclose(rp_model_fit.ar_params, [], rtol=1e-8)
        np.testing.assert_allclose(rp_model_fit.ma_params,
                                   [-1.00581135], rtol=1e-2)

        np.testing.assert_approx_equal(l2err_in_sample, 0.823, 2)
        np.testing.assert_approx_equal(l2err_forecast, 0.684, 1)

    if verbose:
        print("model: ", rp_model_fit)
        print("ll:", rp_model_fit.ll)
        print("err in sample:", l2err_in_sample)
        print("err forecast:", l2err_forecast)
        print("BIC: ", rp_model_fit.bic)
        print("AIC: ", rp_model_fit.aic)

    if plot:
        plt.clf()
        plt.plot(xs, ys)
        plt.plot(x_train, y_rp)
        plt.plot(x_test, y_test)
        plt.plot(x_test, y_rp_fc)
        plt.show()


def test_batched_arima(plot=False, verbose=False, check_asserts=True):
    """
    Run batch test on same y to verify individual values
    """
    num_samples = 100
    xs = np.linspace(0, 1, num_samples)
    np.random.seed(12)
    noise = np.random.normal(scale=0.1, size=num_samples)
    ys = noise + 0.5*xs

    x_train, x_test = xs[0:66], xs[66:]
    y_train, y_test = ys[0:66], ys[66:]

    # get reference solution
    ref_model = arima.fit(y_train, (0,1,1), 0.0, np.array([]), np.array([-1.0]))
    y_rp_ref = arima.predict_in_sample(ref_model)
    y_rp_fc_ref = arima.forecast(ref_model, nsteps=len(y_test))
    l2err_in_sample_ref = np.sqrt(np.sum((y_train - y_rp_ref)**2))
    l2err_forecast_ref = np.sqrt(np.sum((y_test - y_rp_fc_ref)**2))

    # run batching
    num_batches = 10
    v_y = [y_train for i in range(num_batches)]
    models = batched_arima.batched_fit(v_y, (0, 1, 1), 0.0,
                                       np.array([]), np.array([-1.0]))

    for rp_model_fit in models:

        y_rp = arima.predict_in_sample(rp_model_fit)
        y_rp_fc = arima.forecast(rp_model_fit, nsteps=len(y_test))
        l2err_in_sample = np.sqrt(np.sum((y_train - y_rp)**2))
        l2err_forecast = np.sqrt(np.sum((y_test - y_rp_fc)**2))

        if check_asserts:
            # mu=0.005771399176219675, ar=[], ma=[-1.00581135]
            np.testing.assert_approx_equal(rp_model_fit.mu, ref_model.mu, 5)
            np.testing.assert_allclose(rp_model_fit.ar_params, [], rtol=1e-8)
            np.testing.assert_allclose(rp_model_fit.ma_params,
                                       ref_model.ma_params, rtol=1e-5)

            np.testing.assert_approx_equal(l2err_in_sample, l2err_in_sample_ref, 5)
            np.testing.assert_approx_equal(l2err_forecast, l2err_forecast_ref, 5)

        if verbose:
            print("model: ", rp_model_fit)
            print("err in sample:", l2err_in_sample)
            print("err forecast:", l2err_forecast)
            print("BIC: ", rp_model_fit.bic)

        if plot:
            plt.clf()
            plt.plot(xs, ys)
            plt.plot(x_train, y_rp)
            plt.plot(x_test, y_test)
            plt.plot(x_test, y_rp_fc)
            plt.show()


def test_arima_kalman():
    num_samples = 100
    xs = np.linspace(0, 1, num_samples)
    np.random.seed(12)
    noise = np.random.normal(scale=0.1, size=num_samples)
    ys = noise + 0.5*xs

    x_train, x_test = xs[0:66], xs[66:]
    y_train, y_test = ys[0:66], ys[66:]
    order = (0, 1, 1)
    mu = 0.0
    arparams = np.array([])
    maparams = np.array([-1.0])
    model0 = arima.ARIMAModel(order, mu, arparams, maparams, y_train)

    # single version
    ll = arima.loglike(model0)

    # batched version
    num_batches = 100
    models = [arima.ARIMAModel(order, mu, arparams, maparams, y_train) for i in range(num_batches)]
    b_ll = batched_arima.batched_loglike(models, gpu=True)
    b_ll_cpu = batched_arima.batched_loglike(models, gpu=False)

    print("ll vs b_ll: {} vs {}".format(ll, b_ll))
    for lli in b_ll:
        np.testing.assert_approx_equal(ll, lli)

if __name__ == "__main__":
    # test_arima()
    # test_batched_arima()
    test_arima_kalman()
