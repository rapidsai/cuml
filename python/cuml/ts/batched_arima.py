import numpy as np

from typing import List, Tuple
from .arima import ARIMAModel, loglike, predict_in_sample
from .kalman import init_kalman_matrices
from .batched_kalman import batched_kfilter
import scipy.optimize as opt
from IPython.core.debugger import set_trace


def batched_fit(v_y, order: Tuple[int, int, int], mu0: float,
                arparams0, maparams0) -> List[ARIMAModel]:

    num_series = len(v_y)
    p, d, q = order
    num_parameters = 1 + p + q

    def bf(x):

        fsum = 0.0
        models = []
        for (i, y) in enumerate(v_y):
            xi = x[i*num_parameters:(i+1)*num_parameters]
            mu = xi[0]
            arparams = xi[1:p+1]
            maparams = xi[p+1:]
            model0 = ARIMAModel(order, mu, arparams, maparams, y)
            models.append(model0)

        ll_b = batched_loglike(models)
        for lli in ll_b:
            fsum += lli
        # note: we minimize the negative log likelihood
        return -fsum

    x0 = np.r_[mu0, arparams0, maparams0]
    x0 = np.tile(x0, num_series)

    x, f_final, res = opt.fmin_l_bfgs_b(bf, x0, approx_grad=True)

    models = []
    for i in range(num_series):
        xi = x[i*num_parameters:(i+1)*num_parameters]
        mu = xi[0]
        arparams = xi[1:p+1]
        maparams = xi[p+1:]
        model = ARIMAModel(order, mu, arparams, maparams, v_y[i])
        yp = predict_in_sample(model)
        model.ll = loglike(model)
        model.pred = yp
        models.append(model)

    return models


def init_batched_kalman_matrices(b_ar_params, b_ma_params):

    Zb = []
    Rb = []
    Tb = []
    for (ari, mai) in zip(b_ar_params, b_ma_params):
        Z, R, T, r = init_kalman_matrices(ari, mai)
        Zb.append(Z)
        Rb.append(R)
        Tb.append(T)

    return Zb, Rb, Tb, r


def batched_loglike(models: List[ARIMAModel], gpu: bool=True):
    b_ar_params = [model.ar_params for model in models]
    b_ma_params = [model.ma_params for model in models]
    Zb, Rb, Tb, r = init_batched_kalman_matrices(b_ar_params, b_ma_params)
    y_c = []

    for model in models:

        y_diff = np.diff(model.endog)

        B0 = model.mu/(1-np.sum(model.ar_params))
        y_centered = y_diff - B0
        y_c.append(y_centered)

    vs, Fs, ll, s2 = batched_kfilter(y_c, Zb, Rb, Tb, r, gpu)
    return ll
