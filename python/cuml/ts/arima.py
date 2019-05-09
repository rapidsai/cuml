import numpy as np
from typing import Tuple
from IPython.core.debugger import set_trace
import scipy.optimize as opt

from .kalman import kfilter, init_kalman_matrices


class ARIMAModel:
    r"""
    The ARIMA model fits the following to a given input:
    if d=1:
      \delta \tilde{y}_{t} = \mu + \sum_{i=1}^{p} \phi_i \delta y_{t-i}
                                    + \sum_{i=1}^{q} \theta_i (y_{t-i} -
                                                                 \tilde{y}_{t-i})

    Note all fitted parameters, \mu, \phi_i, \theta_i

    """

    def __init__(self, order, mu, ar_params, ma_params, endog):
        self.order = order
        self.ar_params = ar_params
        self.ma_params = ma_params
        self.mu = mu
        self.endog = endog
        self.pred = None
        self.ll = None

    def __repr__(self):
        return "ARIMA Model {}, mu:{}, ar:{}, ma:{}".format(self.order, self.mu,
                                                            self.ar_params, self.ma_params)

    def __str__(self):
        return self.__repr__()

    @property
    def model_complexity(self):
        (p, d, q) = self.order
        # complexity is number of parameters: mu + ar + ma
        return 1 + p + q

    @property
    def bic(self):
        return -2 * self.ll + np.log(len(self.endog)) * (self.model_complexity)
    @property
    def aic(self):
        return -2 * self.ll + 2 * (self.model_complexity)

def undifference(x, x0):
    xi = np.cumsum(x)
    return np.append([x0], xi)


def diff(y, d=1, D=0, s=None):
    if D > 0:
        if s is None:
            raise AssertionError("If seasonal D>0, seasonal lag 's' must be set")

        y_diff = y[s:] - y[:-s]

    y_diff = np.diff(y_diff, d)

    return y_diff


def error_ls(model):

    yt = eval_ls(model)
    error_l2 = np.sqrt(np.sum((yt-model.endog)**2))
    return error_l2


def eval_ls(model):

    p, d, q = model.order
    init_offset = max(p, q)

    init_offset += d

    y = model.endog
    yt = np.zeros(len(y))

    # initialize first values to mean
    yt[0:init_offset] = np.mean(y[0:init_offset])

    for t in range(init_offset, len(y)):
        if d == 0:
            y[t] = model.mu + (np.dot(yt[t-p:t], model.ar_params) -
                               np.dot(y[t-q:t]-yt[t-q:t], model.ma_params))
        elif d == 1:
            ytdiff_ar = yt[t-p:t] - yt[t-p-1:t-1]
            ydiff_ar = y[t-p:t] - y[t-p-1:t-1]
            ar = np.dot(ydiff_ar-ytdiff_ar, model.ar_params)
            ma = np.dot(y[t-q:t] - yt[t-q:t], model.ma_params)
            yt[t] = model.mu + yt[t-1]
            if ar.size:
                yt[t] += ar
            if ma.size:
                yt[t] -= ma
        else:
            raise NotImplementedError("ARIMA only implemented for d=0,1")

    return yt


def eval_kf(model: ARIMAModel):
    Z, R, T, r = init_kalman_matrices(model.ar_params, model.ma_params)
    y_diff = np.diff(model.endog)

    # P&K TSM suggests this:
    # B0 = model.mu/(1-np.sum(model.ar_params))
    # statsmodels uses this
    B0 = model.mu
    y_centered = y_diff - B0
    vs, ll = kfilter(y_centered, Z, R, T, r)

    return vs, model.endog[0:-1]+(y_diff - vs)


def loglike(model: ARIMAModel):
    Z, R, T, r = init_kalman_matrices(model.ar_params, model.ma_params)
    y_diff = np.diff(model.endog)

    # P&K TSM suggests this:
    # B0 = model.mu/(1-np.sum(model.ar_params))
    # statsmodels uses this
    B0 = model.mu
    y_centered = y_diff - B0
    vs, ll= kfilter(y_centered, Z, R, T, r)
    return ll


def predict_in_sample(model: ARIMAModel, method="mle"):

    if method == "mle":

        vs, yt = eval_kf(model)

        # extend prediction by 1 and return
        num_previous_values = max(1, len(model.ar_params)+1)
        y_fc = forecast(model, 1, prev_values=yt[-num_previous_values:])
        return np.append(yt, y_fc)

    elif method == "ls":
        yt = eval_ls(model)
        return yt
    else:
        raise NotImplementedError("Please choose 'mle' or 'ls'")


def forecast(model, nsteps, prev_values=None, method="mle"):
    return forecast_values(model.order, model.pred, model.ar_params, model.mu,
                           nsteps, prev_values, method)

def forecast_values(order, pred, ar_params, mu, nsteps, prev_values, method="mle"):

    p, d, _ = order

    if prev_values is not None:
        nop = len(prev_values)
    else:
        nop = len(pred)

    if nop < 1 or nop < len(ar_params):
        raise AssertionError("Forecast ERROR: Not enough previous values to compute forecast")

    yp = np.zeros(nop + nsteps)

    # if given previous values, use those to for initial previous values in
    # forecast, otherwise use in-sample prediction created when fitting model
    if prev_values is not None:
        yp[0:nop] = prev_values
    else:
        yp[0:nop] = pred

    # evaluate model
    for t in range(nop, nop + nsteps):
        if d == 0:
            yp[t] = mu + np.dot(yp[t-len(ar_params):t],
                                      ar_params)
        elif d == 1:
            # set_trace()
            yp[t] = yp[t-1] + mu
            if ar_params.size:
                ypdiff = yp[t-p:t] - yp[t-p-1:t-1]
                ar = np.dot(ypdiff, ar_params)
                yp[t] += ar
        else:
            raise NotImplementedError("ARIMA only works for d=0,1")

    # only return forecast
    return yp[nop:nop+nsteps]


def fit(y, order: Tuple[int, int, int], mu0: float, arparams0, maparams0,
        method="mle") -> ARIMAModel:

    def f(x):
        mu = x[0]
        arparams = x[1:len(arparams0)+1]
        maparams = x[len(arparams0)+1:]
        model0 = ARIMAModel(order, mu, arparams, maparams, y)
        if method == "ls":
            e = error_ls(model0)
            return e
        elif method == "mle":
            # maximize log likelihood (and minimize the negative...)
            ll = loglike(model0)
            return -ll/len(y)
            # ll, gll = loglike_with_grad(model0)
            # return -ll, -np.array(gll)
        else:
            raise NotImplementedError("Options 'ls'/'mle'")

    x0 = np.append([mu0], arparams0)
    x0 = np.append(x0, maparams0)
    # TODO:
    # Consider the following gradient approximation
    # http://mdolab.utias.utoronto.ca/resources/complex-step
    # def complex_step_grad(f, x, h=1.0e-20):
    #     dim = np.size(x)
    #     increments = np.identity(dim) * 1j * h
    #     partials = [f(x+ih).imag / h for ih in increments]
    #     return np.array(partials)

    x, f_final, res = opt.fmin_l_bfgs_b(f, x0, approx_grad=True, m=12, pgtol=1e-8,
                                        factr=100, iprint=5)
    mu = x[0]
    arparams = x[1:len(arparams0)+1]
    maparams = x[len(arparams0)+1:]
    print("mu={}, ar={}, ma={}".format(mu, arparams, maparams))
    model = ARIMAModel(order, mu, arparams, maparams, y)
    # put the prediction and loglikelihood into the model for use in
    # forecasting
    yp = predict_in_sample(model)
    model.ll = loglike(model)
    model.pred = yp
    return model
