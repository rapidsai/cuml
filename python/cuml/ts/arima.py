from typing import List, Tuple
import numpy as np
from IPython.core.debugger import set_trace
import pandas as pd

from .batched_kalman import batched_kfilter
from .batched_kalman import pynvtx_range_push, pynvtx_range_pop
from .batched_kalman import batched_transform as batched_trans_cuda
from .batched_kalman import pack, unpack
from .batched_lbfgs import batched_fmin_lbfgs_b


class ARIMAModel:
    r"""
    The Batched ARIMA model fits the following to each given input:
    if d=1:
      \delta \tilde{y}_{t} = \mu + \sum_{i=1}^{p} \phi_i \delta y_{t-i}
                                    + \sum_{i=1}^{q} \theta_i (y_{t-i} -
                                                                 \tilde{y}_{t-i})

    Note all fitted parameters, \mu, \phi_i, \theta_i.
    """

    def __init__(self, order: List[Tuple[int, int, int]],
                 mu: np.ndarray,
                 ar_params: List[np.ndarray],
                 ma_params: List[np.ndarray],
                 y: pd.DataFrame):
        self.order = order
        self.mu = mu
        self.ar_params = ar_params
        self.ma_params = ma_params
        self.y = y
        self.num_samples = y.shape[0]  # pandas Dataframe shape is (num_batches, num_samples)
        self.num_batches = y.shape[1]
        self.yp = None
        self.niter = None # number of iterations used during fit

    def __repr__(self):
        return "Batched ARIMA Model {}, mu:{}, ar:{}, ma:{}".format(self.order, self.mu,
                                                                    self.ar_params, self.ma_params)

    def __str__(self):
        return self.__repr__()

    @property
    def bic(self):
        llb = loglike(self)
        return [-2 * lli + np.log(len(self.y)) * (_model_complexity(self.order[i]))
                for (i, lli) in enumerate(llb)]

    @property
    def aic(self):
        llb = loglike(self)
        return [-2 * lli + 2 * (_model_complexity(self.order[i]))
                for (i, lli) in enumerate(llb)]


def _model_complexity(order):
    (p, _, q) = order
    # complexity is number of parameters: mu + ar + ma
    return 1 + p + q


def ll_f(num_batches, num_parameters, order, y, x, return_negative_sum=False, gpu=True, trans=False):
    """Computes batched loglikelihood given parameters stored in `x`."""
    pynvtx_range_push("ll_f")

    # Apply stationarity-inducing transform.
    if trans:
        pynvtx_range_push("ll_f_trans")
        p, _, q = order
        x = batch_trans(p, q, num_batches, np.copy(x))
        pynvtx_range_pop()

    p, _, q = order
    mu, arparams, maparams = unpack(p, q, num_batches, x)
    
    b_model = ARIMAModel([order]*num_batches,
                         mu,
                         arparams,
                         maparams,
                         y)

    ll_b = loglike(b_model)
    pynvtx_range_pop()
    if return_negative_sum:
        return -ll_b.sum()
    else:
        return ll_b

def ll_gf(num_batches, num_parameters, order, y, x, h=1e-8, gpu=True, trans=False):
    """Computes fd-gradient of batched loglikelihood given parameters stored in
    `x`. Because batches are independent, it only compute the function for the
    single-batch number of parameters."""
    pynvtx_range_push("ll_gf")
    
    fd = np.zeros(num_parameters)

    grad = np.zeros(len(x))

    assert (len(x) / num_parameters) == float(num_batches)
    for i in range(num_parameters):
        fd[i] = h

        # duplicate the perturbation across batches (they are independent)
        fdph = np.tile(fd, num_batches)

        # reset perturbation
        fd[i] = 0.0

        ll_b_ph = ll_f(num_batches, num_parameters, order, y, x+fdph, gpu=gpu, trans=trans)
        ll_b_mh = ll_f(num_batches, num_parameters, order, y, x-fdph, gpu=gpu, trans=trans)
        # ll_b0 = BatchedARIMAModel.ll_f(num_batches, num_parameters, order, y, x, gpu=gpu, trans=trans)
        np.seterr(all='raise')
        grad_i_b = (ll_b_ph - ll_b_mh)/(2*h)
        # grad_i_b = (ll_b_ph - ll_b0)/(h)

        if num_batches == 1:
            grad[i] = grad_i_b
        else:
            assert len(grad[i::num_parameters]) == len(grad_i_b)
            # Distribute the result to all batches
            grad[i::num_parameters] = grad_i_b

    pynvtx_range_pop()
    return grad


def fit(y: np.ndarray,
        order: Tuple[int, int, int],
        mu0: np.ndarray,
        ar_params0: List[np.ndarray],
        ma_params0: List[np.ndarray],
        opt_disp=-1, h=1e-9, gpu=True, alpha_max=1000):
    """
    Fits the ARIMA model to each time-series (batched together in a dense numpy matrix)
    with the given initial parameters. `y` is (num_samples, num_batches)

    """

    # turn on floating point exceptions!
    np.seterr(all='raise')

    p, d, q = order
    num_parameters = 1 + p + q

    num_samples = y.shape[0]  # pandas Dataframe shape is (num_batches, num_samples)
    num_batches = y.shape[1]

    def f(x: np.ndarray) -> np.ndarray:
        """The (batched) energy functional returning the negative loglikelihood (for each series)."""

        # Recall: Maximimize LL means minimize negative
        n_llf = -(ll_f(num_batches, num_parameters, order, y, x, gpu=gpu, trans=True))
        return n_llf/(num_samples-1)


    # optimized finite differencing gradient for batches
    def gf(x):
        """The gradient of the (batched) energy functional."""
        # Recall: We maximize LL by minimizing -LL
        n_gllf = -ll_gf(num_batches, num_parameters, order, y, x, h, gpu=gpu, trans=True)
        return n_gllf/(num_samples-1)

    x0 = pack(p, q, num_batches, mu0, ar_params0, ma_params0)
    x0 = batch_invtrans(p, q, num_batches, x0)
        
    # check initial parameter sanity
    if ((np.isnan(x0).any()) or (np.isinf(x0).any())):
            raise FloatingPointError("Initial condition 'x0' has NaN or Inf.")    


    # Optimize parameters by minimizing log likelihood.
    x, niter, flags = batched_fmin_lbfgs_b(f, x0, num_batches, gf,
                                          iprint=opt_disp, factr=1000)

    # TODO: Better Handle non-zero `flag` array values: 0 -> ok, 1,2 -> optimizer had trouble
    if (flags != 0).any():
        print("WARNING(`fit()`): Some batch members had optimizer problems.")

    Tx = batch_trans(p, q, num_batches, x)
    mu, ar, ma = unpack(p, q, num_batches, Tx)

    fit_model = ARIMAModel(num_batches*[order], mu, ar, ma, y)
    fit_model.niter = niter
    return fit_model


def diffAndCenter(y: np.ndarray, num_batches: int,
                  mu: np.ndarray, ar_params: np.ndarray):
    """Diff and center batched series `y`"""
    y_diff = np.diff(y, axis=0)

    B0 = np.zeros(num_batches)
    for (i, (mu, ar)) in enumerate(zip(mu, ar_params)):
        # B0[i] = mu/(1-np.sum(ar))
        B0[i] = mu

    return np.asfortranarray(y_diff-B0)


def run_kalman(model, initP_kalman_iterations=False) -> Tuple[np.ndarray, np.ndarray]:
    """Run the (batched) kalman filter for the given model (and contained batched
    series). `initP_kalman_iterations, if true uses kalman iterations, and if false
    uses an analytical approximation (Durbin Koopman pg 138).`"""
    b_ar_params = model.ar_params
    b_ma_params = model.ma_params

    assert_same_d(model.order) # We currently assume the same d for all series
    p, d, q = model.order[0]

    if d == 0:

        ll_b, vs = batched_kfilter(np.asfortranarray(model.y), # numpy
                                   b_ar_params,
                                   b_ma_params,
                                   p, q,
                                   initP_kalman_iterations)
    elif d == 1:

        y_diff_centered = diffAndCenter(model.y, model.num_batches,
                                                          model.mu, model.ar_params)


        ll_b, vs = batched_kfilter(y_diff_centered, # numpy
                                   b_ar_params,
                                   b_ma_params,
                                   p, q,
                                   initP_kalman_iterations)
    else:
        raise NotImplementedError("ARIMA only support d==0,1")

    return ll_b, vs


def loglike(model) -> np.ndarray:
    """Compute the batched loglikelihood (return a LL for each batch)"""
    ll_b, _ = run_kalman(model)
    return ll_b


def predict_in_sample(model):
    """Return in-sample prediction on batched series given batched model"""
    _, vs = run_kalman(model)

    assert_same_d(model.order) # We currently assume the same d for all series
    _, d, _ = model.order[0]

    if d == 0:
        y_p = model.y - vs
    elif d == 1:
        y_diff = np.diff(model.y, axis=0)
        # Following statsmodel `predict(typ='levels')`, by adding original
        # signal back to differenced prediction, we retrive a prediction of
        # the original signal.
        predict = (y_diff - vs)
        y_p = model.y[0:-1, :] + predict
    else:
        # d>1
        raise NotImplementedError("Only support d==0,1")

    # Extend prediction by 1 when d==1
    if d == 1:
        # forecast a single value to make prediction length of original signal
        fc1 = np.zeros(model.num_batches)
        for i in range(model.num_batches):
            fc1[i] = fc_single(1, model.order[i], y_diff[:,i],
                               vs[:,i], model.mu[i],
                               model.ma_params[i],
                               model.ar_params[i])

        final_term = model.y[-1, :] + fc1

        # append final term to prediction
        temp = np.zeros((y_p.shape[0]+1, y_p.shape[1]))
        temp[:-1, :] = y_p
        temp[-1, :] = final_term
        y_p = temp

    model.yp = y_p
    return y_p

def fc_single(num_steps, order, y_diff, vs, mu, ma_params, ar_params):

    p, _, q = order

    y_ = np.zeros(p+num_steps)
    vs_ = np.zeros(q+num_steps)
    if p>0:
        y_[:p] = y_diff[-p:]
    if q>0:
        vs_[:q] = vs[-q:]

    fcast = np.zeros(num_steps)

    for i in range(num_steps):
        mu_star = mu * (1-ar_params.sum())
        fcast[i] = mu_star
        if p > 0:
            fcast[i] += np.dot(ar_params, y_[i:i+p])
        if q > 0 and i < q:
            fcast[i] += np.dot(ma_params, vs_[i:i+q])
        if p > 0:
            y_[i+p] = fcast[i]

    return fcast


def forecast(model, nsteps: int) -> np.ndarray:
    """Forecast the given model `nsteps` into the future."""
    y_fc_b = np.zeros((nsteps, model.num_batches))

    _, vs = run_kalman(model)

    for i in range(model.num_batches):
        p, d, q = model.order[i]
        vsi = vs[:,i]
        ydiff_i = np.diff(model.y[:, i],axis=0)
        fc = fc_single(nsteps, (p,d,q), ydiff_i, vsi,
                       model.mu[i], model.ma_params[i],
                       model.ar_params[i])

        if model.order[i][1] > 0: # d > 0
            fc = undifference(fc, model.y[-1,i])[1:]

        y_fc_b[:, i] = fc[:]

    return y_fc_b


def batch_trans(p, q, nb, x):
    """Apply the stationarity/invertibility guaranteeing transform to batched-parameter vector x."""
    pynvtx_range_push("jones trans")
    mu, ar, ma = unpack(p, q, nb, x)

    (ar2, ma2) = batched_trans_cuda(p, q, nb, ar, ma, False)
    
    Tx = pack(p, q, nb, mu, ar2, ma2)
    pynvtx_range_pop()
    return Tx


def batch_invtrans(p, q, nb, x):
    """Apply the *inverse* stationarity/invertibility guaranteeing transform to
       batched-parameter vector x.
    """
    pynvtx_range_push("jones inv-trans")
    mu, ar, ma = unpack(p, q, nb, x)

    (ar2, ma2) = batched_trans_cuda(p, q, nb, ar, ma, True)

    Tx = pack(p, q, nb, mu, ar2, ma2)
    pynvtx_range_pop()
    return Tx


def undifference(x, x0):
    # set_trace()
    xi = np.append(x0, x)
    return np.cumsum(xi)


def assert_same_d(b_order):
    """Checks that all values of d in batched order are same"""
    b_d = [d for _, d, _ in b_order]
    assert (np.array(b_d) == b_d[0]).all()


def start_params(order, y_diff):
    """A quick approach to determine reasonable starting mu (trend), AR, and MA parameters"""

    # y is mutated so we need a copy
    y = np.copy(y_diff)
    nobs = len(y)

    p, q, d = order
    params_init = np.zeros(p+q+d)
    if d > 0:
        # center y (in `statsmodels`, this is result when exog = [1, 1, 1...])
        mean_y = np.mean(y)
        params_init[0] = mean_y
        y -= mean_y

    if p == 0 and q == 0:
        return params_init

    if p != 0:

        x = np.zeros((len(y) - p, p))

        # create lagged series set
        for lag in range(1, p+1):
            # create lag and trim appropriately from front so they are all the same size
            x[:, lag-1] = y[p-lag:-lag].T

        # LS fit a*X - Y
        y_ar = y[p:]
        (ar_fit, _, _, _) = np.linalg.lstsq(x, y_ar.T, rcond=None)

        if q == 0:
            params_init[d:] = ar_fit
        else:
            residual = y[p:] - np.dot(x, ar_fit)

            x_resid = np.zeros((len(residual) - q, q))
            x_ar2 = np.zeros((len(residual) - q, p))

            # create lagged residual and ar term
            for lag in range(1, q+1):
                x_resid[:, lag-1] = residual[q-lag:-lag].T
            for lag in range(1, p+1):
                x_ar2[:, lag-1] = (y[p-lag:-lag].T)[q:]

            X = np.column_stack((x_ar2, x_resid))
            (arma_fit, _, _, _) = np.linalg.lstsq(X, y_ar[q:].T, rcond=None)

            params_init[d:] = arma_fit

    else:
        # case when p == 0 and q>0

        # when p==0, MA params are often -1
        # TODO: See how `statsmodels` handles this case
        params_init[d:] = -1*np.ones(q)

    return params_init

def init_x0(order, y):
    pynvtx_range_push("init x0")
    (p, d, q) = order
    if d == 1:
        yd = np.diff(y)
    else:
        yd = np.copy(y)
    
    x0 = start_params((p, q, d), yd)

    mu, ar, ma = unpack(p, q, 1, x0)
    # fix ma to avoid bad values in inverse invertibility transform
    for i in range(len(ma[0])):
        mai = ma[0][i]
        # if ma >= 1, then we get "inf" results from inverse transform
        ma[0][i] = np.sign(mai)*min(np.abs(mai), 1-1e-14)
    x0 = pack(p, q, 1, mu, ar, ma)
    pynvtx_range_pop()
    return x0

def grid_search(y_b: np.ndarray, d=1, max_p=3, max_q=3, method="aic"):
    """Grid search to find optimal (lowest `ic`) (p,_,q) values for each
    time-series in y_b, which is a dense `ndarray` with columns as time.
    Optimality is based on minimizing AIC or BIC, which both sum negative
    log-likelihood against model complexity; Higher model complexity might
    yield a lower negative LL, but at higher `aic` due to complexity term.

    """

    num_batches = y_b.shape[1]
    best_ic = np.full(num_batches, np.finfo(np.float64).max/2)
    best_model = ARIMAModel([[]]*num_batches, np.zeros(num_batches), [[]]*num_batches, [[]]*num_batches, y_b)
    # best_model =

    for p in range(0, max_p):
        arparams = np.zeros(p)
        for q in range(0, max_q):
            maparams = np.zeros(q)

            # skip 0,0 case
            if p == 0 and q == 0:
                continue

            b_model = fit(y_b, (p, d, q), 0.0, arparams, maparams)

            if method == "aic":
                ic = b_model.aic
            elif method == "bic":
                ic = b_model.bic
            else:
                raise NotImplementedError("Method '{}' not supported".format(method))

            for (i, ic_i) in enumerate(ic):
                if ic_i < best_ic[i]:
                    best_model.order[i] = (p, d, q)
                    best_model.mu[i] = b_model.mu[i]
                    best_model.ar_params[i] = b_model.ar_params[i]
                    best_model.ma_params[i] = b_model.ma_params[i]
                    best_ic[i] = ic_i

    return (best_model, best_ic)
