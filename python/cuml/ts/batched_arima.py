import numpy as np

from typing import List, Tuple
from .arima import ARIMAModel, loglike, predict_in_sample, forecast_values
from .kalman import init_kalman_matrices
from .batched_kalman import batched_kfilter
import scipy.optimize as opt
from IPython.core.debugger import set_trace
# import cudf
import pandas as pd

class BatchedARIMAModel:
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

    def __repr__(self):
        return "Batched ARIMA Model {}, mu:{}, ar:{}, ma:{}".format(self.order, self.mu,
                                                                    self.ar_params, self.ma_params)

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def model_complexity(order):
        (p, _, q) = order
        # complexity is number of parameters: mu + ar + ma
        return 1 + p + q

    @property
    def bic(self):
        llb = BatchedARIMAModel.loglike(self)
        return [-2 * lli + np.log(len(self.y)) * (BatchedARIMAModel.model_complexity(self.order[i]))
                for (i, lli) in enumerate(llb)]

    @property
    def aic(self):
        llb = BatchedARIMAModel.loglike(self)
        return [-2 * lli + 2 * (BatchedARIMAModel.model_complexity(self.order[i]))
                for (i, lli) in enumerate(llb)]

    @staticmethod
    def ll_f(num_batches, num_parameters, order, y, x, return_negative_sum=False):
        """Computes batched loglikelihood given parameters stored in `x`."""
        p, d, q = order
        mu = np.zeros(num_batches)
        arparams = []
        maparams = []
        for i in range(num_batches):
            xi = x[i*num_parameters:(i+1)*num_parameters]
            mu[i] = xi[0]
            arparams.append(xi[1:p+1])
            maparams.append(xi[p+1:])

        b_model = BatchedARIMAModel([order]*num_batches,
                                    mu,
                                    arparams,
                                    maparams,
                                    y)

        ll_b = BatchedARIMAModel.loglike(b_model)
        if return_negative_sum:
            return -ll_b.sum()
        else:
            return ll_b

    @staticmethod
    def ll_gf(num_batches, num_parameters, order, y, x, h=1e-8):
        """Computes fd-gradient of batched loglikelihood given parameters stored in
        `x`. Because batches are independent, it only compute the function for the
        single-batch number of parameters."""
        p, d, q = order
        mu = np.zeros(num_batches)
        arparams = []
        maparams = []
        for i in range(num_batches):
            xi = x[i*num_parameters:(i+1)*num_parameters]
            mu[i] = xi[0]
            arparams.append(xi[1:p+1])
            maparams.append(xi[p+1:])

        fd = np.zeros(num_parameters)

        grad = np.zeros(len(x))

        assert(len(x) / num_parameters == float(num_batches))
        for i in range(num_parameters):
            fd[i] = h

            # duplicate the perturbation across batches (they are independent)
            fdph = np.tile(fd, num_batches)

            # reset perturbation
            fd[i] = 0.0

            ll_b_ph = BatchedARIMAModel.ll_f(num_batches, num_parameters, order, y, x+fdph)
            ll_b_mh = BatchedARIMAModel.ll_f(num_batches, num_parameters, order, y, x-fdph)

            grad_i_b = (ll_b_ph - ll_b_mh)/(2*h)
            
            if num_batches == 1:
                grad[i] = grad_i_b
            else:
                assert(len(grad[i::num_parameters]) == len(grad_i_b))
                # Distribute the result to all batches
                grad[i::num_parameters] = grad_i_b

        return grad


    @staticmethod
    def fit(y: np.ndarray,
            order: Tuple[int, int, int],
            mu0: float,
            ar_params0: np.ndarray,
            ma_params0: np.ndarray):
        """
        Fits the ARIMA model to each time-series (batched together in a dense numpy matrix)
        with the given initial parameters. `y` is (num_samples, num_batches)

        """
        
        p, d, q = order
        num_parameters = 1 + p + q

        num_samples = y.shape[0]  # pandas Dataframe shape is (num_batches, num_samples)
        num_batches = y.shape[1]

        def f(x):
            # Maximimize LL means minimize negative
            return -(BatchedARIMAModel.ll_f(num_batches, num_parameters, order, y, x).sum())

        # optimized finite differencing gradient for batches
        def gf(x):
            # Recall: We maximize LL by minimizing -LL
            return -BatchedARIMAModel.ll_gf(num_batches, num_parameters, order, y, x)

        x0 = np.r_[mu0, ar_params0, ma_params0]
        x0 = np.tile(x0, num_batches)

        x, f_final, res = opt.fmin_l_bfgs_b(f, x0, fprime=gf, approx_grad=False, iprint=-1)

        mu = np.zeros(num_batches)
        ar = []
        ma = []
        for i in range(num_batches):
            xi = x[i*num_parameters:(i+1)*num_parameters]
            mu[i] = xi[0]
            ar.append(xi[1:p+1])
            ma.append(xi[p+1:])

        fit_model = BatchedARIMAModel(num_batches*[order], mu, ar, ma, y)
        

        return fit_model


    @staticmethod
    def diffAndCenter(y: np.ndarray, num_batches: int,
                      mu: np.ndarray, ar_params: np.ndarray):
        """Diff and center batched series `y`"""
        y_diff = np.diff(y, axis=0)

        B0 = np.zeros(num_batches)
        for (i, (mu, ar)) in enumerate(zip(mu, ar_params)):
            B0[i] = mu/(1-np.sum(ar))

        return np.asfortranarray(y_diff-B0)

    @staticmethod
    def run_kalman(model,
                   gpu=True, initP_kalman_iterations=False) -> Tuple[np.ndarray, np.ndarray]:
        """Run the (batched) kalman filter for the given model (and contained batched
        series). `initP_kalman_iterations, if true uses kalman iterations, and if false
        uses an analytical approximation.`"""
        b_ar_params = model.ar_params
        b_ma_params = model.ma_params
        Zb, Rb, Tb, r = init_batched_kalman_matrices(b_ar_params, b_ma_params)

        assert_same_d(model.order) # We currently assume the same d for all series
        _, d, _ = model.order[0]

        if d == 0:
            ll_b, vs = batched_kfilter(np.asfortranarray(model.y), # numpy
                                       # y_diff_centered.values, # pandas
                                       Zb, Rb, Tb,
                                       # Z_dense, R_dense, T_dense,
                                       r,
                                       gpu,initP_kalman_iterations)
        elif d == 1:
            
            y_diff_centered = BatchedARIMAModel.diffAndCenter(model.y, model.num_batches,
                                                              model.mu, model.ar_params)

            ll_b, vs = batched_kfilter(y_diff_centered, # numpy
                                       # y_diff_centered.values, # pandas
                                       Zb, Rb, Tb,
                                       # Z_dense, R_dense, T_dense,
                                       r,
                                       gpu,initP_kalman_iterations)
        else:
            raise NotImplementedError("ARIMA only support d==0,1")

        return ll_b, vs

    @staticmethod
    def loglike(model, gpu=True) -> np.ndarray:
        """Compute the batched loglikelihood (return a LL for each batch)"""
        ll_b, _ = BatchedARIMAModel.run_kalman(model, gpu)
        return ll_b

    @staticmethod
    def predict_in_sample(model, gpu=True):
        """Return in-sample prediction on batched series given batched model"""
        _, vs = BatchedARIMAModel.run_kalman(model, gpu)

        assert_same_d(model.order) # We currently assume the same d for all series
        _, d, _ = model.order[0]

        if d == 0:
            y_p = model.y - vs
        elif d == 1:
            y_diff = np.diff(model.y, axis=0)
            y_p = model.y[0:-1, :] + (y_diff - vs)
        else:
            # d>1
            raise NotImplementedError("Only support d==0,1")

        # Extend prediction by 1 when d==1
        if d == 1:
            y_f = BatchedARIMAModel.forecast(model, 1, y_p)
            y_p1 = np.zeros((y_p.shape[0]+1, y_p.shape[1]))
            y_p1[:-1, :] = y_p
            y_p1[-1, :] = y_f
            y_p = y_p1
            

        model.yp = y_p
        return y_p

    @staticmethod
    def forecast(model, nsteps: int, prev_values=None) -> np.ndarray:
        """Forecast the given model `nsteps` into the future."""
        y_fc_b = np.zeros((nsteps, model.num_batches))
        for i in range(model.num_batches):
            if prev_values is not None:
                y_fc_b[:, i] = forecast_values(model.order[i], None,
                                               model.ar_params[i], model.mu[i],
                                               nsteps, prev_values[:, i])
            else:
                y_fc_b[:, i] = forecast_values(model.order[i], model.yp[:, i],
                                               model.ar_params[i], model.mu[i],
                                               nsteps, None)

        return y_fc_b


def assert_same_d(b_order):
    """Checks that all values of d in batched order are same"""
    b_d = [d for _, d, _ in b_order]
    assert (np.array(b_d) == b_d[0]).all()


def init_batched_kalman_matrices(b_ar_params, b_ma_params):
    """Builds batched-versions of the kalman matrices given batched AR and MA parameters"""

    Zb = []
    Rb = []
    Tb = []

    # find maximum 'r' across batches; see (3.18) in TSA by D&K for definition of 'r'
    r_max = np.max([max(len(ar), len(ma)+1) for (ar, ma) in zip(b_ar_params, b_ma_params)])

    for (ari, mai) in zip(b_ar_params, b_ma_params):
        Z, R, T, r = init_kalman_matrices(ari, mai, r_max)
        Zb.append(Z)
        Rb.append(R)
        Tb.append(T)

    return Zb, Rb, Tb, r_max


def grid_search(y_b: np.ndarray, d=1, max_p=3, max_q=3, method="aic"):
    """Grid search to find optimal (lowest `ic`) (p,_,q) values for each
    time-series in y_b, which is a dense `ndarray` with columns as time.
    Optimality is based on minimizing AIC or BIC, which both sum negative
    log-likelihood against model complexity; Higher model complexity might
    yield a lower negative LL, but at higher `aic` due to complexity term.

    """

    num_batches = y_b.shape[1]
    best_ic = np.full(num_batches, np.finfo(np.float64).max/2)
    best_model = BatchedARIMAModel([[]]*num_batches, np.zeros(num_batches), [[]]*num_batches, [[]]*num_batches, y_b)
    # best_model =

    for p in range(0, max_p):
        arparams = np.zeros(p)
        for q in range(0, max_q):
            maparams = np.zeros(q)

            # skip 0,0 case
            if p == 0 and q == 0:
                continue

            b_model = BatchedARIMAModel.fit(y_b, (p, d, q), 0.0, arparams, maparams)

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
