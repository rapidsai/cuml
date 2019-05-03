import numpy as np

from typing import List, Tuple
from .arima import ARIMAModel, loglike, predict_in_sample
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
        with the given initial parameters.

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
                   gpu=True) -> Tuple[np.ndarray, np.ndarray]:
        """Run the (batched) kalman filter for the given model (and contained batched series)"""
        b_ar_params = model.ar_params
        b_ma_params = model.ma_params
        Zb, Rb, Tb, r = init_batched_kalman_matrices(b_ar_params, b_ma_params)

        _, d, _ = model.order[0]
        if d == 0:
            ll_b, vs = batched_kfilter(np.asfortranarray(model.y), # numpy
                                       # y_diff_centered.values, # pandas
                                       Zb, Rb, Tb,
                                       # Z_dense, R_dense, T_dense,
                                       r,
                                       gpu)
        elif d == 1:
            
            y_diff_centered = BatchedARIMAModel.diffAndCenter(model.y, model.num_batches,
                                                              model.mu, model.ar_params)

            ll_b, vs = batched_kfilter(y_diff_centered, # numpy
                                       # y_diff_centered.values, # pandas
                                       Zb, Rb, Tb,
                                       # Z_dense, R_dense, T_dense,
                                       r,
                                       gpu)
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
        if model.d == 0:
            return model.y - vs
        elif model.d == 1:
            # TODO: Extend prediction by 1
            y_diff = np.diff(model.y, axis=0)
            return model.y[0:-1, :] + (y_diff - vs)
        else:
            # d>1
            raise NotImplementedError("Only support d==0,1")


    @staticmethod
    def forecast(model, nsteps: int, prev_values=None):
        """Forcast the given model `nsteps` into the future."""
        raise NotImplementedError("WIP")


def init_batched_kalman_matrices(b_ar_params, b_ma_params):
    """Builds batched-versions of the kalman matrices given batched AR and MA parameters"""

    Zb = []
    Rb = []
    Tb = []
    for (ari, mai) in zip(b_ar_params, b_ma_params):
        Z, R, T, r = init_kalman_matrices(ari, mai)
        Zb.append(Z)
        Rb.append(R)
        Tb.append(T)

    return Zb, Rb, Tb, r


