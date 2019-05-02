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
            fdph = np.tile(fd, num_batches)
            fd[i] = 0.0

            ll_b_ph = BatchedARIMAModel.ll_f(num_batches, num_parameters, order, y, x+fdph)
            ll_b_mh = BatchedARIMAModel.ll_f(num_batches, num_parameters, order, y, x-fdph)

            grad_i_b = (ll_b_ph - ll_b_mh)/(2*h)
            
            if num_batches == 1:
                grad[i] = grad_i_b
            else:
                assert(len(grad[i::num_parameters]) == len(grad_i_b))
                grad[i::num_parameters] = grad_i_b

        return grad


    @staticmethod
    def fit(y: pd.DataFrame,
            order: Tuple[int, int, int],
            mu0: float,
            ar_params0: np.ndarray,
            ma_params0: np.ndarray):
        """
        Fits the ARIMA model to each time-series (batched together in a cuDF
        Dataframe) with the given initial parameters.

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
            # Recall: We maximize LL by minimizing negative LL
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

        return BatchedARIMAModel(num_batches*[order], mu, ar, ma, y)


    @staticmethod
    def loglike(model, gpu=True) -> np.ndarray:
        b_ar_params = model.ar_params
        b_ma_params = model.ma_params
        Zb, Rb, Tb, r = init_batched_kalman_matrices(b_ar_params, b_ma_params)

        # TODO: Only do this if d==1
        # TODO: Try to make the following pipeline work in cuDF
        # pandas
        # y_diff = model.y.diff().dropna()
        # numpy
        y_diff = np.diff(model.y, axis=0)

        B0 = np.zeros(model.num_batches)
        for (i, (mu, ar)) in enumerate(zip(model.mu, model.ar_params)):
            B0[i] = mu/(1-np.sum(ar))

        y_diff_centered = np.asfortranarray(y_diff-B0)
        
        # convert the list of kalman matrices into a dense numpy matrix for quick transfer to GPU
        # Z_dense = np.zeros((r * model.num_batches))
        # R_dense = np.zeros((r * model.num_batches))
        # T_dense = np.zeros((r*r * model.num_batches))

        # for (i, (Zi, Ri, Ti)) in enumerate(zip(Zb, Rb, Tb)):
        #     Z_dense[i*r:(i+1)*r] = np.reshape(Zi, r, order="F")
        #     R_dense[i*r:(i+1)*r] = np.reshape(Ri, r, order="F")
        #     T_dense[i*r*r:(i+1)*r*r] = np.reshape(Ti, r*r, order="F")

        ll_b = batched_kfilter(y_diff_centered, # numpy
                               # y_diff_centered.values, # pandas
                               Zb, Rb, Tb,
                               # Z_dense, R_dense, T_dense,
                               r,
                               gpu)

        return ll_b
        

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


