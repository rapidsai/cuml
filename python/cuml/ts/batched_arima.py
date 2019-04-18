import numpy as np

from typing import List, Tuple
from .arima import ARIMAModel, loglike, predict_in_sample
from .kalman import init_kalman_matrices
from .batched_kalman import batched_kfilter, cudf_kfilter
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
    def fit(y: pd.DataFrame,
            order: Tuple[int, int, int],
            mu0: float,
            ar_params0: np.ndarray,
            ma_params0: np.ndarray):
        """
        Fits the ARIMA model to each time-series (batched together in a cuDF
        Dataframe) with the given initial parameters.

        """
        num_series = len(y.shape[0])
        p, d, q = order
        num_parameters = 1 + p + q

        num_batches = y.shape[1]

        def bf(x):
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

            # note: we, maximize the log likelihood, or conversely, minimize
            # the negative log likelihood
            return -ll_b.sum()

        x0 = np.r_[mu0, ar_params0, ma_params0]
        x0 = np.tile(x0, num_series)

        x, f_final, res = opt.fmin_l_bfgs_b(bf, x0, approx_grad=True)

    @staticmethod
    def loglike(model) -> np.ndarray:
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

        ll_b = cudf_kfilter(y_diff_centered, # numpy
                            # y_diff_centered.values, # pandas
                            Zb, Rb, Tb,
                            # Z_dense, R_dense, T_dense,
                            r)

        return ll_b
        

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

    ll, s2 = batched_kfilter(y_c, Zb, Rb, Tb, r, gpu)
    return ll
