import numpy as np

from typing import List, Tuple
from .arima import ARIMAModel, loglike, predict_in_sample, forecast_values
from .kalman import init_kalman_matrices
from .batched_kalman import batched_kfilter, pynvtx_range_push, pynvtx_range_pop
import scipy.optimize as opt
from IPython.core.debugger import set_trace
import statsmodels.tsa.tsatools as sm_tools
import statsmodels.tsa.arima_model as sm_arima
# import cudf
import pandas as pd

# import torch.cuda.nvtx as nvtx

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
        self.yp = None

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
    def ll_f(num_batches, num_parameters, order, y, x, return_negative_sum=False, gpu=True):
        """Computes batched loglikelihood given parameters stored in `x`."""
        pynvtx_range_push("ll_f")
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

        ll_b = BatchedARIMAModel.loglike(b_model, gpu)
        pynvtx_range_pop()
        if return_negative_sum:
            return -ll_b.sum()
        else:
            return ll_b

    #TODO: Fix this for multi-(p,q) case
    @staticmethod
    def ll_gf(num_batches, num_parameters, order, y, x, h=1e-8, gpu=True):
        """Computes fd-gradient of batched loglikelihood given parameters stored in
        `x`. Because batches are independent, it only compute the function for the
        single-batch number of parameters."""
        pynvtx_range_push("ll_gf")
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

        assert (len(x) / num_parameters) == float(num_batches)
        for i in range(num_parameters):
            fd[i] = h

            # duplicate the perturbation across batches (they are independent)
            fdph = np.tile(fd, num_batches)

            # reset perturbation
            fd[i] = 0.0

            ll_b_ph = BatchedARIMAModel.ll_f(num_batches, num_parameters, order, y, x+fdph, gpu=gpu)
            ll_b_mh = BatchedARIMAModel.ll_f(num_batches, num_parameters, order, y, x-fdph, gpu=gpu)

            grad_i_b = (ll_b_ph - ll_b_mh)/(2*h)

            if num_batches == 1:
                grad[i] = grad_i_b
            else:
                assert len(grad[i::num_parameters]) == len(grad_i_b)
                # Distribute the result to all batches
                grad[i::num_parameters] = grad_i_b

        pynvtx_range_pop()        
        return grad


    @staticmethod
    def fit(y: np.ndarray,
            order: Tuple[int, int, int],
            mu0: float,
            ar_params0: np.ndarray,
            ma_params0: np.ndarray,
            opt_disp=-1, h=1e-8, gpu=True):
        """
        Fits the ARIMA model to each time-series (batched together in a dense numpy matrix)
        with the given initial parameters. `y` is (num_samples, num_batches)

        """

        def unpack(p, q, nb, x):
            num_parameters = 1 + p + q
            mu = np.zeros(nb)
            ar = []
            ma = []
            for i in range(nb):
                xi = x[i*num_parameters:(i+1)*num_parameters]
                mu[i] = xi[0]
                ar.append(xi[1:p+1])
                ma.append(xi[p+1:])

            return (mu, ar, ma)

        def pack(p, q, nb, mu, ar, ma):
            num_parameters = 1 + p + q
            x = np.zeros(num_parameters*nb)
            for i in range(nb):
                xi = np.array([mu[i]])
                xi = np.r_[xi, ar[i]]
                xi = np.r_[xi, ma[i]]
                x[i*num_parameters:(i+1)*num_parameters] = xi
            return x

        def batch_trans(p, q, nb, x):
            mu, ar, ma = unpack(p, q, nb, x)
            ar2 = []
            ma2 = []
            for ib in range(nb):
                ari = sm_tools._ar_transparams(np.copy(ar[ib]))
                mai = sm_tools._ma_transparams(np.copy(ma[ib]))
                ar2.append(ari)
                ma2.append(mai)
            Tx = pack(p, q, nb, mu, ar2, ma2)
            return Tx

        def batch_invtrans(p, q, nb, x):
            mu, ar, ma = unpack(p, q, nb, x)
            ar2 = []
            ma2 = []
            for ib in range(nb):
                ari = sm_tools._ar_invtransparams(np.copy(ar[ib]))
                mai = sm_tools._ma_invtransparams(np.copy(ma[ib]))
                ar2.append(ari)
                ma2.append(mai)
            Tx = pack(p, q, nb, mu, ar2, ma2)
            return Tx
        

        p, d, q = order
        num_parameters = 1 + p + q

        num_samples = y.shape[0]  # pandas Dataframe shape is (num_batches, num_samples)
        num_batches = y.shape[1]

        def f(x):
            # Maximimize LL means minimize negative
            x2 = batch_trans(p, q, num_batches, x)
            n_llf_sum = -(BatchedARIMAModel.ll_f(num_batches, num_parameters, order, y, x2, gpu=gpu).sum())
            return n_llf_sum/(num_samples-1)/num_batches

        # optimized finite differencing gradient for batches
        def gf(x):
            # Recall: We maximize LL by minimizing -LL
            x2 = batch_trans(p, q, num_batches, x)
            n_gllf = -BatchedARIMAModel.ll_gf(num_batches, num_parameters, order, y, x2, h, gpu=gpu)
            return n_gllf/(num_samples-1)/num_batches


        ar0_2 = sm_tools._ar_invtransparams(np.copy(ar_params0))
        ma0_2 = sm_tools._ma_invtransparams(np.copy(ma_params0))

        x0 = np.r_[mu0, ar0_2, ma0_2]
        x0 = np.tile(x0, num_batches)

        x, f_final, res = opt.fmin_l_bfgs_b(f, x0, fprime=None,
                                            approx_grad=True,
                                            iprint=opt_disp,
                                            factr=100,
                                            m=12,
                                            pgtol=1e-8)

        print("xf=", x)
        if res['warnflag'] > 0:
            raise ValueError("ERROR: In `fit()`, the optimizer failed to converge with warning: {}. Check the initial conditions (particularly `mu`) or change the finite-difference stepsize `h` (lower or raise..)".format(res))

        Tx = batch_trans(p, q, num_batches, x)
        mu, ar, ma = unpack(p, q, num_batches, Tx)
        
        fit_model = BatchedARIMAModel(num_batches*[order], mu, ar, ma, y)
        
        return fit_model


    @staticmethod
    def diffAndCenter(y: np.ndarray, num_batches: int,
                      mu: np.ndarray, ar_params: np.ndarray):
        """Diff and center batched series `y`"""
        y_diff = np.diff(y, axis=0)

        B0 = np.zeros(num_batches)
        for (i, (mu, ar)) in enumerate(zip(mu, ar_params)):
            # B0[i] = mu/(1-np.sum(ar))
            B0[i] = mu

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
            P0 = []
            if not initP_kalman_iterations:
                for i in range(model.num_batches):
                    Z_bi = Zb[i]
                    R_bi = Rb[i]
                    T_bi = Tb[i]

                    invImTT = np.linalg.pinv(np.eye(r**2) - np.kron(T_bi, T_bi))
                    _P0 = np.reshape(invImTT @ (R_bi @ R_bi.T).ravel(), (r, r), order="F")
                    P0.append(_P0)
                    # print("P0[{}]={}".format(i, P0))

            ll_b, vs = batched_kfilter(np.asfortranarray(model.y), # numpy
                                       Zb, Rb, Tb,
                                       P0,
                                       r,
                                       gpu, initP_kalman_iterations)
        elif d == 1:
            
            y_diff_centered = BatchedARIMAModel.diffAndCenter(model.y, model.num_batches,
                                                              model.mu, model.ar_params)

            
            P0 = []
            if not initP_kalman_iterations:
                pynvtx_range_push("compute P0")
                for i in range(model.num_batches):
                    Z_bi = Zb[i]
                    R_bi = Rb[i]
                    T_bi = Tb[i]

                    invImTT = np.linalg.pinv(np.eye(r**2) - np.kron(T_bi, T_bi))
                    _P0 = np.reshape(invImTT @ (R_bi @ R_bi.T).ravel(), (r, r), order="F")
                    P0.append(_P0)
                    # print("P0[{}]={}".format(i, P0))

                pynvtx_range_pop()
            ll_b, vs = batched_kfilter(y_diff_centered, # numpy
                                       Zb, Rb, Tb,
                                       P0,
                                       r,
                                       gpu, initP_kalman_iterations)
        else:
            raise NotImplementedError("ARIMA only support d==0,1")

        return ll_b, vs

    @staticmethod
    def loglike(model, gpu=True) -> np.ndarray:
        """Compute the batched loglikelihood (return a LL for each batch)"""
        ll_b, _ = BatchedARIMAModel.run_kalman(model, gpu=gpu)
        return ll_b

    @staticmethod
    def predict_in_sample(model, gpu=True):
        """Return in-sample prediction on batched series given batched model"""
        _, vs = BatchedARIMAModel.run_kalman(model, gpu=gpu)

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
                fc1[i] = BatchedARIMAModel.fc_single(1, model.order[i], y_diff[:,i],
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


    @staticmethod
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
        
    @staticmethod
    def forecast(model, nsteps: int) -> np.ndarray:
        """Forecast the given model `nsteps` into the future."""
        y_fc_b = np.zeros((nsteps, model.num_batches))

        _, vs = BatchedARIMAModel.run_kalman(model)

        for i in range(model.num_batches):
            p, d, q = model.order[i]
            vsi = vs[:,i]
            ydiff_i = np.diff(model.y[:, i],axis=0)
            fc = BatchedARIMAModel.fc_single(nsteps, (p,d,q), ydiff_i, vsi,
                                             model.mu[i], model.ma_params[i],
                                             model.ar_params[i])

            if model.order[i][1] > 0: # d > 0
                fc = undifference(fc, model.y[-1,i])[1:]

            y_fc_b[:, i] = fc[:]

        return y_fc_b

def undifference(x, x0):
    # set_trace()
    xi = np.append(x0, x)
    return np.cumsum(xi)

def assert_same_d(b_order):
    """Checks that all values of d in batched order are same"""
    b_d = [d for _, d, _ in b_order]
    assert (np.array(b_d) == b_d[0]).all()

def init_x0(order, y):
    (p, d, q) = order
    if d == 1:
        yd = np.diff(y)
    else:
        yd = np.copy(y)
    arma = sm_arima.ARMA(yd, order)
    arma.exog = np.ones((len(yd),1))
    return arma._fit_start_params_hr((p,q,d))


def init_batched_kalman_matrices(b_ar_params, b_ma_params):
    """Builds batched-versions of the kalman matrices given batched AR and MA parameters"""
    pynvtx_range_push("init_batched_kalman_matrices")

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

    pynvtx_range_pop()
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
