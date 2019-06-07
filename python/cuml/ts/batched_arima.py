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

from .batched_bfgs import batched_fmin_bfgs

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
    def ll_gf(num_batches, num_parameters, order, y, x, h=1e-8, gpu=True, trans=False):
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

            ll_b_ph = BatchedARIMAModel.ll_f(num_batches, num_parameters, order, y, x+fdph, gpu=gpu, trans=trans)
            ll_b_mh = BatchedARIMAModel.ll_f(num_batches, num_parameters, order, y, x-fdph, gpu=gpu, trans=trans)
            # ll_b0 = BatchedARIMAModel.ll_f(num_batches, num_parameters, order, y, x, gpu=gpu, trans=trans)

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


    @staticmethod
    def fit(y: np.ndarray,
            order: Tuple[int, int, int],
            mu0: np.ndarray,
            ar_params0: [np.ndarray],
            ma_params0: [np.ndarray],
            opt_disp=-1, h=1e-8, gpu=True, alpha_max=1000):
        """
        Fits the ARIMA model to each time-series (batched together in a dense numpy matrix)
        with the given initial parameters. `y` is (num_samples, num_batches)

        """

        p, d, q = order
        num_parameters = 1 + p + q

        num_samples = y.shape[0]  # pandas Dataframe shape is (num_batches, num_samples)
        num_batches = y.shape[1]

        def f(x, n=None):
            # Maximimize LL means minimize negative
            if n is not None:
                n_llf = -(BatchedARIMAModel.ll_f(num_batches, num_parameters, order, y, x, gpu=gpu, trans=True))
                return n_llf[n]

            n_llf_sum = -(BatchedARIMAModel.ll_f(num_batches, num_parameters, order, y, x, gpu=gpu, trans=True).sum())
            return n_llf_sum/(num_samples-1)/num_batches

        # optimized finite differencing gradient for batches
        def gf(x, n=None):
            # Recall: We maximize LL by minimizing -LL
            n_gllf = -BatchedARIMAModel.ll_gf(num_batches, num_parameters, order, y, x, h, gpu=gpu, trans=True)
            if n is not None:
                n_gllf = -BatchedARIMAModel.ll_gf(num_batches, num_parameters, order, y, x, h, gpu=gpu, trans=True)
                n_gllf_n = np.zeros(len(n_gllf))
                n_gllf_n[n*num_parameters:(n+1)*num_parameters] = n_gllf[n*num_parameters:(n+1)*num_parameters]
                return n_gllf_n

            return n_gllf/(num_samples-1)/num_batches

        if not isinstance(mu0, np.ndarray):
            ar0_2 = sm_tools._ar_invtransparams(np.copy(ar_params0))
            ma0_2 = sm_tools._ma_invtransparams(np.copy(ma_params0))
            
            x0 = np.r_[mu0, ar0_2, ma0_2]
            x0 = np.tile(x0, num_batches)

        else:
            x0 = pack(p, q, num_batches, mu0, ar_params0, ma_params0)
            x0 = batch_invtrans(p, q, num_batches, x0)
            

        # SciPy l-bfgs (non-batched)
        # x, f_final, res = opt.fmin_l_bfgs_b(f, x0, fprime=gf,
        #                                     approx_grad=False,
        #                                     iprint=opt_disp,
        #                                     factr=10000,
        #                                     m=1,
        #                                     pgtol=1e-7,
        #                                     epsilon=h)
        # if res['warnflag'] > 0:
        #     raise ValueError("ERROR: In `fit()`, the optimizer failed to converge with warning: {}. Check the initial conditions (particularly `mu`) or change the finite-difference stepsize `h` (lower or raise..)".format(res))

        # our batch-aware bfgs solver

        if ((np.isnan(x0).any()) or (np.isinf(x0).any())):
            raise FloatingPointError("Initial condition 'x0' has NaN or Inf.")

        x, fk = batched_fmin_bfgs(f, x0, num_batches, gf, alpha_per_batch=False, disp=opt_disp, alpha_max=alpha_max)
        # print("our results: ", x)
        # scipy again
        # options = {"disp": 1}
        # res = opt.minimize(f, x0, jac=gf, method="BFGS", options=options)
        # x = res.x
        # print("their results: ", x)

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
        uses an analytical approximation (Durbin Koopman pg 138).`"""
        b_ar_params = model.ar_params
        b_ma_params = model.ma_params
        Zb, Rb, Tb, r = init_batched_kalman_matrices(b_ar_params, b_ma_params)

        assert_same_d(model.order) # We currently assume the same d for all series
        _, d, _ = model.order[0]

        if d == 0:

            ll_b, vs = batched_kfilter(np.asfortranarray(model.y), # numpy
                                       Zb, Rb, Tb,
                                       r,
                                       gpu, initP_kalman_iterations)
        elif d == 1:

            y_diff_centered = BatchedARIMAModel.diffAndCenter(model.y, model.num_batches,
                                                              model.mu, model.ar_params)


            ll_b, vs = batched_kfilter(y_diff_centered, # numpy
                                       Zb, Rb, Tb,
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

def unpack(p, q, nb, x):
    """Unpack linearized parameters into mu, ar, and ma batched-groupings"""
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
    """Pack mu, ar, and ma batched-groupings into a linearized vector `x`"""
    num_parameters = 1 + p + q
    x = np.zeros(num_parameters*nb)
    for i in range(nb):
        xi = np.array([mu[i]])
        xi = np.r_[xi, ar[i]]
        xi = np.r_[xi, ma[i]]
        x[i*num_parameters:(i+1)*num_parameters] = xi
    return x

def batch_trans(p, q, nb, x):
    """Apply the stationarity/invertibility guaranteeing transform to batched-parameter vector x."""
    mu, ar, ma = unpack(p, q, nb, x)
    ar2 = []
    ma2 = []
    np.seterr(all='raise')

    for ib in range(nb):
        ari = sm_tools._ar_transparams(np.copy(ar[ib]))
        mai = sm_tools._ma_transparams(np.copy(ma[ib]))
        ar2.append(ari)
        ma2.append(mai)

    np.seterr(all='warn')

    Tx = pack(p, q, nb, mu, ar2, ma2)
    return Tx

def batch_invtrans(p, q, nb, x):
    """Apply the *inverse* stationarity/invertibility guaranteeing transform to
       batched-parameter vector x.
    """
    mu, ar, ma = unpack(p, q, nb, x)
    ar2 = []
    ma2 = []
    for ib in range(nb):
        print("invT: {},{}".format(ar[ib], ma[ib]))
        ari = sm_tools._ar_invtransparams(np.copy(ar[ib]))
        mai = sm_tools._ma_invtransparams(np.copy(ma[ib]))
        assert ((not np.isinf(ari)) and (not np.isinf(mai)))
        assert ((not np.isnan(ari)) and (not np.isnan(mai)))
        ar2.append(ari)
        ma2.append(mai)
    Tx = pack(p, q, nb, mu, ar2, ma2)
    return Tx

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
    arma.transparams = True
    arma.k_trend = 1
    arma.nobs = len(yd)
    x0 = arma._fit_start_params((p,q,d), "css-mle")

    mu, ar, ma = unpack(p, q, 1, x0)
    # fix ma to avoid bad values in inverse invertibility transform
    for i in range(len(ma[0])):
        mai = ma[0][i]
        # if ma >= 1, then we get "inf" results from inverse transform
        ma[0][i] = np.sign(mai)*min(np.abs(mai), 1-1e-14)
    x0 = pack(p, q, 1, mu, ar, ma)
    return x0


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


# # copied from internets just in case i need it
# def gradient_descent(x0, f, f_prime, hessian=None, adaptative=True, gtol=1e-6, ftol=1e-12):

#     all_f_i = []
#     xk = x0
#     for i in range(1, 100):

#         all_f_i.append(f(xk))
#         dx_i = f_prime(xk)

#         if np.linalg.norm(xk) < gtol:
#             print("Gradient tolerance reached: ", gtol)
#             break
#         if i > 1 and np.abs(all_f_i[-1] - all_f_i[-2]) < ftol:
#             print("tolerance for change in function values reached: ", ftol)
#             break

#         if adaptative:
#             # Compute a step size using a line_search to satisfy the Wolf
#             # conditions
#             step = opt.line_search(f, f_prime,
#                                    xk, -dx_i,
#                                    dx_i)
#             step = step[0]
#             if step is None:
#                 step = 0
#         else:
#             step = 1
#         xk = xk - step*dx_i


#     return xk, all_f_i
