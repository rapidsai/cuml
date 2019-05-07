import numpy as np

import statsmodels.tsa.stattools as st
import pmdarima.arima as am

from sklearn.linear_model import LinearRegression

from IPython.core.debugger import set_trace

def is_stationary(yi: np.ndarray, pval_threshold=0.05) -> bool:
    """Test if `yi` is stationary around a constant. Based on paper 'Testing the
    null hypothesis of stationary against the alternative of a unit root' by
    Kwiatkowski et al. 1992. See
    https://www.statsmodels.org/dev/_modules/statsmodels/tsa/stattools.html#kpss
    for additional details."""

    ns = len(yi)

    # Null hypothesis: data is stationary around a constant
    e = yi - yi.mean()

    # Table 1, Kwiatkowski 1992
    crit_vals = [0.347, 0.463, 0.574, 0.739]
    pvals = [0.10, 0.05, 0.025, 0.01]

    s = np.cumsum(e)
    # eq. 11
    eta = s.dot(s) / ns**2

    #########################
    # compute eq. 10

    # from Kwiatkowski et al. referencing Schwert (1989)
    lags = int(np.ceil(12. * np.power(ns / 100., 1 / 4.)))

    s2_A = e.dot(e)/ns

    def w(_s, _l):
        return 1-_s/(_l+1)

    s2_B = 0.0
    for j in range(1, lags+1):
        eprod = np.dot(e[j:], e[:ns - j])
        s2_B += 2/ns * w(j, lags) * eprod

    s2 = s2_A + s2_B
    #########################



    kpss_stat = eta / s2
    p_value = np.interp(kpss_stat, crit_vals, pvals)

    # print("kpss_stat: {}, pvalue:{}".format(kpss_stat, p_value))

    # higher p_value means higher chance data is stationary around constant.
    return p_value > pval_threshold


def stationarity(y: np.ndarray, pval_threshold=0.05) -> np.ndarray:
    """Return recommended differencing parameter `d=0 or 1` for batched series y"""

    ns = y.shape[0]
    nb = y.shape[1]
    d = np.zeros(nb, dtype=np.int32)
    for i in range(nb):

        yi = y[:, i]
        
        if is_stationary(yi):
            d[i] = 0
        elif is_stationary(np.diff(yi)):
            d[i] = 1
        else:
            raise ValueError("Stationarity failed for d=0 or 1.")

    return d
        
