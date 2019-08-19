import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cuml.ts.arima as batched_arima
import statsmodels.tsa.arima_model as sm

import cuml.ts.arima as arima

from IPython.core.debugger import set_trace

from timeit import default_timer as timer

import cProfile
import pstats
from pstats import SortKey


def paperTowelNoBatch(plot=False):
    data = pd.read_csv("/home/max/dev/arima-experiments/data/data_paper_towel.csv",
                       names=["store", "week", "sold"])

    w = data.groupby("store")["week"].apply(np.array)
    s = data.groupby("store")["sold"].apply(np.array)
    np.set_printoptions(precision=6)

    ns = 44
    nb = 50
    yb = np.zeros((ns, 1), order="F")

    # for (i, si) in enumerate(s[:nb]):
    for (i, si) in enumerate(s[160:161]):
        yb[:, 0] = si[0:ns]
        x0 = batched_arima.init_x0((1,1,1), yb[:,0])
        mu0, ar0, ma0 = batched_arima.unpack(1, 1, 1, x0)
        batched_model = batched_arima.fit(yb, (1, 1, 1),
                                          mu0,
                                          ar0,
                                          ma0,
                                          opt_disp=101, gpu=True)
        print("batch results: ", batched_model.mu, batched_model.ar_params, batched_model.ma_params)
        y = yb[:, 0]
        sm_model = sm.ARIMA(y, (1, 1, 1))
        sm_model_fit = sm_model.fit(disp=101)
        print("sm results: ", sm_model_fit.params)


def test_paperTowel_param_init():

    data = pd.read_csv("/home/max/dev/arima-experiments/data/data_paper_towel.csv",
                       names=["store", "week", "sold"])

    w = data.groupby("store")["week"].apply(np.array)
    s = data.groupby("store")["sold"].apply(np.array)
    np.set_printoptions(precision=6)

    ns = 44
    nb = 5
    yb = np.zeros((ns, nb), order="F")

    num_parameters = 3
    x0 = np.zeros(nb*num_parameters)
    np.seterr(all='warn')
    for (i, si) in enumerate(s[:nb]):
        yb[:, i] = si[0:ns]
        # np.seterr(all='raise')
        x0i = batched_arima.init_x0((1,1,1), yb[:,i])
        # np.seterr(all='warn')
        x0[i*num_parameters:(i+1)*num_parameters] = x0i



def paperTowelBatch(plot=False):
    data = pd.read_csv("/home/max/dev/arima-experiments/data/data_paper_towel.csv",
                       names=["store", "week", "sold"])

    w = data.groupby("store")["week"].apply(np.array)
    s = data.groupby("store")["sold"].apply(np.array)
    np.set_printoptions(precision=6)

    ns = 44
    nb = 10
    yb = np.zeros((ns, nb), order="F")

    start = timer()
    num_parameters = 3
    x0 = np.zeros(nb*num_parameters)
    np.seterr(all='warn')
    for (i, si) in enumerate(s[:nb]):
        yb[:, i] = si[0:ns]
        # np.seterr(all='raise')
        x0i = batched_arima.init_x0((1,1,1), yb[:,i])
        # np.seterr(all='warn')
        x0[i*num_parameters:(i+1)*num_parameters] = x0i

    mu0, ar0, ma0 = batched_arima.unpack(1, 1, nb, x0)
    
    batched_model = batched_arima.fit(yb, (1, 1, 1),
                                      mu0,
                                      ar0,
                                      ma0,
                                      opt_disp=-1, h=1e-9, gpu=True)
    end = timer()

    print("time: {}s, iterations (max/min/avg): ".format(end-start),
          np.max(batched_model.niter), np.min(batched_model.niter),
          np.mean(batched_model.niter))

    # the gpu version achieves these iteration counts
    if nb == 240:
        np.testing.assert_almost_equal(45, np.max(batched_model.niter))
        np.testing.assert_almost_equal(16, np.min(batched_model.niter))
        np.testing.assert_almost_equal(33.416666666666664, np.mean(batched_model.niter))
    else:
        print("batched_model.niter=", batched_model.niter)
        print("batched_model.mu=", batched_model.mu)


def paperTowels(plot=False):
    data = pd.read_csv("/home/max/dev/arima-experiments/data/data_paper_towel.csv",
                       names=["store", "week", "sold"])

    w = data.groupby("store")["week"].apply(np.array)
    s = data.groupby("store")["sold"].apply(np.array)
    np.set_printoptions(precision=16)

    
    print("Total number of series:", len(w))
    # nb = len(w)
    nb = 10
    ns_all = np.zeros(nb, dtype=np.int32)
    for (i, si) in enumerate(s[:nb]):
        ns_all[i-1] = len(si)

    ns = np.min(ns_all)
    ns = 44
    print("shortest data: {}, vs longest: {}".format(ns, np.max(ns_all)))

    yb = np.zeros((ns, nb), order="F")

    # i_to_try = [8]
    # i_to_try = range(nb)

    # set_trace()
    for (i, si) in enumerate(s[:nb]):
        # ii = i_to_try[i]
        yb[:, i] = si[0:ns]
        # yb[:, i] = s[ii][0:ns]

    y_sm_p_all = np.zeros((ns, nb), order="F")
    start = timer()
    ym_fit = []
    sm_fail = []

    for i in range(nb):
    # for i in [29]:

        # 29 fails in `statsmodels`
        if i == 29 or i == 111:
            continue
        y = yb[:, i]
        sm_model = sm.ARIMA(y, (1, 1, 1))
        sm_model_fit = sm_model.fit(disp=-1, epsilon=1e-9)
        if np.isinf(sm_model_fit.params).any() or np.isnan(sm_model_fit.params).any():
            sm_fail.append(i)
        ym_fit.append(sm_model_fit)
        y_sm_p = sm_model_fit.predict(start=1, end=ns)
        # print("vals: ", sm_model_fit.mle_retvals)
        y_sm_p_all[:, i] = y_sm_p
        print("i={}/{}".format(i, nb))
    end = timer()
    print("'Statsmodels' Time {} batches = {}s ({}s / batch / timestep)".format(nb, end-start, (end-start)/nb/ns))

    print("Statsmodels failed on: ", sm_fail)

    # best_model, ic = batched_arima.grid_search(y, d=1)

    # yb2 = np.concatenate((yb, yb), 1)
    # yb3 = np.concatenate((yb2, yb2), 1)
    # yb4 = np.concatenate((yb3, yb3), 1)
    yb4 = yb
    # x0 = np.array([-208.35376519, -0.20170006, -0.98930038])
    
    # x0 = np.array([0, -0.25, -0.2])
    # print("i=", i)
    #i=1 my x0= [-3.83534884e+02 -9.18237006e-02 -6.63006374e-01]
    start = timer()

    x0 = np.array([])
    for i in range(nb):
        x0i = batched_arima.init_x0((1,1,1), yb[:,i])
        x0 = np.r_[x0, x0i]
    # x0[0] = x0[0]/2
    # print("my x0=", x0)
    # x0 =
    # x0 = np.array([-2.08696500e+02, -2.00178809e-01, -8.15867515e-01])
    mu0, ar0, ma0 = batched_arima.unpack(1, 1, nb, x0)
    
    batched_model = batched_arima.fit(yb4, (1, 1, 1),
                                      mu0,
                                      ar0,
                                      ma0,
                                      opt_disp=-1, h=1e-9, gpu=True)

    y_b = batched_arima.predict_in_sample(batched_model)
    end = timer()

    print("GPU Time for {} batches = {}s ({}s / batch)".format(yb4.shape[1], (end-start),
                                                               (end-start)/yb4.shape[1]/ns))

    # plt.clf()
    # set_trace()
    mu_vs = [(ymi.params[0], mui) for (ymi, mui) in zip(ym_fit, batched_model.mu)]
    ar_vs = [(ymi.params[1], ari) for (ymi, ari) in zip(ym_fit, batched_model.ar_params)]
    ma_vs = [(ymi.params[2], mai) for (ymi, mai) in zip(ym_fit, batched_model.ma_params)]
    print("mu_vs=", mu_vs)
    print("ar_vs=", ar_vs)
    print("ma_vs=", ma_vs)
    print("batched_model.niter=", batched_model.niter)

    # x0 = np.array([-220.35376519, -0.26170006, -2.18930038], dtype=np.float64)
    # x0 = np.array([-220.35376518754148, -0.26170006272244173, -2.1893003751753457], dtype=np.float64)
    # y = yb[:, 0]
    # sm_model = sm.ARIMA(y, (1, 1, 1))
    # sm_model_fit = sm_model.fit(disp=-1)
    # # sm_model.params = x0
    # # set_trace()                                                    
    # lls = -sm_model.loglike_kalman(x0)/(len(y)-1)
    # # glls = sm_model.score(x0)
    # print("lls=", lls)
    # # print("glls=", glls)
    # b_model_x0 = batched_arima.BatchedARIMAModel([(1,1,1)], np.array([x0[0]]),
    #                                              [np.array([x0[1]])], [np.array([x0[2]])], yb)
    # llf = -batched_arima.BatchedARIMAModel.ll_f(1, 3, (1,1,1), yb, x0)/(len(y)-1)
    # gllf = batched_arima.BatchedARIMAModel.ll_gf(1, 3, (1,1,1), yb, x0, h=1e-7)
    # print("llf=", llf)
    # print("gllf=", gllf)
    # set_trace()

    sqn_errors = np.zeros(nb)
    for i in range(nb):
        # statsmodels struggles with series 29
        if i == 29 or i == 111:
            continue
        y_sm = s[s.keys()[i]][:ns] + y_sm_p_all[:, i]
        y_i = y_b[:, i]
        sqn_errors[i] = np.sum((y_sm - y_i)**2/np.abs(y_sm))

    largest_error_indices = np.argsort(sqn_errors)

    if plot:

        nb_plot = 5
        fig, axes = plt.subplots(nb_plot, 1)
        # axes[0].plot(w[1][:-1], s[1][:-1] + y_sm_p_all[:, 0], "b-")
        # axes[0].plot(w[1], s[1], "k", w[1][:ns], y_b[:, 0], "r--")
        for i in range(nb_plot):
            if i >= nb:
                break
            idx = largest_error_indices[-(i+1)]
            idxp = s.keys()[idx]
            axes[i].plot(w[idxp][:ns], s[idxp][:ns] + y_sm_p_all[:, idx], "b-")
            axes[i].plot(w[idxp], s[idxp], "k-", w[idxp][:ns], y_b[:, idx], "r--")
            axes[i].set_title("series {}".format(idx))

        plt.show()

def run_profile():
    cProfile.run("paperTowels()", "profile.dat")

def analyze_profile():
    p = pstats.Stats("profile.dat")
    p.strip_dirs().sort_stats(SortKey.TIME).print_stats(5)
    p.sort_stats(SortKey.TIME).print_stats(20)

if __name__ == "__main__":
    paperTowels(True)
    # paperTowelBatch(False)
