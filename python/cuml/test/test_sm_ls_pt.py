import numpy as np
import statsmodels.tsa.arima_model as sm
import pandas as pd

def paperTowels(plot=False):
    data = pd.read_csv("/home/max/dev/arima-experiments/data/data_paper_towel.csv",
                       names=["store", "week", "sold"])

    w = data.groupby("store")["week"].apply(np.array)
    s = data.groupby("store")["sold"].apply(np.array)
    np.set_printoptions(precision=16)

    
    print("Total number of series:", len(w))
    # nb = len(w)
    nb = 50
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

    ym_fit = []
    sm_fail = []

    # for i in range(nb):
    for i in [4]:
        y = yb[:, i]
        sm_model = sm.ARIMA(y, (1, 1, 1))
        sm_model_fit = sm_model.fit(disp=-101, epsilon=1e-9)
        if np.isinf(sm_model_fit.params).any() or np.isnan(sm_model_fit.params).any():
            sm_fail.append(i)
        ym_fit.append(sm_model_fit)
        # y_sm_p = sm_model_fit.predict(start=1, end=ns)
        # print("vals: ", sm_model_fit.mle_retvals)
        # y_sm_p_all[:, i] = y_sm_p
        # print("i={}/{}".format(i, nb))


if __name__ == "__main__":
    paperTowels(False)
