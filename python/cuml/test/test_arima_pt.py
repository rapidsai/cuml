import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cuml.ts.batched_arima as batched_arima
import statsmodels.tsa.arima_model as sm

import cuml.ts.arima as arima

from IPython.core.debugger import set_trace

from timeit import default_timer as timer

data = pd.read_csv("/home/max/dev/arima-experiments/data/data_paper_towel.csv",
                   names=["store", "week", "sold"])

w = data.groupby("store")["week"].apply(np.array)
s = data.groupby("store")["sold"].apply(np.array)

nb = len(w)
ns_all = np.zeros(nb, dtype=np.int32)
for (i, si) in enumerate(s[:nb]):
    ns_all[i-1] = len(si)

ns = np.min(ns_all)
print("shortest data: {}, vs longest: {}".format(ns, np.max(ns_all)))

yb = np.zeros((ns, nb), order="F")

for (i, si) in enumerate(s[:nb]):
    yb[:, i] = si[0:ns]

y = yb[:, 0]
start = timer()
sm_model = sm.ARIMA(y, (1, 1, 1))
sm_model_fit = sm_model.fit(disp=-1)
y_sm_p = sm_model_fit.predict(start=1, end=ns)
end = timer()
print("CPU Time 1 batch = {}s, estimate for {} batches = {}s".format(end-start, nb, (end-start)*nb))

# best_model, ic = batched_arima.grid_search(y, d=1)

start = timer()
batched_model = batched_arima.BatchedARIMAModel.fit(yb, (1, 1, 1),
                                                    -200.0,
                                                    np.array([-0.005]),
                                                    np.array([-1.0]),
                                                    opt_disp=-1)

y_b = batched_arima.BatchedARIMAModel.predict_in_sample(batched_model)
end = timer()

print("GPU Time for all batches ({}) = {}s".format(nb, (end-start)))

# plt.clf()

plot = False
if plot:

    fig, axes = plt.subplots(nb, 1)
    axes[0].plot(w[1][:-1], s[1][:-1] + y_sm_p, "b-")
    axes[0].plot(w[1], s[1], "k", w[1][:ns], y_b[:, 0], "r--")
    for (i, (wi, si)) in enumerate(zip(w, s)):
        if i >= nb:
            break
        axes[i].plot(wi, si, "k", wi[:ns], y_b[:, i], "r--")

    plt.show()
