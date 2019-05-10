import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cuml.ts.batched_arima as batched_arima
import statsmodels.tsa.arima_model as sm

import cuml.ts.arima as arima

from IPython.core.debugger import set_trace

data = pd.read_csv("/home/max/dev/arima-experiments/data/data_paper_towel.csv",
                   names=["store", "week", "sold"])

w = data.groupby("store")["week"].apply(np.array)
s = data.groupby("store")["sold"].apply(np.array)
plt.plot(w[1], s[1])

nb = 1
ns = len(w[1])-1

yb = np.zeros((ns, nb), order="F")

for (i, si) in enumerate(s[:nb]):
    yb[:, i] = si[0:ns]

y = yb[:, 0]
# best_model, ic = batched_arima.grid_search(y, d=1)

batched_model = batched_arima.BatchedARIMAModel.fit(yb, (1, 1, 1),
                                                    -20.0,
                                                    np.array([-0.005]),
                                                    np.array([-1.0]),
                                                    opt_disp=5)

sm_model = sm.ARIMA(y, (1, 1, 1))
sm_model_fit = sm_model.fit(disp=5)

y_sm_p = sm_model_fit.predict(start=1, end=ns)

y_b = batched_arima.BatchedARIMAModel.predict_in_sample(batched_model)

plt.clf()
plt.plot(w[1], s[1], "k", w[1][:-1], y_b[:, 0], "r--")
plt.plot(w[1][:-1], s[1][:-1] + y_sm_p, "b-")
plt.show()
