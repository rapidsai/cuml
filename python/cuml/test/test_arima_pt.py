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

sm_model = sm.ARIMA(y, (1, 1, 1))
sm_model_fit = sm_model.fit()
# sm_model.method = "css-mle"
# sm_model.k_trend = 1
# sm_model.nobs = ns
params = np.array([-2.43238151e+02, 5.50347570e-03, -9.99818447e-01])
stats_model_values_model = arima.ARIMAModel((1, 1, 1),
                                            params[0],
                                            np.array([params[1]]),
                                            np.array([params[2]]),
                                            y)

ll_sm = sm_model.loglike(params)
print("sm ll: ", ll_sm)
ll_a = arima.loglike(stats_model_values_model)
print("am ll: ", ll_a)

# # set_trace()
# sm_model_fit = sm_model.fit(start_params=[-2.43238151e+02, 5.50347570e-03, -9.99818447e-01])
# sm_model_fit = sm_model.fit()



batched_model = batched_arima.BatchedARIMAModel.fit(yb, (1, 1, 1),
                                                    -243.0,
                                                    np.array([-0.005]),
                                                    np.array([-1.0]))


sm_model = sm.ARIMA(y, (1, 1, 1))
# set_trace()
# sm_model_fit = sm_model.fit(start_params=[-2.43238151e+02, 5.50347570e-03, -9.99818447e-01])
sm_model_fit = sm_model.fit()
# stats models yields
# params: array([-2.43238151e+02,  5.50347570e-03, -9.99818447e-01])
# llf: -425.6396771866815
# out model currently yields:
# mu: -276.86347549], ar:-0.09367334, ma:-1.17024532

y_sm_p = sm_model_fit.predict(start=1, end=ns)

y_b = batched_arima.BatchedARIMAModel.predict_in_sample(batched_model)
# -220.35376519   -0.26170006   -2.18930038
arima_model = arima.fit(y, (1, 1, 1), -220.35376519,
                        np.array([-0.26170006]), np.array([-2.18930038]))

stats_model_values_model = arima.ARIMAModel((1,1,1),
                                            sm_model_fit.params[0],
                                            # -243,
                                            np.array([sm_model_fit.params[1]]),
                                            # np.array([1.10070625e-02]),
                                            np.array([sm_model_fit.params[2]]),
                                            # np.array([-1.17]),
                                            y)

print("their values our ll: {} vs their llf: {}".format(arima.loglike(stats_model_values_model)/ns,
                                                        sm_model_fit.llf/ns))
print("our values our ll: {}".format(arima.loglike(arima_model)/ns))
print("our arima: {}, their values: {}".format(arima_model, sm_model_fit.params))

y_our_model_their_values = arima.predict_in_sample(stats_model_values_model)
y_our_model_our_values = arima.predict_in_sample(arima_model)

plt.clf()
plt.plot(w[1], s[1], w[1][:-1], y_b[:,0])
plt.plot(w[1][:-1], y_our_model_their_values, "r--")
plt.plot(w[1][:-1], y_our_model_our_values, "b--")
plt.plot(w[1][:-1], s[1][:-1]+y_sm_p,"g")
plt.show()
