
def my_run_line_magic(*args, **kwargs):
    g=globals()
    l={}
    for a in args:
        try:
            exec(str(a),g,l)
        except Exception as e:
            print('WARNING: %s\n   While executing this magic function code:\n%s\n   continuing...\n' % (e, a))
        else:
            g.update(l)

def my_run_cell_magic(*args, **kwargs):
    my_run_line_magic(*args, **kwargs)

get_ipython().run_line_magic=my_run_line_magic
get_ipython().run_cell_magic=my_run_cell_magic


#!/usr/bin/env python
# coding: utf-8

# # ARIMA
# 
# An [AutoRegressive Integrated Moving Average](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) model is a popular model used in time series analysis to understand the data or forecast future points.
# 
# This implementation can fit a model to each time series in a batch and perform in-sample predictions and out-of-sample forecasts. It is designed to give the best performance when working on a large batch of time series.
# 
# Useful links:
# 
# - cuDF documentation: https://docs.rapids.ai/api/cudf/stable
# - cuML's ARIMA API docs: https://rapidsai.github.io/projects/cuml/en/stable/api.html#arima
# - a good introduction to ARIMA: https://otexts.com/fpp2/arima.html

# ## Setup
# 
# ### Imports

# In[ ]:


import cudf
from cuml.tsa.arima import ARIMA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ### Loading util
# 
# The data for this demo is stored in a simple CSV format:
# - the data series are stored in columns
# - the first column contains the date of the data points
# - the first row contains the name of each variable
# 
# For example, let's check the *population estimate* dataset:

# In[ ]:


get_ipython().system('cat data/time_series/population_estimate.csv | head')


# We define a helper function to load a dataset with a given name and return a GPU dataframe. We discard the date, and limit the batch size for convenience.

# In[ ]:


def load_dataset(name, max_batch=4):
    import os
    pdf = pd.read_csv(os.path.join("data", "time_series", "%s.csv" % name))
    return cudf.from_pandas(pdf[pdf.columns[1:max_batch+1]].astype(np.float64))


# ### Visualization util
# 
# We define a helper function that displays the data, and optionally a prediction starting from a given index. Each time series is plot separately for better readability.

# In[ ]:


def visualize(y, pred=None, pred_start=None):
    n_obs, batch_size = y.shape

    # Create the subplots
    c = min(batch_size, 2)
    r = (batch_size + c - 1) // c
    fig, ax = plt.subplots(r, c, squeeze=False)
    ax = ax.flatten()
    
    # Range for the prediction
    if pred is not None:
        pred_start = pred_start or n_obs
        pred_end = pred_start + pred.shape[0]
    
    # Plot the data
    for i in range(batch_size):
        title = y.columns[i]
        ax[i].plot(np.r_[:n_obs], y[title].to_array())
        if pred is not None:
            ax[i].plot(np.r_[pred_start:pred_end],
                       pred[pred.columns[i]].to_array(),
                       linestyle="--")
        ax[i].title.set_text(title)
    for i in range(batch_size, r*c):
        fig.delaxes(ax[i])
    fig.tight_layout()
    plt.show()


# ## Non-seasonal ARIMA models
# 
# A basic `ARIMA(p,d,q)` model is made of three components:
#  - An **Integrated** (I) component: the series is differenced `d` times until it is stationary
#  - An **AutoRegressive** (AR) component: the variable is regressed on its `p` past values
#  - A **Moving Average** (MA) component: the variable is regressed on `q` past error terms
# 
# The model can also incorporate an optional constant term (called *intercept*).
# 
# ### A simple MA(2) example
# 
# We start with a simple Moving Average model. Let's first load and visualize the *migrations in Auckland by age* dataset:

# In[ ]:


df_mig = load_dataset("net_migrations_auckland_by_age", 4)
visualize(df_mig)


# We want to fit the model with `q`=2 and with an intercept.
# The `ARIMA` class accepts cuDF dataframes or array-like types as input (host or device), e.g numpy arrays. Here we already have a dataframe so we can simply pass it to the `ARIMA` constructor with the model parameters:

# In[ ]:


model_mig = ARIMA(df_mig, (0,0,2), fit_intercept=True)
model_mig.fit()


# We can now forecast and visualize the results:

# In[ ]:


fc_mig = model_mig.forecast(10)
visualize(df_mig, fc_mig)


# If we want to get the parameters that were fitted to the model, we use the `get_params` method:

# In[ ]:


param_mig = model_mig.get_params()
print(param_mig.keys())


# The parameters are organized in 2D arrays: one row represents one parameter and the columns are different batch members.

# In[ ]:


# Print the ma.L1 and ma.L2 parameters for each of 4 batch members
print(param_mig["ma"])


# We can also get the log-likelihood of the parameters w.r.t to the series, and evaluate various information criteria:

# In[ ]:


print("log-likelihood:\n", model_mig.llf)
print("\nAkaike Information Criterion (AIC):\n", model_mig.aic)
print("\nCorrected Akaike Information Criterion (AICc):\n", model_mig.aicc)
print("\nBayesian Information Criterion (BIC):\n", model_mig.bic)


# ### An ARIMA(1,2,1) example
# 
# Let's now load the *population estimate* dataset. For this dataset a first difference is not enough to make the data stationary because of the quadratic trend, so we decide to go with `d`=2.
# 
# This time we won't simply forecast but also predict in-sample:

# In[ ]:


df_pop = load_dataset("population_estimate")

# Fit an ARIMA(1,2,1) model
model_pop = ARIMA(df_pop, (1,2,1), fit_intercept=True)
model_pop.fit()

# Predict in-sample and forecast out-of-sample
fc_pop = model_pop.predict(80, 160)
visualize(df_pop, fc_pop, 80)


# ## Seasonal ARIMA models
# 
# [Seasonal ARIMA models](https://otexts.com/fpp2/seasonal-arima.html) are expressed in the form `ARIMA(p,d,q)(P,D,Q)s` and have additional seasonal components that we denote SAR and SMA.
# 
# We can also choose to apply a first or second seasonal difference, or combine a non-seasonal and a seasonal difference (note: `p+P <= 2` is required).
# 
# ### An ARIMA(1,1,1)(1,1,1)12 example
# 
# We load the *guest nights by region* dataset. This dataset shows a strong seasonal component with a period of 12 (annual cycle, monthly data), and also a non-seasonal trend. A good choice is to go with `d`=1, `D`=1 and `s`=12.
# 
# We create the model with seasonal parameters, and forecast:

# In[ ]:


df_guests = load_dataset("guest_nights_by_region", 4)

# Create and fit an ARIMA(1,1,1)(1,1,1)12 model:
model_guests = ARIMA(df_guests, (1,1,1), (1,1,1,12), fit_intercept=False)
model_guests.fit()


# In[ ]:


# Forecast
fc_guests = model_guests.forecast(40)

# Visualize after the time step 200
visualize(df_guests[200:], fc_guests)

