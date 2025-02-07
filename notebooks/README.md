# cuML Notebooks
## Intro
These notebooks provide examples of how to use cuML.  These notebooks are designed to be self-contained with the `runtime` version of the [RAPIDS Docker Container](https://hub.docker.com/r/rapidsai/rapidsai/) and [RAPIDS Nightly Docker Containers](https://hub.docker.com/r/rapidsai/rapidsai-nightly) and can run on air-gapped systems.  You can quickly get this container using the install guide from the [RAPIDS.ai Getting Started page](https://rapids.ai/start.html#get-rapids)

## Getting started notebooks
For a good overview of how cuML works, see [the introductory notebook
on estimators](../docs/source/estimator_intro.ipynb) in the
documentation tree.

## Additional notebooks
Notebook Title | Status | Description
--- | --- | ---
[ARIMA Demo](arima_demo.ipynb) | Working | Forecast using ARIMA on time-series data.
[Forest Inference Demo](forest_inference_demo.ipynb) | Working | Save and load an XGBoost model into FIL and infer on new data.
[KMeans Demo](kmeans_demo.ipynb) | Working | Predict using k-means, visualize and compare the results with Scikit-learn's k-means.
[KMeans Multi-Node Multi-GPU Demo](kmeans_mnmg_demo.ipynb) | Working | Predict with MNMG k-means using dask distributed inputs.
[Linear Regression Demo](linear_regression_demo.ipynb) | Working | Demonstrate the use of OLS Linear Regression for prediction.
[Nearest Neighbors Demo](nearest_neighbors_demo.ipynb) | Working | Predict using Nearest Neighbors algorithm.
[Random Forest Demo](random_forest_demo.ipynb) | Working | Use Random Forest for classification, and demonstrate how to pickle the cuML model.
[Random Forest Multi-Node Multi-GPU Demo](random_forest_mnmg_demo.ipynb) | Working | Solve a classification problem using MNMG Random Forest.
[Target Encoder Walkthrough](target_encoder_walkthrough.ipynb) | Working | Understand how to use target encoding and why it is preferred over one-hot and label encoding with the help of criteo dataset for click-through rate modelling.

## For more details
Many more examples can be found in the [RAPIDS Notebooks
Contrib](https://github.com/rapidsai/notebooks-contrib) repository,
which contains community-maintained notebooks.
