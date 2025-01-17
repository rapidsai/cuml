# Copyright (c) 2025, NVIDIA CORPORATION.

from cuml import LinearRegression
from cuml import using_output_type

import numpy as np
import cupy as cp
from sklearn.datasets import make_regression

# Create synthetic data
X, y = make_regression(n_samples=20, n_features=2, noise=0.1, random_state=42)

# Instantiate and train the estimator
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
print("Predictions:", predictions, type(predictions))
print("coef", model.coef_, type(model.coef_))

with using_output_type("cupy"):
    assert isinstance(model.coef_, cp.ndarray)

assert isinstance(model.coef_, np.ndarray)
