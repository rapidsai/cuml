#!/usr/bin/env python3
#
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Simple demo to test cuML's acceleration profiling capabilities.
"""

import sys
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
import numpy as np

def main():
    # Generate synthetic data
    X, y = make_regression(n_samples=1000, n_features=10, random_state=42)

    # Create and use a proxy estimator
    estimator = Ridge(random_state=42)

    print("Fitting estimator with 1D y...", file=sys.stderr)
    estimator.fit(X, y)
    estimator.predict(X)

    # Trigger SYNC_ATTRS by accessing fit attributes
    print("\nAccessing fit attributes (triggers SYNC_ATTRS)...", file=sys.stderr)
    print(f"  coef_: {estimator.coef_.shape}")
    print(f"  intercept_: {estimator.intercept_}")
    print(f"  n_features_in_: {estimator.n_features_in_}")

    # Trigger SYNC_PARAMS by changing parameters
    print("\nChanging parameters (triggers SYNC_PARAMS)...", file=sys.stderr)
    estimator.alpha = 2.0
    estimator.fit(X, y)

    # Access attributes again to trigger another SYNC_ATTRS
    print("\nAccessing fit attributes again...", file=sys.stderr)
    print(f"  coef_: {estimator.coef_.shape}")
    print(f"  intercept_: {estimator.intercept_}")

    # Second fit with multi-dimensional y (will trigger UnsupportedOnGPU)
    print("\nTrying fit with multi-dimensional y...", file=sys.stderr)
    y_2d = y.reshape(-1, 1)  # Make y 2D
    estimator.fit(X, y_2d)
    estimator.predict(X)

    # Access attributes after CPU fallback (triggers SYNC_ATTRS)
    print("\nAccessing attributes after CPU fallback...", file=sys.stderr)
    print(f"  coef_: {estimator.coef_.shape}")
    print(f"  intercept_: {estimator.intercept_}")

    # Try to change parameters again (this will trigger SYNC_PARAMS but fail)
    print("\nTrying to change parameters after CPU fallback...", file=sys.stderr)
    estimator.alpha = 1.5
    estimator.fit(X, y_2d)

    # Try to initialize estimator with unsupported parameters
    estimator = KMeans(init=lambda X, n_clusters, random_state: np.random.rand(n_clusters, X.shape[1]))
    estimator.fit(X)
    estimator.predict(X)




if __name__ == "__main__":
    main()
