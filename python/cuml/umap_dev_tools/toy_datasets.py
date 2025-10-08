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

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
    make_blobs,
    make_s_curve,
    make_swiss_roll,
)
from sklearn.preprocessing import StandardScaler


def generate_sphere(n_samples=1000, noise=0.1, random_state=42):
    """Generate points on a 3D sphere with optional noise."""
    np.random.seed(random_state)

    # Generate random points on sphere using spherical coordinates
    theta = np.random.uniform(0, 2 * np.pi, n_samples)  # azimuth
    phi = np.random.uniform(0, np.pi, n_samples)  # polar angle

    # Convert to Cartesian coordinates
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Add noise
    if noise > 0:
        x += np.random.normal(0, noise, n_samples)
        y += np.random.normal(0, noise, n_samples)
        z += np.random.normal(0, noise, n_samples)

    # Color by phi (latitude)
    colors = phi

    return np.column_stack([x, y, z]), colors


def generate_torus(n_samples=1000, R=3, r=1, noise=0.1, random_state=42):
    """Generate points on a 3D torus."""
    np.random.seed(random_state)

    # Generate random angles
    theta = np.random.uniform(0, 2 * np.pi, n_samples)  # angle around the tube
    phi = np.random.uniform(0, 2 * np.pi, n_samples)  # angle around the torus

    # Convert to Cartesian coordinates
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)

    # Add noise
    if noise > 0:
        x += np.random.normal(0, noise, n_samples)
        y += np.random.normal(0, noise, n_samples)
        z += np.random.normal(0, noise, n_samples)

    # Color by theta (position around tube)
    colors = theta

    return np.column_stack([x, y, z]), colors


def generate_datasets():
    datasets = {}

    # =============================
    # Generating synthetic datasets
    # =============================

    # Swiss roll
    X_swiss, color_swiss = make_swiss_roll(
        n_samples=1000, noise=0.1, random_state=42
    )
    datasets["Swiss Roll"] = (X_swiss, color_swiss)

    # S-curve
    X_scurve, color_scurve = make_s_curve(
        n_samples=1000, noise=0.1, random_state=42
    )
    datasets["S-Curve"] = (X_scurve, color_scurve)

    # Sphere
    X_sphere, color_sphere = generate_sphere(
        n_samples=1000, noise=0.05, random_state=42
    )
    datasets["Sphere"] = (X_sphere, color_sphere)

    # Torus
    X_torus, color_torus = generate_torus(
        n_samples=1000, R=3, r=1, noise=0.05, random_state=42
    )
    datasets["Torus"] = (X_torus, color_torus)

    # Gaussian blobs
    X_blobs, y_blobs = make_blobs(
        n_samples=1000,
        centers=4,
        n_features=8,
        cluster_std=1.5,
        random_state=42,
    )
    # Standardize the high-dimensional blobs
    X_blobs = StandardScaler().fit_transform(X_blobs)
    datasets["Gaussian Blobs"] = (X_blobs, y_blobs)

    # ===========================
    # Classic scikit-learn datasets
    # ===========================

    # Iris
    iris = load_iris()
    X_iris = StandardScaler().fit_transform(iris.data)
    datasets["Iris"] = (X_iris, iris.target)

    # Wine
    wine = load_wine()
    X_wine = StandardScaler().fit_transform(wine.data)
    datasets["Wine"] = (X_wine, wine.target)

    # Breast Cancer Wisconsin
    bc = load_breast_cancer()
    X_bc = StandardScaler().fit_transform(bc.data)
    datasets["Breast Cancer"] = (X_bc, bc.target)

    # Digits (64-D pixel intensities for 8×8 images)
    digits = load_digits()
    X_digits = StandardScaler().fit_transform(digits.data)
    datasets["Digits"] = (X_digits, digits.target)

    # Diabetes (regression dataset) – use target discretised for colouring
    diabetes = load_diabetes()
    X_diabetes = StandardScaler().fit_transform(diabetes.data)
    # Discretise continuous target into 5 bins for colouring
    y_diabetes = np.digitize(
        diabetes.target,
        bins=np.quantile(diabetes.target, [0.2, 0.4, 0.6, 0.8]),
    )
    datasets["Diabetes"] = (X_diabetes, y_diabetes)

    return datasets
