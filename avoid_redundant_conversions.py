# Copyright (c) 2025, NVIDIA CORPORATION.

import cupy as cp
import numpy as np
import hashlib
from collections import defaultdict

conversions = defaultdict(list)


def _hash_array(array):
    array_data = array.tobytes()
    array_shape = str(array.shape).encode()
    hash_object = hashlib.sha256(array_data + array_shape)
    return hash_object.hexdigest()


def get_array_type(X):
    if isinstance(X, cp.ndarray):
        return "cupy"
    elif isinstance(X, np.ndarray):
        return "numpy"
    else:
        raise TypeError(type(X))


def as_cupy_array(X):
    try:
        return X.to_output("cupy")
    except AttributeError:
        if get_array_type(X) == "cupy":
            return X
        else:
            h = _hash_array(X)
            conversions["host2device"].append(h)
            return cp.asarray(X)


def as_numpy_array(X):
    try:
        return X.to_output("numpy")
    except AttributeError:
        if get_array_type(X) == "numpy":
            return X
        else:  # must be cupy in this example
            ret = np.asarray(X.get())
            h = _hash_array(ret)
            conversions["device2host"].append(h)
            return ret


class ArrayConversionCache:

    def __init__(self, *arrays):
        self.arrays = {get_array_type(a): a for a in arrays}

    def to_output(self, kind: str):
        try:
            return self.arrays[kind]
        except KeyError:
            match kind:
                case "numpy":
                    ret = as_numpy_array(self.arrays["cupy"])
                    self.arrays[kind] = ret
                    return ret
                case "cupy":
                    ret = as_cupy_array(self.arrays["numpy"])
                    self.arrays[kind] = ret
                    return ret


class HostNormalizer:
    """Host implementation"""

    def fit(self, X, y=None):
        X_m = as_numpy_array(X)
        self.mean = np.mean(X_m, axis=0)
        self.std = np.std(X_m, axis=0)
        return self

    def transform(self, X):
        X_m = as_numpy_array(X)
        return (X_m - self.mean) / self.std

    def fit_transform(self, X):
        X_m = as_numpy_array(X)
        return self.fit(X_m).transform(X_m)


def compute_some_param(X):
    return np.sum(as_numpy_array(X))


class DeviceSolverModel:
    """CUDA implementation"""

    def __init__(self):
        self.intercept_ = None
        self.coef_ = None

    def fit(self, X, y):
        X_m, y_m = as_cupy_array(X), as_cupy_array(y)

        X_design = cp.hstack([cp.ones((X_m.shape[0], 1)), X])

        # Compute coefficients using normal equation
        weights = cp.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y_m

        # Separate intercept and coefficients
        self.intercept_ = weights[0]
        self.coef_ = weights[1:]

        return self

    def predict(self, X):
        X_m = as_cupy_array(X)
        return X_m @ self.coef_ + self.intercept_


class DeviceThresholder:

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = as_cupy_array(X)
        return cp.where(cp.abs(X_) < 1e-2, 0, X_)


class Estimator:

    def __init__(self, normalizer=None):
        self.normalizer = normalizer

    def _a_bad_fit_implementation(self, X, y=None):
        X_m, y_m = as_cupy_array(X), as_cupy_array(y)
        if self.normalizer:
            # The following conversion is a bug, because it will potentially
            # perform multiple redundant transformations dependent on the
            # unknown normalizer implementation.

            # X_m = self.normalizer.fit_transform(X_m)  # bug

            # This can only be reliably avoided by wrapping the array in some
            # kind of caching primitive.
            X_m = self.normalizer.fit_transform(ArrayConversionCache(X, X_m))

        # The solver model expects device arrays, so we're good here.
        self.solver_model = DeviceSolverModel().fit(X_m, y_m)
        return self

    def fit(self, X, y=None):
        # Calling compute_some_param without knowing the implementation means we
        # introduce a migration bug, unless we use a caching primitive.
        X = ArrayConversionCache(X)
        self.some_param = compute_some_param(X)  # potential bug

        # We don't know the implementation path of the normalizer either.
        X_m = self.normalizer.fit_transform(X) if self.normalizer else X

        # At this point we know the implementation path, but it also doesn't
        # really matter, because we have to pass either the transformed or the
        # original X, but never both.
        self.solver_model = DeviceSolverModel().fit(X_m, y)
        return self

    def predict(self, X):
        # simulating type reflection here
        return as_numpy_array(self.solver_model.predict(X))


def main():
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Create synthetic data
    X, y = make_regression(n_samples=20, n_features=2, noise=0.1, random_state=42)
    X_train, X_test, y_train, _ = train_test_split(X, y)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("thresholder", DeviceThresholder()),
        ("model", Estimator(normalizer=HostNormalizer())),
    ])

    print("Expecting 4 host2device migrations and two device2host migration.")
    pipeline.fit(X_train, y_train)
    pipeline.predict(X_test)

    print("# of total conversions:", {k: len(v) for k, v in conversions.items()})
    print("# of unique conversions:", {k: len(set(v)) for k, v in conversions.items()})
    conversions.clear()  # reset


if __name__ == "__main__":
    main()
