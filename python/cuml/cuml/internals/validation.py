#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import numbers
import warnings

import cudf
import cupy as cp
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

__all__ = (
    "check_is_fitted",
    "check_random_seed",
    "check_features",
)


def check_random_seed(random_state) -> int:
    """Turn a `random_state` argument into a seed.

    Parameters
    ----------
    random_state : None | int | instance of RandomState
        If random_state is None, return a random int as seed.
        If random_state is an int, return it.
        If random_state is a RandomState instance, derive a seed from it.

    Returns
    -------
    seed : int
        A seed in the range [0, 2**32 - 1].
    """
    if isinstance(random_state, numbers.Integral):
        if random_state < 0 or random_state >= 2**32:
            raise ValueError(
                f"Expected `0 <= random_state <= 2**32 - 1`, got {random_state}"
            )
        return int(random_state)

    if random_state is None:
        randint = np.random.randint
    elif isinstance(
        random_state, (np.random.RandomState, cp.random.RandomState)
    ):
        randint = random_state.randint
    else:
        raise TypeError(
            f"`random_state` must be an `int`, an instance of `RandomState`, or `None`. "
            f"Got {random_state!r} instead."
        )

    # randint returns in [low, high), so high=2**32 to sample all uint32s
    return int(randint(low=0, high=2**32, dtype=np.uint32))


def _get_n_features(X):
    if isinstance(X, (list, tuple)):
        if len(X) == 0:
            return 0
        row = X[0]
        # For non-array inputs, we assume that all nested lists have the same
        # length. This matches sklearn's implementation as well. If this
        # assumption isn't true, then later validation code will error anyway.
        # We only take the length of sub-sequences that numpy wouldn't treat as
        # single elements.
        if not isinstance(row, (str, bytes, dict)):
            try:
                return len(row)
            except Exception:
                pass

    if hasattr(X, "shape"):
        shape = X.shape
    elif hasattr(X, "__cuda_array_interface__"):
        shape = X.__cuda_array_interface__["shape"]
    elif hasattr(X, "__array_interface__"):
        shape = X.__array_interface__["shape"]
    else:
        shape = np.asarray(X).shape

    ndim = len(shape)

    if ndim < 2:
        if isinstance(X, (cudf.Series, pd.Series)):
            msg = (
                f"Expected a 2-dimensional container but got {type(X).__name__} "
                "instead. Pass a DataFrame containing a single row (i.e. "
                "single sample) or a single column (i.e. single feature) "
                "instead."
            )
        else:
            kind = "scalar" if ndim == 0 else f"{ndim}D"
            msg = (
                f"Expected 2D array, got {kind} array instead. Reshape your data "
                "using array.reshape(-1, 1) if your data has a single feature, "
                "or array.reshape(1, -1) if it contains a single sample."
            )
        raise ValueError(msg)
    elif ndim > 2:
        raise ValueError(f"Expected 2D array, got {ndim}D array instead.")

    return shape[1]


def _get_feature_names(X):
    """Get feature names from X.

    Returns
    -------
    names: ndarray or None
        Feature names of `X`. Unrecognized array containers will return `None`.
    """
    if isinstance(X, (pd.DataFrame, cudf.DataFrame)):
        feature_names = np.asarray(X.columns, dtype=object)
    elif hasattr(X, "__dataframe__"):
        feature_names = np.asarray(
            list(X.__dataframe__().column_names()), dtype=object
        )
    else:
        return None

    if len(feature_names) == 0:
        # No features, just return None
        return None

    # Check the types of the column names.
    types = sorted(t.__qualname__ for t in set(type(v) for v in feature_names))
    if len(types) == 1 and types[0] == "str":
        return feature_names
    elif len(types) > 1 and "str" in types:
        raise TypeError(
            "Feature names are only supported if all input features have string names, "
            f"but your input has {types} as feature name / column name types. "
            "If you want feature names to be stored and validated, you must convert "
            "them all to strings, by using X.columns = X.columns.astype(str) for "
            "example. Otherwise you can remove feature / column names from your input "
            "data, or convert them all to a non-string data type."
        )

    return None


def check_features(estimator, X, reset=False) -> None:
    """Check or set ``n_features_in_`` and ``feature_names_in_``.

    Parameters
    ----------
    estimator : Base
        The estimator to check.
    X : array-like
        The original user-provided `X` input. No conversion or processing steps
        should have occurred to this array yet.
    reset : bool, default=False
        If True, ``n_features_in_`` and ``feature_names_in_`` are set on
        ``estimator`` to match ``X``. Otherwise ``X`` is checked to match the
        existing ``n_features_in_`` and ``feature_names_in_``. ``reset=True``
        should be used for fit-like methods, and False otherwise.
    """
    n_features = _get_n_features(X)
    feature_names = _get_feature_names(X)

    if reset:
        estimator.n_features_in_ = n_features
        if feature_names is not None:
            estimator.feature_names_in_ = feature_names
        elif hasattr(estimator, "feature_names_in_"):
            # Clear old feature names if present
            delattr(estimator, "feature_names_in_")
        return

    est_feature_names = getattr(estimator, "feature_names_in_", None)

    # Check feature_names_in_ first
    if est_feature_names is not None or feature_names is not None:
        if est_feature_names is None:
            warnings.warn(
                f"X has feature names, but {estimator.__class__.__name__} was fitted "
                "without feature names"
            )

        elif feature_names is None:
            warnings.warn(
                "X does not have valid feature names, but"
                f" {estimator.__class__.__name__} was fitted with feature names"
            )

        elif len(est_feature_names) != len(feature_names) or np.any(
            est_feature_names != feature_names
        ):
            unexpected = sorted(
                set(feature_names).difference(est_feature_names)
            )
            missing = sorted(set(est_feature_names).difference(feature_names))

            parts = [
                "The feature names should match those that were passed during fit."
            ]
            for heading, names in [
                ("Feature names unseen at fit time:", unexpected),
                ("Feature names seen at fit time, yet now missing:", missing),
            ]:
                if names:
                    parts.append(heading)
                    parts.extend([f"- {name}" for name in names[:5]])
                    if len(names) > 5:
                        parts.append("- ...")

            if not missing and not unexpected:
                parts.append(
                    "Feature names must be in the same order as they were in fit."
                )

            msg = "\n".join(parts)
            raise ValueError(msg)

    # Then check n_features_in_
    if n_features != estimator.n_features_in_:
        raise ValueError(
            f"X has {n_features} features, but {estimator.__class__.__name__} "
            f"is expecting {estimator.n_features_in_} features as input."
        )
