#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import numbers

import cupy as cp
import numpy as np
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
        return 1

    if hasattr(X, "shape"):
        shape = X.shape
    elif hasattr(X, "__cuda_array_interface__"):
        shape = X.__cuda_array_interface__["shape"]
    elif hasattr(X, "__array_interface__"):
        shape = X.__array_interface__["shape"]
    else:
        shape = np.asarray(X).shape

    # TODO: Can remove the fallback to 1 when we finish dropping support
    # for 1D X inputs
    return shape[1] if len(shape) >= 2 else 1


def check_features(estimator, X, reset=False) -> None:
    """Check or set ``n_features_in_``.

    Parameters
    ----------
    estimator : Base
        The estimator to check.
    X : array-like
        The original user-provided `X` input. No conversion or processing steps
        should have occurred to this array yet.
    reset : bool, default=False
        If true, ``n_features_in_`` is set on ``estimator`` to match ``X``.
        Otherwise the ``X`` is checked to match the existing
        ``n_features_in_``. ``reset=True`` should be used for fit-like methods.
    """
    n_features = _get_n_features(X)

    if reset:
        estimator.n_features_in_ = n_features
    else:
        if n_features != estimator.n_features_in_:
            raise ValueError(
                f"X has {n_features} features, but {estimator.__class__.__name__} "
                f"is expecting {estimator.n_features_in_} features as input."
            )
