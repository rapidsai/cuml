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
