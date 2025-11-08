#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import numbers

import cupy as cp
import numpy as np


def check_random_seed(seed):
    """Turn a np.random.RandomState instance into a seed.

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return a random int as seed.
        If seed is an int, return it.
        If seed is a RandomState instance, derive a seed from it.
        Otherwise raise ValueError.
    """
    if seed is None:
        seed = np.random.RandomState(None)

    if isinstance(seed, numbers.Integral):
        return seed
    if isinstance(seed, np.random.RandomState):
        return seed.randint(
            low=0, high=np.iinfo(np.uint32).max, dtype=np.uint32
        )
    if isinstance(seed, cp.random.RandomState):
        return seed.randint(
            low=0, high=np.iinfo(cp.uint32).max, dtype=cp.uint32
        ).get()
    raise ValueError("%r cannot be used to create a seed." % seed)
