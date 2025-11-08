# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#


import cupy as cp


def _create_rs_generator(random_state):
    """
    This is a utility function that returns an instance of CuPy RandomState
    Parameters
    ----------
    random_state : None, int, or CuPy RandomState
        The random_state from which the CuPy random state is generated
    """

    if isinstance(random_state, (type(None), int)):
        return cp.random.RandomState(seed=random_state)
    elif isinstance(random_state, cp.random.RandomState):
        return random_state
    else:
        raise ValueError("random_state type must be int or CuPy RandomState")
