# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import cupy as cp
import numpy as np
import pytest

from cuml.internals.validation import check_random_seed


@pytest.mark.parametrize(
    "seed",
    [
        pytest.param(None, id="none"),
        pytest.param(42, id="int"),
        pytest.param(np.random.RandomState(42), id="numpy"),
        pytest.param(cp.random.RandomState(42), id="cupy"),
    ],
)
def test_check_random_seed(seed):
    res = check_random_seed(seed)
    assert isinstance(res, int)
    assert 0 <= res <= (2**32 - 1)  # in range for uint32
    if isinstance(seed, int):
        assert check_random_seed(seed) == res


def test_check_random_seed_errors():
    for bad in [-1, 2**32]:
        with pytest.raises(
            ValueError, match=r"Expected `0 <= random_state <= 2\*\*32 - 1`"
        ):
            check_random_seed(bad)
    for ok in [0, 2**32 - 1]:
        check_random_seed(ok)
    with pytest.raises(TypeError, match="`random_state` must be"):
        check_random_seed("incorrect type")
