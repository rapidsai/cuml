#
# SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import time
from contextlib import contextmanager


# Helper function for timing blocks of code.
@contextmanager
def timed(name):
    """
    For timing blocks of code.

    Examples
    --------

    >>> with timed("Print Call"):
    ...     print("Hello, World") # doctest: +SKIP
    Hello, World
    ..Print Call              :    0.0005

    """
    t0 = time.time()
    yield
    t1 = time.time()
    print("..%-24s:  %8.4f" % (name, t1 - t0))
