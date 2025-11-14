#
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#


try:
    from nvtx import annotate
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def annotate(*args, **kwargs):
        if len(kwargs) == 0 and len(args) == 1 and callable(args[0]):
            return args[0]
        else:

            def inner(func):
                return func

            return inner
