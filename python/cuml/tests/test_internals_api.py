# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pickle

from cuml.internals import GraphBasedDimRedCallback


class CustomCallback(GraphBasedDimRedCallback):
    pass


def test_callback_pickleable():
    obj = CustomCallback()
    buf = pickle.dumps(obj)
    obj2 = pickle.loads(buf)
    assert isinstance(obj2, CustomCallback)
