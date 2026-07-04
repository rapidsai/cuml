#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from cuml.benchmark import datagen


def test_sparsify_and_convert_invalid_input_type_raises_typeerror():
    data = np.zeros((4, 4), dtype=np.float32)

    with pytest.raises(TypeError, match="Wrong sparse input type unknown"):
        datagen._sparsify_and_convert(data, "unknown")
