#
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import cudf
import numpy as np

from cuml.preprocessing.text.stem.porter_stemmer_utils import (
    porter_stemmer_rules,
)


def test_ends_with_suffix():
    test_strs = cudf.Series(["happy", "apple", "nappy", ""])
    expect = np.asarray([True, False, True, False])
    got = porter_stemmer_rules.ends_with_suffix(test_strs, "ppy").values.get()
    np.testing.assert_array_equal(got, expect)


def test_ends_with_empty_suffix():
    test_strs = cudf.Series(["happy", "sad"])
    expect = np.asarray([True, True])
    got = porter_stemmer_rules.ends_with_suffix(test_strs, "").values.get()
    np.testing.assert_array_equal(got, expect)
