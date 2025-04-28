#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cudf
import numpy as np

from cuml.preprocessing.text.stem.porter_stemmer_utils.len_flags_utils import (
    len_eq_n,
    len_gt_n,
)


def test_len_gt_n():
    word_str_ser = cudf.Series(["a", "abcd", "abc", "abcd"])
    got = len_gt_n(word_str_ser, 3).values.get()
    expect = np.asarray([False, True, False, True])
    np.testing.assert_array_equal(got, expect)


def test_len_eq_n():
    word_str_ser = cudf.Series(["a", "abcd", "abc", "abcd"])
    got = len_eq_n(word_str_ser, 3).values.get()
    expect = np.asarray([False, False, True, False])
    np.testing.assert_array_equal(got, expect)
