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
