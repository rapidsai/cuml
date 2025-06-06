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

from cuml.preprocessing.text.stem.porter_stemmer_utils.suffix_utils import (
    get_stem_series,
    replace_suffix,
)


def test_get_stem_series():
    word_str_ser = cudf.Series(
        ["ihop", "packit", "mishit", "crow", "girl", "boy"]
    )
    can_replace_mask = cudf.Series([True, True, True, False, False, False])

    expect = ["ih", "pack", "mish", "crow", "girl", "boy"]
    got = get_stem_series(
        word_str_ser, suffix_len=2, can_replace_mask=can_replace_mask
    )
    assert sorted(list(got.to_pandas().values)) == sorted(expect)


def test_replace_suffix():
    # test 'ing' -> 's'
    word_str_ser = cudf.Series(
        ["shopping", "parking", "drinking", "sing", "bing"]
    )
    can_replace_mask = cudf.Series([True, True, True, False, False])
    got = replace_suffix(word_str_ser, "ing", "s", can_replace_mask)
    expect = ["shopps", "parks", "drinks", "sing", "bing"]
    assert sorted(list(got.to_pandas().values)) == sorted(expect)

    # basic test 'ies' -> 's'
    word_str_ser = cudf.Series(["shops", "ties"])
    can_replace_mask = cudf.Series([False, True])
    got = replace_suffix(word_str_ser, "ies", "i", can_replace_mask)

    expect = ["shops", "ti"]
    assert sorted(list(got.to_pandas().values)) == sorted(expect)
