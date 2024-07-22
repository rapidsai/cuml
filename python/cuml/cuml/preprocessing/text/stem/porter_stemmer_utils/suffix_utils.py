#
# Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

from cuml.internals.safe_imports import cpu_only_import
from cuml.internals.safe_imports import gpu_only_import
from cuml.internals.safe_imports import gpu_only_import_from

cuda = gpu_only_import_from("numba", "cuda")
cudf = gpu_only_import("cudf")
np = cpu_only_import("numpy")
cp = gpu_only_import("cupy")


def get_str_replacement_series(replacement, bool_mask):
    """
    Get replacement series with replacement at
    Places marked by bool mask and empty other wise
    """
    word_ser = cudf.Series([""] * len(bool_mask))
    word_ser.iloc[bool_mask] = replacement

    return word_ser


def get_index_replacement_series(word_str_ser, replacment_index, bool_mask):
    """
    Get replacement series with nulls at places marked by bool mask
    """
    valid_indexes = ~bool_mask
    word_str_ser = word_str_ser.str.get(replacment_index)
    word_str_ser = cudf.Series(word_str_ser)
    word_str_ser.iloc[valid_indexes] = ""

    return word_str_ser


def replace_suffix(word_str_ser, suffix, replacement, can_replace_mask):
    """
    replaces string column with valid suffix with replacement
    """

    len_suffix = len(suffix)
    if replacement == "":
        stem_ser = get_stem_series(word_str_ser, len_suffix, can_replace_mask)
        return stem_ser
    else:
        stem_ser = get_stem_series(word_str_ser, len_suffix, can_replace_mask)
        if isinstance(replacement, str):
            replacement_ser = get_str_replacement_series(
                replacement, can_replace_mask
            )
        if isinstance(replacement, int):
            replacement_ser = get_index_replacement_series(
                word_str_ser, replacement, can_replace_mask
            )
        else:
            assert ValueError(
                "replacement: {} value should be a string or a int".format(
                    replacement
                )
            )

        return stem_ser + replacement_ser


@cuda.jit()
def subtract_valid(input_array, valid_bool_array, sub_val):
    pos = cuda.grid(1)
    if pos < input_array.size:
        if valid_bool_array[pos]:
            input_array[pos] = input_array[pos] - sub_val


@cudf.core.buffer.acquire_spill_lock()
def get_stem_series(word_str_ser, suffix_len, can_replace_mask):
    """
    word_str_ser: input string column
    suffix_len: length of suffix to replace
    can_repalce_mask: bool array marking strings where to replace
    """
    NTHRD = 1024
    NBLCK = int(np.ceil(float(len(word_str_ser)) / float(NTHRD)))

    start_series = cudf.Series(cp.zeros(len(word_str_ser), dtype=cp.int32))
    end_ser = word_str_ser.str.len()

    end_ar = end_ser._column.data_array_view(mode="read")
    can_replace_mask_ar = can_replace_mask._column.data_array_view(mode="read")

    subtract_valid[NBLCK, NTHRD](end_ar, can_replace_mask_ar, suffix_len)
    return word_str_ser.str.slice_from(
        starts=start_series, stops=end_ser.fillna(0)
    )
