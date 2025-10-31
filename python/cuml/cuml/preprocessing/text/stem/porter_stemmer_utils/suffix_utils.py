#
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cudf
import cupy as cp


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


def get_stem_series(word_str_ser, suffix_len, can_replace_mask):
    """
    word_str_ser: input string column
    suffix_len: length of suffix to replace
    can_repalce_mask: bool array marking strings where to replace
    """
    starts = cudf.Series(cp.zeros(len(word_str_ser), dtype=cp.int32))
    stops = (word_str_ser.str.len() - suffix_len).fillna(0).astype(cp.int32)
    sliced = word_str_ser.str.slice_from(starts, stops)
    return sliced.where(can_replace_mask, word_str_ser)
