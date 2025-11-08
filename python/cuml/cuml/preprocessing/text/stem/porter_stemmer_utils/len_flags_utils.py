#
# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#


def len_gt_n(word_str_ser, n):
    return word_str_ser.str.len() > n


def len_eq_n(word_str_ser, n):
    return word_str_ser.str.len() == n
