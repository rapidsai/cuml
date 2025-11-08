#
# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#


def has_positive_measure(word_ser):
    measure_ser = word_ser.str.porter_stemmer_measure()
    return measure_ser > 0


def measure_gt_n(word_ser, n):
    measure_ser = word_ser.str.porter_stemmer_measure()
    return measure_ser > n


def measure_eq_n(word_ser, n):
    measure_ser = word_ser.str.porter_stemmer_measure()
    return measure_ser == n
