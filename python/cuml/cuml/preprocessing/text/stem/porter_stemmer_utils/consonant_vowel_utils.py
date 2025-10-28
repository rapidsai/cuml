#
# SPDX-FileCopyrightText: Copyright (c) 2020-2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#


def is_consonant(str_ser, i):
    """Returns True if word[i] is a consonant, False otherwise

    A consonant is defined in the paper as follows:
        A consonant in a word is a letter other than A, E, I, O or
        U, and other than Y preceded by a consonant. (The fact that
        the term `consonant' is defined to some extent in terms of
        itself does not make it ambiguous.) So in TOY the consonants
        are T and Y, and in SYZYGY they are S, Z and G. If a letter
        is not a consonant it is a vowel.
    """
    return str_ser.str.is_consonant(i)


def is_vowel(str_ser, i):
    """Returns True if word[i] is a vowel, False otherwise
    see:  is_consonant for more description
    """
    return str_ser.str.is_vowel(i)


def contains_vowel(stem_ser):
    """
    Returns True if stem contains a vowel, else False
    """
    len_ser = stem_ser.str.len()
    max_len = len_ser.max()
    contains_vowel_flag = None

    for i in range(0, max_len):
        if contains_vowel_flag is None:
            contains_vowel_flag = is_vowel(stem_ser, i)
        else:
            contains_vowel_flag = contains_vowel_flag | is_vowel(stem_ser, i)

    return contains_vowel_flag
