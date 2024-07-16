#
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

from cuml.preprocessing.text.stem.porter_stemmer import PorterStemmer
from cuml.internals.safe_imports import gpu_only_import

cudf = gpu_only_import("cudf")


def test_step1a():
    word_str_ser = cudf.Series(
        ["caresses", "ponies", "ties", "caress", "cats"]
    )

    st = PorterStemmer()
    got = st._step1a(word_str_ser)

    expect = ["caress", "poni", "tie", "caress", "cat"]
    assert list(got.to_pandas().values) == expect

    # mask test
    mask = cudf.Series([True, False, True, True, False])
    expect = ["caress", "ponies", "tie", "caress", "cats"]
    got = st._step1a(word_str_ser, mask)

    assert list(got.to_pandas().values) == expect


def test_step1b():
    word_str_ser_ls = [
        "feed",
        "agreed",
        "plastered",
        "bled",
        "motoring",
        "sing",
        "conflated",
        "troubled",
        "sized",
        "hopping",
        "tanned",
        "falling",
        "hissing",
        "fizzed",
        "failing",
        "filing",
    ]

    expected = [
        "feed",
        "agree",
        "plaster",
        "bled",
        "motor",
        "sing",
        "conflate",
        "trouble",
        "size",
        "hop",
        "tan",
        "fall",
        "hiss",
        "fizz",
        "fail",
        "file",
    ]

    word_str_ser = cudf.Series(word_str_ser_ls)
    st = PorterStemmer()
    got = st._step1b(word_str_ser)

    assert list(got.to_pandas().values) == expected

    # mask test
    expected = expected[:-3] + ["fizzed", "failing", "filing"]
    mask = cudf.Series([True] * (len(expected) - 3) + [False] * 3)
    got = st._step1b(word_str_ser, mask)
    assert list(got.to_pandas().values) == expected


def test_step1c():
    word_str_ser_ls = ["happy", "sky", "enjoy", "boy", "toy", "y"]
    word_str_ser = cudf.Series(word_str_ser_ls)
    st = PorterStemmer()
    got = st._step1c(word_str_ser)

    expect = ["happi", "ski", "enjoy", "boy", "toy", "y"]
    assert list(got.to_pandas().values) == expect

    # mask test
    expect = ["happi", "sky", "enjoy", "boy", "toy", "y"]
    mask = cudf.Series([True, False, False, False, False, True])
    got = st._step1c(word_str_ser, mask)
    assert list(got.to_pandas().values) == expect


def test_step2():
    word_str_ser_ls = [
        "relational",
        "conditional",
        "rational",
        "valenci",
        "hesitanci",
        "digitizer",
        "conformabli",
        "radicalli",
        "differentli",
        "vileli",
        "analogousli",
        "vietnamization",
        "predication",
        "operator",
        "feudalism",
        "decisiveness",
        "hopefulness",
        "callousness",
        "formaliti",
        "sensitiviti",
        "sensibiliti",
    ]

    expect = [
        "relate",
        "condition",
        "rational",
        "valence",
        "hesitance",
        "digitize",
        "conformable",
        "radical",
        "different",
        "vile",
        "analogous",
        "vietnamize",
        "predicate",
        "operate",
        "feudal",
        "decisive",
        "hopeful",
        "callous",
        "formal",
        "sensitive",
        "sensible",
    ]

    word_str_ser = cudf.Series(word_str_ser_ls)
    st = PorterStemmer()
    got = st._step2(word_str_ser)
    assert list(got.to_pandas().values) == expect

    # mask test
    expect = expect[:-3] + ["formaliti", "sensitiviti", "sensibiliti"]
    mask = cudf.Series([True] * (len(expect) - 3) + [False] * 3)
    got = st._step2(word_str_ser, mask)
    assert list(got.to_pandas().values) == expect


def test_step3():
    word_str_ser_ls = [
        "triplicate",
        "formative",
        "formalize",
        "electriciti",
        "electriciti",
        "hopeful",
        "goodness",
    ]
    expect = [
        "triplic",
        "form",
        "formal",
        "electric",
        "electric",
        "hope",
        "good",
    ]

    word_str_ser = cudf.Series(word_str_ser_ls)
    st = PorterStemmer()
    got = st._step3(word_str_ser)
    assert list(got.to_pandas().values) == expect

    # mask test
    expect = expect[:-2] + ["hopeful", "goodness"]
    mask = cudf.Series([True] * (len(expect) - 2) + [False] * 2)
    got = st._step3(word_str_ser, mask)
    assert list(got.to_pandas().values) == expect


def test_step4():
    word_str_ser_ls = [
        "revival",
        "allowance",
        "inference",
        "airliner",
        "gyroscopic",
        "adjustable",
        "defensible",
        "irritant",
        "replacement",
        "adjustment",
        "dependent",
        "adoption",
        "homologou",
        "communism",
        "activate",
        "angulariti",
        "homologous",
        "effective",
        "bowdlerize",
    ]

    expect = [
        "reviv",
        "allow",
        "infer",
        "airlin",
        "gyroscop",
        "adjust",
        "defens",
        "irrit",
        "replac",
        "adjust",
        "depend",
        "adopt",
        "homolog",
        "commun",
        "activ",
        "angular",
        "homolog",
        "effect",
        "bowdler",
    ]

    word_str_ser = cudf.Series(word_str_ser_ls)
    st = PorterStemmer()
    got = st._step4(word_str_ser)
    assert list(got.to_pandas().values) == expect

    # mask test
    expect = expect[:-2] + ["effective", "bowdlerize"]
    mask = cudf.Series([True] * (len(expect) - 2) + [False] * 2)
    got = st._step4(word_str_ser, mask)
    assert list(got.to_pandas().values) == expect


def test_step5a():
    word_str_ser_ls = ["probate", "rate", "cease", "ones"]
    word_str_ser = cudf.Series(word_str_ser_ls)

    expect = ["probat", "rate", "ceas", "ones"]
    st = PorterStemmer()
    got = st._step5a(word_str_ser)
    assert list(got.to_pandas().values) == expect

    # mask test
    expect = expect[:-2] + ["cease", "ones"]
    mask = cudf.Series([True] * (len(expect) - 2) + [False] * 2)
    got = st._step5a(word_str_ser, mask)
    assert list(got.to_pandas().values) == expect


def test_step5b():
    word_str_ser_ls = ["controll", "roll"]
    word_str_ser = cudf.Series(word_str_ser_ls)
    expect = ["control", "roll"]

    st = PorterStemmer()
    got = st._step5b(word_str_ser)
    assert list(got.to_pandas().values) == expect

    # mask test
    expect = ["controll", "roll"]
    mask = cudf.Series([False, True])
    got = st._step5b(word_str_ser, mask)
    assert list(got.to_pandas().values) == expect
