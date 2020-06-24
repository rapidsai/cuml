import cudf
import numpy as np
from cuml.preprocessing.text.stem.porter_stemmer_rules import ends_with_suffix


def test_ends_with_suffix():
    test_strs = cudf.Series(["happy", "apple", "nappy", ""])
    expect = np.asarray([True, False, True, False])
    got = ends_with_suffix(test_strs, "ppy").values.get()
    np.testing.assert_array_equal(got, expect)


def test_ends_with_empty_suffix():
    test_strs = cudf.Series(["happy", "sad"])
    expect = np.asarray([True, True])
    got = ends_with_suffix(test_strs, "").values.get()
    np.testing.assert_array_equal(got, expect)
