import cudf
import numpy as np
from cuml.preprocessing.text.stem.porter_stemmer_utils.len_flags_utils import len_eq_n, len_gt_n

def test_has_positive_measure():
    word_strs = cudf.Series(["tr", "ee", "tree", "trouble", "troubles", "private"])
    got = has_positive_measure(word_strs).values.get()
    expect = np.asarray([False, False, False, True, True, True])
    np.testing.assert_array_equal(got, expect)


def test_has_measure_gt_n():
    word_strs = cudf.Series(["tr", "ee", "tree", "trouble", "troubles", "private"])
    got = measure_gt_n(word_strs, 1).values.get()
    expect = np.asarray([False, False, False, False, True, True])
    np.testing.assert_array_equal(got, expect)
