import cudf
import numpy as np
import nvtext
from numba import cuda


def has_positive_measure(word_ser):
    measure_ser = cudf.Series(cuda.device_array(len(word_ser), dtype=np.int32))
    nvtext.porter_stemmer_measure(
        word_ser._column.nvstrings, devptr=measure_ser.data.ptr
    )
    return measure_ser > 0


def measure_gt_n(word_ser, n):
    measure_ser = cudf.Series(cuda.device_array(len(word_ser), dtype=np.int32))
    nvtext.porter_stemmer_measure(
        word_ser._column.nvstrings, devptr=measure_ser.data.ptr
    )
    return measure_ser > n


def measure_eq_n(word_ser, n):
    measure_ser = cudf.Series(cuda.device_array(len(word_ser), dtype=np.int32))
    nvtext.porter_stemmer_measure(
        word_ser._column.nvstrings, devptr=measure_ser.data.ptr
    )
    return measure_ser == n
