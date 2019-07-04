import cudf
import numpy as np
from data import MinMaxScaler as MMS


@pytest.mark.parametrize(
        "values",
        [cudf.DataFrame({'a':[1,2,3], 'b': [4,5,6]})])
def test_MinMaxScaler(values):
    mms = MMS()
    mms.fit_transform(values)



@pytest.mark.parametrize(
    "values, expected",
    [(cudf.Series([1, 2, 3]), cudf.Series([0, 0.5, 1.0]))])
def test_minmax_scale(values, expected):
    res = minmax_scale(values)
    assert(len(res[res == expected]) == len(expected))