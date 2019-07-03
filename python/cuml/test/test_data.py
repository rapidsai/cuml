import cudf
import numpy as np
from data import MinMaxScaler as MMS



@pytest.mark.parametrize(
        "fit, transform, ",
        [cudf.DataFrame({'a':[1,2,3], 'b': [4,5,6]})])
def test_MinMaxScaler(values):
    mms = MMS()
    mms.fit(values)
    mms.transform()
    assert()


def test_StandardScaler()