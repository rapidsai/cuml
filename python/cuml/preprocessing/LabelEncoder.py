import cudf
import nvcategory

from librmm_cffi import librmm
import numpy as np


def _enforce_str(y: cudf.Series) -> cudf.Series:
    if y.dtype != "object":
        return y.astype("str")
    return y


# FIXME I didn't have cuml built. this would be cuml.common.Base
class Base(object):
    def __init__(self, *args, **kwargs):
        self._fitted = False

    def check_is_fitted(self):
        if not self._fitted:
            raise TypeError("Model must first be .fit()")


class LabelEncoder(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cats: nvcategory.nvcategory = None
        self._dtype = None

    def fit(self, y: cudf.Series) -> "LabelEncoder":
        self._dtype = y.dtype
        y = _enforce_str(y)

        self._cats = nvcategory.from_strings(y.data)
        self._fitted = True
        return self

    def transform(self, y: cudf.Series) -> cudf.Series:
        self.check_is_fitted()
        y = _enforce_str(y)
        encoded = cudf.Series(
            nvcategory.from_strings(y.data)
            .set_keys(self._cats.keys())
            .values()
        )
        if -1 in encoded:
            raise KeyError("Attempted to encode unseen key")
        return encoded

    def fit_transform(self, y: cudf.Series) -> cudf.Series:
        self._dtype = y.dtype
        y = _enforce_str(y)
        self._cats = nvcategory.from_strings(y.data)
        self._fitted = True
        arr: librmm.device_array = librmm.device_array(
            y.data.size(), dtype=np.int32
        )
        self._cats.values(devptr=arr.device_ctypes_pointer.value)
        return cudf.Series(arr)

    def inverse_transform(self, y: cudf.Series):
        raise NotImplementedError
