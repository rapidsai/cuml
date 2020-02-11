# Copyright (c) 2020, NVIDIA CORPORATION.
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

from cuml.preprocessing.label import LabelBinarizer as LB
from dask.distributed import default_client
from cuml.dask.common import extract_arr_partitions, to_sp_dask_array

from cuml.utils import rmm_cupy_ary

import scipy
import dask
import cupy as cp

class LabelBinarizer(object):
    """
    A distributed version of LabelBinarizer for one-hot encoding
    a collection of labels.
    """

    def __init__(self, client=None, **kwargs):

        self.client_ = client if client is not None else default_client()
        self.kwargs = kwargs

        if "sparse_output" in self.kwargs and \
                self.kwargs["sparse_output"] is True:
            raise ValueError("Sparse output not yet "
                             "supported in distributed mode")

    @staticmethod
    def _func_create_model(**kwargs):
        return LB(**kwargs)

    @staticmethod
    def _func_unique_classes(y):
        return cp.unique(y)

    @staticmethod
    def _func_xform(model, y):
        xform_in = cp.asarray(y, dtype=y.dtype)
        return model.transform(xform_in).get()

    @staticmethod
    def _func_inv_xform(model, y, threshold):
        if not cp.sparse.isspmatrix(y) and not scipy.sparse.isspmatrix(y):
            y = cp.asarray(y, dtype=y.dtype)
        return model.inverse_transform(y, threshold)

    def fit(self, y):
        """Fit label binarizer`

        Parameters
        ----------
        y : Dask.Array of shape [n_samples,] or [n_samples, n_classes]
            chunked by row.
            Target values. The 2-d matrix should only contain 0 and 1,
            represents multilabel classification.

        Returns
        -------
        self : returns an instance of self.
        """

        # Take the unique classes and broadcast them all around the cluster.
        futures = self.client_.sync(extract_arr_partitions, y)

        unique = [self.client_.submit(LabelBinarizer._func_unique_classes, f)
                  for w, f in futures]

        classes = self.client_.compute(unique, True)
        self.classes_ = cp.unique(cp.stack(classes, axis=0))

        self.model = LB(**self.kwargs).fit(self.classes_)

        return self

    def fit_transform(self, y):
        """
        Fit the label encoder and return transformed labels

        Parameters
        ----------
        y : array of shape [n_samples,] or [n_samples, n_classes]
            Target values. The 2-d matrix should only contain 0 and 1,
            represents multilabel classification.
        :return:
        """
        return self.fit(y).transform(y)

    def transform(self, y):

        parts = self.client_.sync(extract_arr_partitions, y)

        xform_func = dask.delayed(LabelBinarizer._func_xform)
        meta = cp.sparse.csr_matrix(rmm_cupy_ary(cp.zeros, 1))
        f = [dask.array.from_delayed(
            xform_func(self.model, part), meta=meta, dtype=y.dtype,
            shape=y.shape) for w, part in parts]

        arr = dask.array.stack(f, axis=0)

        def map_func(x):
            cparr = cp.asarray(x, dtype=cp.float32)
            if cparr.ndim == 3:
                cparr = cparr.reshape(cparr.shape[1:])
            return cp.sparse.csr_matrix(cparr)

        return arr.map_blocks(map_func)

    def inverse_transform(self, y, threshold=None):

        parts = self.client_.sync(extract_arr_partitions, y)
        inv_func = dask.delayed(LabelBinarizer._func_inv_xform)

        dtype = self.classes_.dtype
        meta = rmm_cupy_ary(cp.zeros, 1, dtype=dtype)

        f = [dask.array.from_delayed(
            inv_func(self.model, part, threshold),
            dtype=dtype, shape=(y.shape[0],), meta=meta)
             for w, part in parts]

        return dask.array.stack(f, axis=0)
