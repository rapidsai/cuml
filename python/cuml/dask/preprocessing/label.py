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
from cuml.dask.common import extract_ddf_partitions, to_dask_cudf

import numba.cuda

import cudf
import cupy as cp


def cp_to_df(cp_ndarr, sparse):
    numba_arr = numba.cuda.as_cuda_array(cp_ndarr)
    if not sparse:
        return cudf.DataFrame.from_gpu_matrix(numba_arr)
    else:
        raise ValueError("Sparse outputs are not yet supported")


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
        return y.unique()

    @staticmethod
    def _func_xform(model, y):
        xform_in = cp.asarray(y.to_gpu_array(), dtype=y.dtype)

        xformed = model.transform(xform_in)
        return cp_to_df(xformed, model.sparse_output)

    @staticmethod
    def _func_inv_xform(model, y, threshold):
        inv_xform_in = cp.asarray(y.to_gpu_matrix(), dtype=y.dtype)
        return cudf.Series(model.inverse_transform(inv_xform_in, threshold))

    def fit(self, y):
        """Fit label binarizer`

        Parameters
        ----------
        y : array of shape [n_samples,] or [n_samples, n_classes]
            Target values. The 2-d matrix should only contain 0 and 1,
            represents multilabel classification.

        Returns
        -------
        self : returns an instance of self.
        """

        # Take the unique classes and broadcast them all around the cluster.
        futures = self.client_.sync(extract_ddf_partitions, y)

        unique = [self.client_.submit(LabelBinarizer._func_unique_classes, f)
                  for w, f in futures]

        classes = self.client_.compute(unique, True)
        classes = cudf.concat(classes).unique().to_gpu_array()

        self.classes_ = cp.asarray(classes, dtype=y.dtype)

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

        parts = self.client_.sync(extract_ddf_partitions, y)
        f = [self.client_.submit(LabelBinarizer._func_xform,
                                 self.model,
                                 part) for w, part in parts]

        # Assume dense output for now
        return to_dask_cudf(f)

    def inverse_transform(self, y, threshold=None):

        parts = self.client_.sync(extract_ddf_partitions, y)
        f = [self.client_.submit(LabelBinarizer._func_inv_xform,
                                 self.model, part, threshold)
             for w, part in parts]

        return to_dask_cudf(f)
