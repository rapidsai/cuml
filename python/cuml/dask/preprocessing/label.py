
# Copyright (c) 2019, NVIDIA CORPORATION.
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

from cuml.preprocessing.label import label_binarize
from cuml.preprocessing.label import LabelBinarizer as LB
from dask.distributed import default_client, wait
from cuml.dask.common import extract_ddf_partitions, to_dask_cudf

import cudf
import cupy as cp


class LabelBinarizer(object):

    def __init__(self, client=None, **kwargs):

        self.client_ = client if client is not None else default_client()
        self.kwargs = kwargs

    @staticmethod
    def _func_create_model(**kwargs):
        return LB(**kwargs)

    @staticmethod
    def _func_unique_classes(y):
        return y.unique()

    @staticmethod
    def _func_fit(self, model, y):
        return model.fit(y)

    @staticmethod
    def _func_xform(self, model, y):
        return model.transform(y)

    @staticmethod
    def _func_inv_xform(self, model, y, threshold):
        return model.inverse_transform(y, threshold)

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
                  for f in futures]

        classes = self.client_.compute(unique)
        classes = cudf.concat(classes).unique().as_gpu_matrix()

        self.model = LB(**self.kwargs)
        self.classes_ = cp.array(classes)

        f = [self.client_.submit(f, classes) for f in self.models_]
        wait(f)

        return self

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def transform(self, y):

        parts = self.client_.sync(extract_ddf_partitions, y)
        f = [self.client_.submit(LabelBinarizer._func_xform,
                                 self.model,
                                 part) for part in parts]

        return to_dask_cudf(f)

    def inverse_transform(self, y, threshold=None):

        parts = self.client_.sync(extract_ddf_partitions, y)
        f = [self.client_.submit(LabelBinarizer._func_inv_xform,
                                 self.model, part, threshold)
             for part in parts]

        return to_dask_cudf(f)


