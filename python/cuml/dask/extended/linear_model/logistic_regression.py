# Copyright (c) 2021, NVIDIA CORPORATION.
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

from cuml.dask.common.base import BaseEstimator
from cuml.common.import_utils import has_daskglm
import cupy as cp
import numpy as np
import dask_cudf


class LogisticRegression(BaseEstimator):

    def __init__(self, client=None, verbose=False, **kwargs):
        super(LogisticRegression, self).__init__(client=client,
                                                 verbose=verbose,
                                                 **kwargs)
        
        if not has_daskglm("0.2.1.dev"):
            raise ImportError("dask-glm >= 0.2.1.dev was not found, please install it "
                              " to use multi-GPU logistic regression. ")

    def fit(self, X, y):
        """
        Fit the model with X and y.

        Parameters
        ----------
        X : Dask cuDF dataframe or CuPy backed Dask Array (n_rows, n_features)
            Features for regression
        y : Dask cuDF Series or CuPy backed Dask Array (n_rows,)
            Label (outcome values)
        """
        from dask_glm.estimators import LogisticRegression as LogisticRegressionGLM

        X = self._to_dask_cupy_array(X)
        y = self._to_dask_cupy_array(y)
        lr = LogisticRegressionGLM(**self.kwargs)
        lr.fit(X, y)
        self.lr = lr
        self.coef_ = self.lr.coef_
        self.intercept_ = self.lr.intercept_
        return self

    def predict(self, X):
        """
        Make predictions for X and returns a dask collection.

        Parameters
        ----------
        X : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, n_features)
            Distributed dense matrix (floats or doubles) of shape
            (n_samples, n_features).

        Returns
        -------
        y : Dask cuDF Series or CuPy backed Dask Array (n_rows,)
        """
        X = self._to_dask_cupy_array(X)
        return self.lr.predict(X)
    
    def predict_proba(self, X):
        return self.lr.predict_proba(X)
    
    def decision_function(self, X):
        X_ = self.lr._maybe_add_intercept(X)
        return np.dot(X_, self.lr._coef)

    def _to_dask_cupy_array(self, X):
        if isinstance(X, dask_cudf.DataFrame) or \
           isinstance(X, dask_cudf.Series):
            X = X.values
            X._meta = cp.asarray(X._meta)
        X.compute_chunk_sizes()
        return X

    def get_param_names(self):
        return list(self.kwargs.keys())
