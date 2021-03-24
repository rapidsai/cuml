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
import pandas as pd
from dask.utils import is_dataframe_like, is_series_like, is_arraylike
import cudf


class LogisticRegression(BaseEstimator):
    """
    """

    def __init__(self, client=None, verbose=False, **kwargs):
        super(LogisticRegression, self).__init__(client=client,
                                                 verbose=verbose,
                                                 **kwargs)

        if not has_daskglm("0.2.1.dev"):
            raise ImportError(
                "dask-glm >= 0.2.1.dev was not found, please install it"
                " to use multi-GPU logistic regression.")

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

        X = self._input_to_dask_cupy_array(X)
        y = self._input_to_dask_cupy_array(y)
        self.internal_model = LogisticRegressionGLM(**self.kwargs)
        self.internal_model.fit(X, y)
        self._finalize_coefs()
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
        X = self._input_to_dask_cupy_array(X)
        return self.internal_model.predict(X)

    def predict_proba(self, X):
        X = self._input_to_dask_cupy_array(X)
        return self.internal_model.predict_proba(X)

    def decision_function(self, X):
        X = self._input_to_dask_cupy_array(X)
        X_ = self.internal_model._maybe_add_intercept(X)
        return np.dot(X_, self.internal_model._coef)

    def score(self, X, y):
        from dask_glm.utils import accuracy_score

        X = self._input_to_dask_cupy_array(X)
        return accuracy_score(y, self.predict(X))

    def _finalize_coefs(self):
        if self.internal_model.fit_intercept:
            self.coef_ = self.internal_model._coef[:-1]
            self.intercept_ = self.internal_model._coef[-1]
        else:
            self.coef_ = self.internal_model._coef

    def _input_to_dask_cupy_array(self, X):
        if (is_dataframe_like(X) or is_series_like(X)) and \
            hasattr(X, "dask"):
            
            if not isinstance(X._meta, (cudf.Series, cudf.DataFrame)):
                raise TypeError("Please convert your Dask DataFrame" 
                                " to a Dask-cuDF DataFrame using dask_cudf.")
            X = X.values
            X._meta = cp.asarray(X._meta)
                
        elif is_arraylike(X) and hasattr(X, "dask"):
            if not isinstance(X._meta, cp.ndarray):
                raise TypeError("Please convert your CPU Dask Array" 
                                " to a GPU Dask Array using" 
                                " arr.map_blocks(cp.asarray).")
        else:
            raise TypeError(
                "Please pass a GPU backed Dask DataFrame or Dask Array."
            )
        
        X.compute_chunk_sizes()
        return X

    def get_param_names(self):
        return list(self.kwargs.keys())
