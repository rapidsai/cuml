# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
from cuml.common import with_cupy_rmm
from cuml.internals.import_utils import has_daskglm

import cupy as cp
import numpy as np
from dask.utils import is_dataframe_like, is_series_like, is_arraylike
import cudf


class LogisticRegression(BaseEstimator):
    """
    Distributed Logistic Regression for Binary classification.


    Parameters
    ----------
    fit_intercept: boolean (default = True)
       If True, the model tries to correct for the global mean of y.
       If False, the model expects that you have centered the data.
    solver : 'admm'
        Solver to use. Only admm is supported currently.
    penalty : {'l1', 'l2', 'elastic_net'} (default = 'l2')
        Regularization technique for the solver.
    C: float (default = 1.0)
       Inverse of regularization strength; must be a positive float.
    max_iter: int (default = 100)
        Maximum number of iterations taken for the solvers to converge.
    verbose : int or boolean (default=False)
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    Attributes
    ----------
    coef_: device array (n_features, 1)
        The estimated coefficients for the logistic regression model.
    intercept_: device array (1,)
        The independent term. If `fit_intercept` is False, will be 0.
    solver: string
        Algorithm to use in the optimization process. Currently only `admm` is
        supported.

    Notes
    ------

    This estimator is a wrapper class around Dask-GLM's
    Logistic Regression estimator. Several methods in this wrapper class
    duplicate code from Dask-GLM to create a user-friendly namespace.
    """

    def __init__(
        self,
        *,
        client=None,
        fit_intercept=True,
        solver="admm",
        penalty="l2",
        C=1.0,
        max_iter=100,
        verbose=False,
        **kwargs
    ):
        super(LogisticRegression, self).__init__(
            client=client, verbose=verbose, **kwargs
        )

        if not has_daskglm("0.2.1.dev"):
            raise ImportError(
                "dask-glm >= 0.2.1.dev was not found, please install it"
                " to use multi-GPU logistic regression."
            )

        from dask_glm.estimators import LogisticRegression \
            as LogisticRegressionGLM

        self.fit_intercept = fit_intercept
        self.solver = solver
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter

        if self.penalty not in ("l2", "l1", "elastic_net"):
            raise TypeError(
                "Only l2, l1, and elastic_net penalties are"
                " currently supported."
            )

        self.solver_model = LogisticRegressionGLM(
            solver=self.solver,
            fit_intercept=self.fit_intercept,
            regularizer=self.penalty,
            max_iter=self.max_iter,
            lamduh=1 / self.C,
        )

    @with_cupy_rmm
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

        X = self._input_to_dask_cupy_array(X)
        y = self._input_to_dask_cupy_array(y)
        self.solver_model.fit(X, y)
        self._finalize_coefs()
        return self

    @with_cupy_rmm
    def predict(self, X):
        """
        Predicts the ŷ for X.

        Parameters
        ----------
        X : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, n_features)
            Distributed dense matrix (floats or doubles) of shape
            (n_samples, n_features).

        Returns
        -------
        y : Dask cuDF Series or CuPy backed Dask Array (n_rows,)
        """
        return self.predict_proba(X) > 0.5

    @with_cupy_rmm
    def predict_proba(self, X):
        from dask_glm.utils import sigmoid

        X = self._input_to_dask_cupy_array(X)
        return sigmoid(self.decision_function(X))

    @with_cupy_rmm
    def decision_function(self, X):
        X = self._input_to_dask_cupy_array(X)
        X_ = self._maybe_add_intercept(X)
        return np.dot(X_, self._coef)

    @with_cupy_rmm
    def score(self, X, y):
        from dask_glm.utils import accuracy_score

        X = self._input_to_dask_cupy_array(X)
        y = self._input_to_dask_cupy_array(y)
        return accuracy_score(y, self.predict(X))

    @with_cupy_rmm
    def _finalize_coefs(self):
        # _coef contains coefficients and (potentially) intercept
        self._coef = cp.asarray(self.solver_model._coef)
        if self.fit_intercept:
            self.coef_ = self._coef[:-1]
            self.intercept_ = self.solver_model._coef[-1]
        else:
            self.coef_ = self._coef

    @with_cupy_rmm
    def _maybe_add_intercept(self, X):
        from dask_glm.utils import add_intercept

        if self.fit_intercept:
            return add_intercept(X)
        else:
            return X

    @with_cupy_rmm
    def _input_to_dask_cupy_array(self, X):
        if (is_dataframe_like(X) or is_series_like(X)) and hasattr(X, "dask"):

            if not isinstance(X._meta, (cudf.Series, cudf.DataFrame)):
                raise TypeError(
                    "Please convert your Dask DataFrame"
                    " to a Dask-cuDF DataFrame using dask_cudf."
                )
            X = X.values
            X._meta = cp.asarray(X._meta)

        elif is_arraylike(X) and hasattr(X, "dask"):
            if not isinstance(X._meta, cp.ndarray):
                raise TypeError(
                    "Please convert your CPU Dask Array"
                    " to a GPU Dask Array using"
                    " arr.map_blocks(cp.asarray)."
                )
        else:
            raise TypeError("Please pass a GPU backed Dask DataFrame"
                            " or Dask Array.")

        X.compute_chunk_sizes()
        return X

    def get_param_names(self):
        return list(self.kwargs.keys())
