# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

from dask.distributed import get_worker
from raft_dask.common.comms import get_raft_comm_state

from cuml.dask.common.base import (
    BaseEstimator,
    DelayedPredictionMixin,
    SyncFitMixinLinearModel,
    mnmg_import,
)


class LinearRegression(
    BaseEstimator, SyncFitMixinLinearModel, DelayedPredictionMixin
):
    """
    LinearRegression is a simple machine learning model where the response y is
    modelled by a linear combination of the predictors in X.

    cuML's dask Linear Regression (multi-node multi-gpu) expects dask cuDF
    DataFrame and provides an algorithms, Eig, to fit a linear model.
    And provides an eigendecomposition-based algorithm to fit a linear model.
    (SVD, which is more stable than eig, will be added in an upcoming version.)
    Eig algorithm is usually preferred when the X is a tall and skinny matrix.
    As the number of features in X increases, the accuracy of Eig algorithm
    drops.

    This is an experimental implementation of dask Linear Regression. It
    supports input X that has more than one column. Single column input
    X will be supported after SVD algorithm is added in an upcoming version.

    Parameters
    ----------
    algorithm : 'eig'
        Eig uses a eigendecomposition of the covariance matrix, and is much
        faster.
        SVD is slower, but guaranteed to be stable.
    fit_intercept : boolean (default = True)
        LinearRegression adds an additional term c to correct for the global
        mean of y, modeling the response as "x * beta + c".
        If False, the model expects that you have centered the data.
    normalize : boolean (default = False)
        If True, the predictors in X will be normalized by dividing by its
        L2 norm.
        If False, no scaling will be done.

    Attributes
    ----------
    coef_ : cuDF series, shape (n_features)
        The estimated coefficients for the linear regression model.
    intercept_ : array
        The independent term. If `fit_intercept` is False, will be 0.
    """

    def __init__(self, *, client=None, verbose=False, **kwargs):
        super().__init__(client=client, verbose=verbose, **kwargs)

    def fit(self, X, y):
        """
        Fit the model with X and y.

        Parameters
        ----------
        X : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, n_features)
            Features for regression
        y : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, 1)
            Labels (outcome values)
        """

        models = self._fit(
            model_func=LinearRegression._create_model, data=(X, y)
        )

        self._set_internal_model(models[0])

        return self

    def predict(self, X, delayed=True):
        """
        Make predictions for X and returns a dask collection.

        Parameters
        ----------
        X : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, n_features)
            Distributed dense matrix (floats or doubles) of shape
            (n_samples, n_features).

        delayed : bool (default = True)
            Whether to do a lazy prediction (and return Delayed objects) or an
            eagerly executed one.

        Returns
        -------
        y : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, 1)
        """
        return self._predict(X, delayed=delayed)

    def _get_param_names(self):
        return list(self.kwargs.keys())

    @staticmethod
    @mnmg_import
    def _create_model(sessionId, datatype, **kwargs):
        from cuml.linear_model.linear_regression_mg import LinearRegressionMG

        handle = get_raft_comm_state(sessionId, get_worker())["handle"]
        return LinearRegressionMG(
            handle=handle, output_type=datatype, **kwargs
        )
