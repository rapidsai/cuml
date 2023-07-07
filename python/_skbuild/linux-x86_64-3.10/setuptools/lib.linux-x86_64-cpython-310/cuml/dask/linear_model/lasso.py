#
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

from cuml.dask.solvers import CD
from cuml.dask.common.base import BaseEstimator


class Lasso(BaseEstimator):

    """
    Lasso extends LinearRegression by providing L1 regularization on the
    coefficients when predicting response y with a linear combination of the
    predictors in X. It can zero some of the coefficients for feature
    selection and improves the conditioning of the problem.

    cuML's Lasso an array-like object or cuDF DataFrame and
    uses coordinate descent to fit a linear model.

    Parameters
    ----------
    alpha : float (default = 1.0)
        Constant that multiplies the L1 term.
        alpha = 0 is equivalent to an ordinary least square, solved by the
        LinearRegression class.
        For numerical reasons, using alpha = 0 with the Lasso class is not
        advised.
        Given this, you should use the LinearRegression class.
    fit_intercept : boolean (default = True)
        If True, Lasso tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    normalize : boolean (default = False)
        If True, the predictors in X will be normalized by dividing by it's L2
        norm.
        If False, no scaling will be done.
    max_iter : int (default = 1000)
        The maximum number of iterations
    tol : float (default = 1e-3)
        The tolerance for the optimization: if the updates are smaller than
        tol, the optimization code checks the dual gap for optimality and
        continues until it is smaller than tol.
    selection : {'cyclic', 'random'} (default='cyclic')
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default.
        This (setting to 'random') often leads to significantly faster
        convergence especially when tol is higher than 1e-4.

    Attributes
    ----------
    coef_ : array, shape (n_features)
        The estimated coefficients for the linear regression model.
    intercept_ : array
        The independent term. If `fit_intercept` is False, will be 0.

    For additional docs, see `scikitlearn's Lasso
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>`_.
    """

    def __init__(self, *, client=None, **kwargs):
        super().__init__(client=client, **kwargs)

        kwargs["shuffle"] = False

        if "selection" in kwargs:
            if kwargs["selection"] == "random":
                kwargs["shuffle"] = True

            del kwargs["selection"]

        self.solver = CD(client=client, **kwargs)

    def fit(self, X, y):
        """
        Fit the model with X and y.

        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array
            Dense matrix (floats or doubles) of shape (n_samples, n_features).

        y : Dask cuDF DataFrame or CuPy backed Dask Array
            Dense matrix (floats or doubles) of shape (n_samples, n_features).

        """

        self.solver.fit(X, y)

        return self

    def predict(self, X, delayed=True):
        """
        Predicts the y for X.

        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array
            Dense matrix (floats or doubles) of shape (n_samples, n_features).

        delayed : bool (default = True)
            Whether to do a lazy prediction (and return Delayed objects) or an
            eagerly executed one.


        Returns
        -------
        y : Dask cuDF DataFrame or CuPy backed Dask Array
            Dense matrix (floats or doubles) of shape (n_samples, n_features).

        """

        return self.solver.predict(X, delayed=delayed)
