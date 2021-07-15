#
# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

# distutils: language = c++

from cuml.solvers import CD
from cuml.common.base import Base
from cuml.common.mixins import RegressorMixin
from cuml.common.doc_utils import generate_docstring
from cuml.common.mixins import FMajorInputTagMixin
from cuml.linear_model.base import LinearPredictMixin


class Lasso(Base,
            LinearPredictMixin,
            RegressorMixin,
            FMajorInputTagMixin):

    """
    Lasso extends LinearRegression by providing L1 regularization on the
    coefficients when predicting response y with a linear combination of the
    predictors in X. It can zero some of the coefficients for feature
    selection and improves the conditioning of the problem.

    cuML's Lasso can take array-like objects, either in host as
    NumPy arrays or in device (as Numba or `__cuda_array_interface__`
    compliant), in addition to cuDF objects. It uses coordinate descent to fit
    a linear model.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        import cudf
        from cuml.linear_model import Lasso

        ls = Lasso(alpha = 0.1)

        X = cudf.DataFrame()
        X['col1'] = np.array([0, 1, 2], dtype = np.float32)
        X['col2'] = np.array([0, 1, 2], dtype = np.float32)

        y = cudf.Series( np.array([0.0, 1.0, 2.0], dtype = np.float32) )

        result_lasso = ls.fit(X, y)
        print("Coefficients:")
        print(result_lasso.coef_)
        print("intercept:")
        print(result_lasso.intercept_)

        X_new = cudf.DataFrame()
        X_new['col1'] = np.array([3,2], dtype = np.float32)
        X_new['col2'] = np.array([5,5], dtype = np.float32)
        preds = result_lasso.predict(X_new)

        print(preds)

    Output:

    .. code-block:: python

        Coefficients:

                    0 0.85
                    1 0.0

        Intercept:
                    0.149999

        Preds:

                    0 2.7
                    1 1.85

    Parameters
    -----------
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
    max_iter : int
        The maximum number of iterations
    tol : float (default = 1e-3)
        The tolerance for the optimization: if the updates are smaller than
        tol, the optimization code checks the dual gap for optimality and
        continues until it is smaller than tol.
    selection : {'cyclic', 'random'} (default='cyclic')
        If set to ‘random’, a random coefficient is updated every iteration
        rather than looping over features sequentially by default.
        This (setting to ‘random’) often leads to significantly faster
        convergence especially when tol is higher than 1e-4.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_settings.output_type`.
        See :ref:`output-data-type-configuration` for more info.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    Attributes
    -----------
    coef_ : array, shape (n_features)
        The estimated coefficients for the linear regression model.
    intercept_ : array
        The independent term. If `fit_intercept` is False, will be 0.

    Notes
    -----
    For additional docs, see `scikitlearn's Lasso
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>`_.
    """

    def __init__(self, *, alpha=1.0, fit_intercept=True, normalize=False,
                 max_iter=1000, tol=1e-3, selection='cyclic', handle=None,
                 output_type=None, verbose=False):

        # Hard-code verbosity as CoordinateDescent does not have verbosity
        super().__init__(handle=handle,
                         verbose=verbose,
                         output_type=output_type)

        self._check_alpha(alpha)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
        self.solver_model = None
        if selection in ['cyclic', 'random']:
            self.selection = selection
        else:
            msg = "selection {!r} is not supported"
            raise TypeError(msg.format(selection))

        self.intercept_value = 0.0

        shuffle = False
        if self.selection == 'random':
            shuffle = True

        self.solver_model = CD(fit_intercept=self.fit_intercept,
                               normalize=self.normalize, alpha=self.alpha,
                               l1_ratio=1.0, shuffle=shuffle,
                               max_iter=self.max_iter, handle=self.handle)

    def _check_alpha(self, alpha):
        if alpha <= 0.0:
            msg = "alpha value has to be positive"
            raise ValueError(msg.format(alpha))

    @generate_docstring()
    def fit(self, X, y, convert_dtype=True) -> "Lasso":
        """
        Fit the model with X and y.

        """
        self.solver_model.fit(X, y, convert_dtype=convert_dtype)

        return self

    def get_param_names(self):
        return super().get_param_names() + [
            "alpha",
            "fit_intercept",
            "normalize",
            "max_iter",
            "tol",
            "selection",
        ]
