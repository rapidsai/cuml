#
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

from inspect import signature

from cuml.solvers import CD, QN
from cuml.internals.base import UniversalBase
from cuml.internals.mixins import RegressorMixin, FMajorInputTagMixin
from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.internals.logger import warn
from cuml.linear_model.base import LinearPredictMixin
from cuml.internals.api_decorators import device_interop_preparation
from cuml.internals.api_decorators import enable_device_interop


class ElasticNet(UniversalBase,
                 LinearPredictMixin,
                 RegressorMixin,
                 FMajorInputTagMixin):

    """
    ElasticNet extends LinearRegression with combined L1 and L2 regularizations
    on the coefficients when predicting response y with a linear combination of
    the predictors in X. It can reduce the variance of the predictors, force
    some coefficients to be small, and improves the conditioning of the
    problem.

    cuML's ElasticNet an array-like object or cuDF DataFrame, uses coordinate
    descent to fit a linear model.

    Examples
    --------

    .. code-block:: python

        >>> import cupy as cp
        >>> import cudf
        >>> from cuml.linear_model import ElasticNet
        >>> enet = ElasticNet(alpha = 0.1, l1_ratio=0.5)
        >>> X = cudf.DataFrame()
        >>> X['col1'] = cp.array([0, 1, 2], dtype = cp.float32)
        >>> X['col2'] = cp.array([0, 1, 2], dtype = cp.float32)
        >>> y = cudf.Series(cp.array([0.0, 1.0, 2.0], dtype = cp.float32) )
        >>> result_enet = enet.fit(X, y)
        >>> print(result_enet.coef_)
        0    0.448...
        1    0.443...
        dtype: float32
        >>> print(result_enet.intercept_)
        0.1082506...
        >>> X_new = cudf.DataFrame()
        >>> X_new['col1'] = cp.array([3,2], dtype = cp.float32)
        >>> X_new['col2'] = cp.array([5,5], dtype = cp.float32)
        >>> preds = result_enet.predict(X_new)
        >>> print(preds)
        0    3.670...
        1    3.221...
        dtype: float32

    Parameters
    ----------
    alpha : float (default = 1.0)
        Constant that multiplies the L1 term.
        alpha = 0 is equivalent to an ordinary least square, solved by the
        LinearRegression object.
        For numerical reasons, using alpha = 0 with the Lasso object is not
        advised.
        Given this, you should use the LinearRegression object.
    l1_ratio : float (default = 0.5)
        The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is
        an L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
    fit_intercept : boolean (default = True)
        If True, Lasso tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    normalize : boolean (default = False)
        If True, the predictors in X will be normalized by dividing by the
        column-wise standard deviation.
        If False, no scaling will be done.
        Note: this is in contrast to sklearn's deprecated `normalize` flag,
        which divides by the column-wise L2 norm; but this is the same as if
        using sklearn's StandardScaler.
    max_iter : int (default = 1000)
        The maximum number of iterations
    tol : float (default = 1e-3)
        The tolerance for the optimization: if the updates are smaller than
        tol, the optimization code checks the dual gap for optimality and
        continues until it is smaller than tol.
    solver : {'cd', 'qn'} (default='cd')
        Choose an algorithm:

          * 'cd' - coordinate descent
          * 'qn' - quasi-newton

        You may find the alternative 'qn' algorithm is faster when the number
        of features is sufficiently large, but the sample size is small.
    selection : {'cyclic', 'random'} (default='cyclic')
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default.
        This (setting to 'random') often leads to significantly faster
        convergence especially when tol is higher than 1e-4.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    Attributes
    ----------
    coef_ : array, shape (n_features)
        The estimated coefficients for the linear regression model.
    intercept_ : array
        The independent term. If `fit_intercept` is False, will be 0.

    Notes
    -----
    For additional docs, see `scikitlearn's ElasticNet
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html>`_.
    """

    _cpu_estimator_import_path = 'sklearn.linear_model.ElasticNet'
    coef_ = CumlArrayDescriptor(order='F')

    @device_interop_preparation
    def __init__(self, *, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
                 normalize=False, max_iter=1000, tol=1e-3,
                 solver='cd', selection='cyclic',
                 handle=None, output_type=None, verbose=False):
        """
        Initializes the elastic-net regression class.

        Parameters
        ----------
        alpha : float or double.
        l1_ratio : float or double.
        fit_intercept: boolean.
        normalize: boolean.
        max_iter: int
        tol: float or double.
        solver: str, 'cd' or 'qn'
        selection : str, 'cyclic', or 'random'

        For additional docs, see `scikitlearn's ElasticNet
        <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html>`_.
        """

        # Hard-code verbosity as CoordinateDescent does not have verbosity
        super().__init__(handle=handle,
                         verbose=verbose,
                         output_type=output_type)

        self._check_alpha(alpha)
        self._check_l1_ratio(l1_ratio)

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.solver = solver
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

        if solver == 'qn':
            pams = signature(self.__init__).parameters
            if (pams['selection'].default != selection):
                warn("Parameter 'selection' has no effect "
                     "when 'qn' solver is used.")
            if (pams['normalize'].default != normalize):
                warn("Parameter 'normalize' has no effect "
                     "when 'qn' solver is used.")

            self.solver_model = QN(
                fit_intercept=self.fit_intercept,
                l1_strength=self.alpha * self.l1_ratio,
                l2_strength=self.alpha * (1.0 - self.l1_ratio),
                max_iter=self.max_iter, handle=self.handle,
                loss='l2', tol=self.tol, penalty_normalized=False,
                verbose=self.verbose)
        elif solver == 'cd':
            self.solver_model = CD(
                fit_intercept=self.fit_intercept,
                normalize=self.normalize, alpha=self.alpha,
                l1_ratio=self.l1_ratio, shuffle=shuffle,
                max_iter=self.max_iter, handle=self.handle,
                tol=self.tol)
        else:
            raise TypeError(f"solver {solver} is not supported")

    def _check_alpha(self, alpha):
        if alpha <= 0.0:
            msg = "alpha value has to be positive"
            raise ValueError(msg.format(alpha))

    def _check_l1_ratio(self, l1_ratio):
        if l1_ratio < 0.0 or l1_ratio > 1.0:
            msg = "l1_ratio value has to be between 0.0 and 1.0"
            raise ValueError(msg.format(l1_ratio))

    @generate_docstring()
    @enable_device_interop
    def fit(self, X, y, convert_dtype=True,
            sample_weight=None) -> "ElasticNet":
        """
        Fit the model with X and y.

        """
        self.n_features_in_ = X.shape[1] if X.ndim == 2 else 1
        if hasattr(X, 'index'):
            self.feature_names_in_ = X.index

        self.solver_model.fit(X, y, convert_dtype=convert_dtype,
                              sample_weight=sample_weight)
        if isinstance(self.solver_model, QN):
            coefs = self.solver_model.coef_
            self.coef_ = CumlArray(
                data=coefs,
                index=coefs._index,
                dtype=coefs.dtype,
                order=coefs.order,
                shape=(coefs.shape[1],)
            )
            self.intercept_ = self.solver_model.intercept_.item()

        return self

    def set_params(self, **params):
        super().set_params(**params)
        if 'selection' in params:
            params.pop('selection')
            params['shuffle'] = self.selection == 'random'
        self.solver_model.set_params(**params)
        return self

    def get_param_names(self):
        return super().get_param_names() + [
            "alpha",
            "l1_ratio",
            "fit_intercept",
            "normalize",
            "max_iter",
            "tol",
            "solver",
            "selection",
        ]

    def get_attr_names(self):
        return ['intercept_', 'coef_', 'n_features_in_', 'feature_names_in_']
