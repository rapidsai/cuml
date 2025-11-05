#
# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.interop import (
    InteropMixin,
    UnsupportedOnGPU,
    to_cpu,
    to_gpu,
)
from cuml.internals.mixins import FMajorInputTagMixin, RegressorMixin
from cuml.linear_model.base import (
    LinearPredictMixin,
    check_deprecated_normalize,
)
from cuml.solvers import QN
from cuml.solvers.cd import fit_coordinate_descent


class ElasticNet(
    Base, InteropMixin, LinearPredictMixin, RegressorMixin, FMajorInputTagMixin
):
    """
    Linear regression with combined L1 and L2 priors as regularizer.

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the L1 term.
        alpha = 0 is equivalent to an ordinary least square, solved by the
        LinearRegression object.
        For numerical reasons, using alpha = 0 with the Lasso object is not
        advised.
        Given this, you should use the LinearRegression object.
    l1_ratio : float, default=0.5
        The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is
        an L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
    fit_intercept : boolean, default=True
        If True, Lasso tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    max_iter : int, default=1000
        The maximum number of iterations
    tol : float, default=1e-3
        The tolerance for the optimization: if the updates are smaller than
        tol, the optimization code checks the dual gap for optimality and
        continues until it is smaller than tol.
    solver : {'cd', 'qn'}, default='cd'
        Choose an algorithm:

          * 'cd' - coordinate descent
          * 'qn' - quasi-newton

        You may find the alternative 'qn' algorithm is faster when the number
        of features is sufficiently large but the sample size is small.
    selection : {'cyclic', 'random'}, default='cyclic'
        How selections are made when `solver="cd"`. If set to 'random', a
        random coefficient is updated every iteration rather than looping over
        features sequentially by default. This (setting to 'random') often
        leads to significantly faster convergence especially when tol is higher
        than 1e-4.
    normalize : boolean, default=False

        .. deprecated:: 25.12
            ``normalize`` is deprecated and will be removed in 26.02. When
            needed, please use a ``StandardScaler`` to normalize your data
            before passing to ``fit``.

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
    intercept_ : float
        The independent term, will be 0 if `fit_intercept` is False.

    Notes
    -----
    For additional docs, see `scikitlearn's ElasticNet
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html>`_.

    Examples
    --------
    >>> import cupy as cp
    >>> import cudf
    >>> from cuml.linear_model import ElasticNet
    >>> enet = ElasticNet(alpha = 0.1, l1_ratio=0.5, solver='qn')
    >>> X = cudf.DataFrame()
    >>> X['col1'] = cp.array([0, 1, 2], dtype = cp.float32)
    >>> X['col2'] = cp.array([0, 1, 2], dtype = cp.float32)
    >>> y = cudf.Series(cp.array([0.0, 1.0, 2.0], dtype = cp.float32) )
    >>> result_enet = enet.fit(X, y)
    >>> print(result_enet.coef_)
    0    0.445...
    1    0.445...
    dtype: float32
    >>> print(result_enet.intercept_)
    0.108433...
    >>> X_new = cudf.DataFrame()
    >>> X_new['col1'] = cp.array([3,2], dtype = cp.float32)
    >>> X_new['col2'] = cp.array([5,5], dtype = cp.float32)
    >>> preds = result_enet.predict(X_new)
    >>> print(preds)
    0    3.674...
    1    3.228...
    dtype: float32
    """

    coef_ = CumlArrayDescriptor(order="F")

    _cpu_class_path = "sklearn.linear_model.ElasticNet"

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "alpha",
            "l1_ratio",
            "fit_intercept",
            "max_iter",
            "tol",
            "solver",
            "selection",
            "normalize",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.positive:
            raise UnsupportedOnGPU("`positive=True` is not supported")

        if model.warm_start:
            raise UnsupportedOnGPU("`warm_start=True` is not supported")

        if model.precompute is not False:
            raise UnsupportedOnGPU("`precompute` is not supported")

        # We use different algorithms than sklearn, adjust tolerance by a
        # factor empirically determined to be ~equivalent.
        tol = 10 * model.tol

        return {
            "alpha": model.alpha,
            "l1_ratio": model.l1_ratio,
            "fit_intercept": model.fit_intercept,
            "tol": tol,
            "max_iter": model.max_iter,
            "selection": model.selection,
        }

    def _params_to_cpu(self):
        # We use different algorithms than sklearn, adjust tolerance by a
        # factor empirically determined to be ~equivalent.
        tol = 0.1 * self.tol

        return {
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "fit_intercept": self.fit_intercept,
            "tol": tol,
            "max_iter": self.max_iter,
            "selection": self.selection,
        }

    def _attrs_from_cpu(self, model):
        return {
            "intercept_": to_gpu(model.intercept_, order="F"),
            "coef_": to_gpu(model.coef_, order="F"),
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "intercept_": to_cpu(self.intercept_),
            "coef_": to_cpu(self.coef_),
            **super()._attrs_to_cpu(model),
        }

    def __init__(
        self,
        alpha=1.0,
        *,
        l1_ratio=0.5,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        solver="cd",
        selection="cyclic",
        normalize=False,
        handle=None,
        output_type=None,
        verbose=False,
    ):
        super().__init__(
            handle=handle, verbose=verbose, output_type=output_type
        )

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.selection = selection
        self.normalize = normalize

    @generate_docstring()
    def fit(
        self, X, y, sample_weight=None, *, convert_dtype=True
    ) -> "ElasticNet":
        """
        Fit the model with X and y.

        """
        check_deprecated_normalize(self)

        if self.alpha < 0.0:
            raise ValueError(f"Expected alpha >= 0, got {self.alpha}")
        if self.selection not in ["cyclic", "random"]:
            raise ValueError(f"selection {self.selection!r} is not supported")
        if self.l1_ratio < 0.0 or self.l1_ratio > 1.0:
            raise ValueError(
                f"Expected 0.0 <= l1_ratio <= 1.0, got {self.l1_ratio}"
            )

        if self.solver == "qn":
            if self.normalize:
                raise ValueError(
                    "`normalize=True` is not supported with `solver='qn'"
                )

            solver = QN(
                handle=self.handle,
                verbose=self.verbose,
                output_type=self.output_type,
                fit_intercept=self.fit_intercept,
                l1_strength=self.alpha * self.l1_ratio,
                l2_strength=self.alpha * (1.0 - self.l1_ratio),
                loss="l2",
                penalty_normalized=False,
                max_iter=self.max_iter,
                tol=self.tol,
            ).fit(
                X, y, sample_weight=sample_weight, convert_dtype=convert_dtype
            )

            coef = CumlArray(data=solver.coef_.to_output("cupy").flatten())
            intercept = solver.intercept_.item()
        elif self.solver == "cd":
            coef, intercept = fit_coordinate_descent(
                X,
                y,
                sample_weight=sample_weight,
                convert_dtype=convert_dtype,
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                l1_ratio=self.l1_ratio,
                normalize=self.normalize,
                shuffle=self.selection == "random",
                max_iter=self.max_iter,
                tol=self.tol,
                handle=self.handle,
            )
        else:
            raise ValueError(f"solver {self.solver} is not supported")

        self.coef_ = coef
        self.intercept_ = intercept

        return self
