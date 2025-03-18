# Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

from cuml.dask.common.base import mnmg_import
from cuml.dask.linear_model import LinearRegression
from raft_dask.common.comms import get_raft_comm_state
from dask.distributed import get_worker

from cuml.common.sparse_utils import is_sparse, has_scipy
from cuml.dask.common.input_utils import concatenate
from cuml.internals.safe_imports import cpu_only_import
from cuml.internals.safe_imports import gpu_only_import

cp = gpu_only_import("cupy")
cupyx = gpu_only_import("cupyx")
np = cpu_only_import("numpy")
scipy = cpu_only_import("scipy")


class LogisticRegression(LinearRegression):
    """
    LogisticRegression is a linear model that is used to model probability of
    occurrence of certain events, for example probability of success or fail of
    an event.

    cuML's dask Logistic Regression (multi-node multi-gpu) expects dask cuDF
    DataFrame and provides an algorithms, L-BFGS, to fit the logistic model. It
    currently supports single class, l2 regularization, and sigmoid loss.

    Note that, just like in Scikit-learn, the bias will not be regularized.

    Examples
    --------

    .. code-block:: python

        >>> from dask_cuda import LocalCUDACluster
        >>> from dask.distributed import Client
        >>> import dask_cudf
        >>> import cudf
        >>> import numpy as np
        >>> from cuml.dask.linear_model import LogisticRegression

        >>> cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES="0,1")
        >>> client = Client(cluster)

        >>> X = cudf.DataFrame()
        >>> X['col1'] = np.array([1,1,2,2], dtype = np.float32)
        >>> X['col2'] = np.array([1,2,2,3], dtype = np.float32)
        >>> y = cudf.Series(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32))

        >>> X_ddf = dask_cudf.from_cudf(X, npartitions=2)
        >>> y_ddf = dask_cudf.from_cudf(y, npartitions=2)

        >>> reg = LogisticRegression()
        >>> reg.fit(X_ddf, y_ddf)
        LogisticRegression()

        >>> print(reg.coef_)
                 0         1
        0  0.69861  0.570058
        >>> print(reg.intercept_)
        0   -2.188068
        dtype: float32

        >>> X_new = cudf.DataFrame()
        >>> X_new['col1'] = np.array([1,5], dtype = np.float32)
        >>> X_new['col2'] = np.array([2,5], dtype = np.float32)
        >>> X_new_ddf = dask_cudf.from_cudf(X_new, npartitions=2)
        >>> preds = reg.predict(X_new_ddf)

        >>> print(preds.compute())
        0    0.0
        1    1.0
        dtype: float32

    Parameters
    ----------
    tol : float (default = 1e-4)
        Tolerance for stopping criteria.
        The exact stopping conditions depend on the L-BFGS solver.
        Check the solver's documentation for more details:

          * :class:`Quasi-Newton (L-BFGS)<cuml.QN>`

    C : float (default = 1.0)
        Inverse of regularization strength; must be a positive float.
    fit_intercept : boolean (default = True)
        If True, the model tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    max_iter : int (default = 1000)
        Maximum number of iterations taken for the solvers to converge.
    linesearch_max_iter : int (default = 50)
        Max number of linesearch iterations per outer iteration used in the
        solver.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    coef_: dev array, dim (n_classes, n_features) or (n_classes, n_features+1)
        The estimated coefficients for the linear regression model.
    intercept_: device array (n_classes, 1)
        The independent term. If `fit_intercept` is False, will be 0.

    Notes
    -----
    cuML's LogisticRegression uses a different solver that the equivalent
    Scikit-learn, except when there is no penalty and `solver=lbfgs` is
    used in Scikit-learn. This can cause (smaller) differences in the
    coefficients and predictions of the model, similar to
    using different solvers in Scikit-learn.

    For additional information, see `Scikit-learn's LogisticRegression
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_.
    """

    def __init__(self, *, standardization=False, **kwargs):
        super().__init__(**kwargs)
        self.kwargs["standardization"] = standardization

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
            model_func=LogisticRegression._create_model, data=(X, y)
        )

        self._set_internal_model(models[0])

        return self

    @staticmethod
    @mnmg_import
    def _create_model(sessionId, datatype, **kwargs):
        from cuml.linear_model.logistic_regression_mg import (
            LogisticRegressionMG,
        )

        handle = get_raft_comm_state(sessionId, get_worker())["handle"]
        return LogisticRegressionMG(handle=handle, **kwargs)

    @staticmethod
    def _func_fit(f, data, n_rows, n_cols, partsToSizes, rank):
        if is_sparse(data[0][0]) is False:
            inp_X = concatenate([X for X, _ in data])

        elif has_scipy() and scipy.sparse.isspmatrix(data[0][0]):
            inp_X = scipy.sparse.vstack([X for X, _ in data])

        elif cupyx.scipy.sparse.isspmatrix(data[0][0]):
            total_nnz = sum([X.nnz for X, _ in data])
            if total_nnz > np.iinfo(np.int32).max:
                raise ValueError(
                    f"please use scipy csr_matrix because cupyx uses int32 index dtype that does not support {total_nnz} non-zero values of a partition"
                )
            inp_X = cupyx.scipy.sparse.vstack([X for X, _ in data])

        else:
            raise ValueError(
                "input matrix must be dense, scipy sparse, or cupyx sparse"
            )

        inp_y = concatenate([y for _, y in data])
        n_ranks = max([p[0] for p in partsToSizes]) + 1
        aggregated_partsToSizes = [[i, 0] for i in range(n_ranks)]
        for p in partsToSizes:
            aggregated_partsToSizes[p[0]][1] += p[1]

        ret_status = f.fit(
            [(inp_X, inp_y)], n_rows, n_cols, aggregated_partsToSizes, rank
        )

        if len(f.classes_) == 1:
            raise ValueError(
                f"This solver needs samples of at least 2 classes in the data, but the data contains only one class: {f.classes_[0]}"
            )

        return ret_status
