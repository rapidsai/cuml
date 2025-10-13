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
import numpy as np

import cuml.internals.nvtx as nvtx
from cuml.common import input_to_cuml_array
from cuml.common.doc_utils import generate_docstring, insert_into_docstring
from cuml.ensemble.randomforest_common import BaseRandomForestModel
from cuml.internals.array import CumlArray
from cuml.internals.mixins import RegressorMixin
from cuml.metrics import r2_score


class RandomForestRegressor(BaseRandomForestModel, RegressorMixin):
    """
    Implements a Random Forest regressor model which fits multiple decision
    trees in an ensemble.

    .. note:: Note that the underlying algorithm for tree node splits differs
      from that used in scikit-learn. By default, the cuML Random Forest uses a
      quantile-based algorithm to determine splits, rather than an exact
      count. You can tune the size of the quantiles with the `n_bins` parameter

    .. note:: You can export cuML Random Forest models and run predictions
      with them on machines without an NVIDIA GPUs. See
      https://docs.rapids.ai/api/cuml/nightly/pickling_cuml_models.html
      for more details.

    Examples
    --------

    .. code-block:: python

        >>> import cupy as cp
        >>> from cuml.ensemble import RandomForestRegressor as curfr
        >>> X = cp.asarray([[0,10],[0,20],[0,30],[0,40]], dtype=cp.float32)
        >>> y = cp.asarray([0.0,1.0,2.0,3.0], dtype=cp.float32)
        >>> cuml_model = curfr(max_features=1.0, n_bins=128,
        ...                    min_samples_leaf=1,
        ...                    min_samples_split=2,
        ...                    n_estimators=40)
        >>> cuml_model.fit(X,y)
        RandomForestRegressor()
        >>> cuml_score = cuml_model.score(X,y)
        >>> print("R2 score of cuml : ", cuml_score) # doctest: +SKIP
        R2 score of cuml :  0.9076250195503235

    Parameters
    ----------
    n_estimators : int (default = 100)
        Number of trees in the forest. (Default changed to 100 in cuML 0.11)
    split_criterion : str or int (default = ``'mse'``)
        The criterion used to split nodes.\n
         * ``'mse'`` or ``2`` for mean squared error
         * ``'poisson'`` or ``4`` for poisson half deviance
         * ``'gamma'`` or ``5`` for gamma half deviance
         * ``'inverse_gaussian'`` or ``6`` for inverse gaussian deviance
    bootstrap : boolean (default = True)
        Control bootstrapping.\n
            * If ``True``, each tree in the forest is built
              on a bootstrapped sample with replacement.
            * If ``False``, the whole dataset is used to build each tree.
    max_samples : float (default = 1.0)
        Ratio of dataset rows used while fitting each tree.
    max_depth : int (default = 16)
        Maximum tree depth. Must be greater than 0.
        Unlimited depth (i.e, until leaves are pure)
        is not supported.\n
        .. note:: This default differs from scikit-learn's
          random forest, which defaults to unlimited depth.
    max_leaves : int (default = -1)
        Maximum leaf nodes per tree. Soft constraint. Unlimited,
        If ``-1``.
    max_features : {'sqrt', 'log2', None}, int or float (default = 1.0)
        The number of features to consider per node split:

        * If an int then ``max_features`` is the absolute count of features to be used.
        * If a float then ``max_features`` is used as a fraction.
        * If ``'sqrt'`` then ``max_features=1/sqrt(n_features)``.
        * If ``'log2'`` then ``max_features=log2(n_features)/n_features``.
        * If ``None`` then ``max_features=n_features``

        .. versionchanged:: 24.06
          The default of `max_features` changed from `"auto"` to 1.0.

    n_bins : int (default = 128)
        Maximum number of bins used by the split algorithm per feature.
        For large problems, particularly those with highly-skewed input data,
        increasing the number of bins may improve accuracy.
    n_streams : int (default = 4 )
        Number of parallel streams used for forest building
    min_samples_leaf : int or float (default = 1)
        The minimum number of samples (rows) in each leaf node.\n
         * If type ``int``, then ``min_samples_leaf`` represents the minimum
           number.\n
         * If ``float``, then ``min_samples_leaf`` represents a fraction and
           ``ceil(min_samples_leaf * n_rows)`` is the minimum number of
           samples for each leaf node.
    min_samples_split : int or float (default = 2)
        The minimum number of samples required to split an internal
        node.\n
         * If type ``int``, then min_samples_split represents the minimum
           number.
         * If type ``float``, then ``min_samples_split`` represents a fraction
           and ``max(2, ceil(min_samples_split * n_rows))`` is the minimum
           number of samples for each split.
    min_impurity_decrease : float (default = 0.0)
        The minimum decrease in impurity required for node to be split
    max_batch_size : int (default = 4096)
        Maximum number of nodes that can be processed in a given batch.
    random_state : int (default = None)
        Seed for the random number generator. Unseeded by default.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Notes
    -----
    For additional docs, see `scikitlearn's RandomForestRegressor
    <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`_.
    """

    _cpu_class_path = "sklearn.ensemble.RandomForestRegressor"

    def __init__(
        self,
        *,
        split_criterion="mse",
        max_features=1.0,
        handle=None,
        verbose=False,
        output_type=None,
        **kwargs,
    ):
        super().__init__(
            split_criterion=split_criterion,
            max_features=max_features,
            handle=handle,
            verbose=verbose,
            output_type=output_type,
            **kwargs,
        )

    @nvtx.annotate(
        message="fit RF-Regressor @randomforestregressor.pyx",
        domain="cuml_python",
    )
    @generate_docstring()
    def fit(self, X, y, *, convert_dtype=True) -> "RandomForestRegressor":
        """
        Perform Random Forest Regression on the input data

        """
        X_m = input_to_cuml_array(
            X,
            convert_to_dtype=(np.float32 if convert_dtype else None),
            check_dtype=[np.float32, np.float64],
            order="F",
        ).array

        y_m = input_to_cuml_array(
            y,
            convert_to_dtype=(X_m.dtype if convert_dtype else None),
            check_dtype=X_m.dtype,
            check_rows=X_m.shape[0],
            check_cols=1,
        ).array
        return self._fit_forest(X_m, y_m)

    @nvtx.annotate(
        message="predict RF-Regressor @randomforestclassifier.pyx",
        domain="cuml_python",
    )
    @insert_into_docstring(
        parameters=[("dense", "(n_samples, n_features)")],
        return_values=[("dense", "(n_samples, 1)")],
    )
    def predict(
        self,
        X,
        *,
        convert_dtype=True,
        layout="depth_first",
        default_chunk_size=None,
        align_bytes=None,
    ) -> CumlArray:
        """
        Predicts the values for X.

        Parameters
        ----------
        X : {}
        convert_dtype : bool, optional (default = True)
            When set to True, the predict method will, when necessary, convert
            the input to the data type which was used to train the model. This
            will increase memory used for the method.
        layout : string (default = 'depth_first')
            Specifies the in-memory layout of nodes in FIL forests. Options:
            'depth_first', 'layered', 'breadth_first'.
        default_chunk_size : int, optional (default = None)
            Determines how batches are further subdivided for parallel processing.
            The optimal value depends on hardware, model, and batch size.
            If None, will be automatically determined.
        align_bytes : int, optional (default = None)
            If specified, trees will be padded such that their in-memory size is
            a multiple of this value. This can improve performance by guaranteeing
            that memory reads from trees begin on a cache line boundary.
            Typical values are 0 or 128.

        Returns
        -------
        y : {}
        """
        fil = self._get_inference_fil_model(
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
        )
        preds = fil.predict(X)

        # Reshape to 1D array if the output would be (n, 1) to match
        # the output shape behavior of scikit-learn.
        if len(preds.shape) == 2 and preds.shape[1] == 1:
            preds = CumlArray(preds.to_output("cupy").reshape(-1))
        return preds

    @nvtx.annotate(
        message="score RF-Regressor @randomforestclassifier.pyx",
        domain="cuml_python",
    )
    @insert_into_docstring(
        parameters=[
            ("dense", "(n_samples, n_features)"),
            ("dense", "(n_samples, 1)"),
        ]
    )
    def score(
        self,
        X,
        y,
        *,
        convert_dtype=True,
        layout="depth_first",
        default_chunk_size=None,
        align_bytes=None,
    ):
        """
        Calculates the r2 score of the model on test data.

        Parameters
        ----------
        X : {}
        y : {}
        convert_dtype : bool (default = True)
            When True, automatically convert the input to the data type used
            to train the model. This may increase memory usage.
        layout : string (default = 'depth_first')
            Specifies the in-memory layout of nodes in FIL forests. Options:
            'depth_first', 'layered', 'breadth_first'.
        default_chunk_size : int, optional (default = None)
            Determines how batches are further subdivided for parallel processing.
            The optimal value depends on hardware, model, and batch size.
            If None, will be automatically determined.
        align_bytes : int, optional (default = None)
            If specified, trees will be padded such that their in-memory size is
            a multiple of this value. This can improve performance by guaranteeing
            that memory reads from trees begin on a cache line boundary.
            Typical values are 0 or 128.

        Returns
        -------
        r2_score : float
        """
        y_pred = self.predict(
            X,
            convert_dtype=convert_dtype,
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
        )
        return r2_score(y, y_pred)
