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

import cupy as cp
import numpy as np

import cuml.internals
import cuml.internals.nvtx as nvtx
from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring, insert_into_docstring
from cuml.ensemble.randomforest_common import BaseRandomForestModel
from cuml.internals.array import CumlArray
from cuml.internals.interop import UnsupportedOnGPU, to_cpu, to_gpu
from cuml.internals.mixins import ClassifierMixin
from cuml.metrics import accuracy_score
from cuml.prims.label.classlabels import invert_labels, make_monotonic


class RandomForestClassifier(BaseRandomForestModel, ClassifierMixin):
    """
    Implements a Random Forest classifier model which fits multiple decision
    tree classifiers in an ensemble.

    .. note:: Note that the underlying algorithm for tree node splits differs
      from that used in scikit-learn. By default, the cuML Random Forest uses a
      quantile-based algorithm to determine splits, rather than an exact
      count. You can tune the size of the quantiles with the `n_bins`
      parameter.

    .. note:: You can export cuML Random Forest models and run predictions
      with them on machines without an NVIDIA GPUs. See
      https://docs.rapids.ai/api/cuml/nightly/pickling_cuml_models.html
      for more details.

    Examples
    --------

    .. code-block:: python

        >>> import cupy as cp
        >>> from cuml.ensemble import RandomForestClassifier as cuRFC

        >>> X = cp.random.normal(size=(10,4)).astype(cp.float32)
        >>> y = cp.asarray([0,1]*5, dtype=cp.int32)

        >>> cuml_model = cuRFC(max_features=1.0,
        ...                    n_bins=8,
        ...                    n_estimators=40)
        >>> cuml_model.fit(X,y)
        RandomForestClassifier()
        >>> cuml_predict = cuml_model.predict(X)

        >>> print("Predicted labels : ", cuml_predict)
        Predicted labels :  [0. 1. 0. 1. 0. 1. 0. 1. 0. 1.]

    Parameters
    ----------
    n_estimators : int (default = 100)
        Number of trees in the forest. (Default changed to 100 i 0.11)
    split_criterion : str or int (default = ``'gini'``)
        The criterion used to split nodes.\n
         * ``'gini'`` or ``0`` for gini impurity
         * ``'entropy'`` or ``1`` for information gain (entropy)
    bootstrap : boolean (default = True)
        Control bootstrapping.\n
            * If ``True``, each tree in the forest is built on a bootstrapped
              sample with replacement.
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
    max_features : {'sqrt', 'log2', None}, int or float (default = 'sqrt')
        The number of features to consider per node split:

        * If an int then ``max_features`` is the absolute count of features to be used.
        * If a float then ``max_features`` is used as a fraction.
        * If ``'sqrt'`` then ``max_features=1/sqrt(n_features)``.
        * If ``'log2'`` then ``max_features=log2(n_features)/n_features``.
        * If ``None`` then ``max_features=n_features``

        .. versionchanged:: 24.06
          The default of `max_features` changed from `"auto"` to `"sqrt"`.

    n_bins : int (default = 128)
        Maximum number of bins used by the split algorithm per feature.
        For large problems, particularly those with highly-skewed input data,
        increasing the number of bins may improve accuracy.
    n_streams : int (default = 4)
        Number of parallel streams used for forest building.
    min_samples_leaf : int or float (default = 1)
        The minimum number of samples (rows) in each leaf node.\n
         * If type ``int``, then ``min_samples_leaf`` represents the minimum
           number.
         * If ``float``, then ``min_samples_leaf`` represents a fraction and
           ``ceil(min_samples_leaf * n_rows)`` is the minimum number of
           samples for each leaf node.
    min_samples_split : int or float (default = 2)
        The minimum number of samples required to split an internal node.\n
         * If type ``int``, then min_samples_split represents the minimum
           number.
         * If type ``float``, then ``min_samples_split`` represents a fraction
           and ``max(2, ceil(min_samples_split * n_rows))`` is the minimum
           number of samples for each split.
    min_impurity_decrease : float (default = 0.0)
        Minimum decrease in impurity required for
        node to be split.
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
    While training the model for multi class classification problems, using
    deep trees or `max_features=1.0` provides better performance.

    For additional docs, see `scikitlearn's RandomForestClassifier
    <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_.
    """

    classes_ = CumlArrayDescriptor()

    _cpu_class_path = "sklearn.ensemble.RandomForestClassifier"

    @classmethod
    def _params_from_cpu(cls, model):
        if model.class_weight is not None:
            raise UnsupportedOnGPU("`class_weight` is not supported")
        return super()._params_from_cpu(model)

    def _attrs_from_cpu(self, model):
        return {
            "classes_": to_gpu(model.classes_),
            "n_classes_": model.n_classes_,
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "classes_": to_cpu(self.classes_),
            "n_classes_": self.n_classes_,
            **super()._attrs_to_cpu(model),
        }

    def __init__(
        self,
        *,
        split_criterion="gini",
        handle=None,
        verbose=False,
        output_type=None,
        **kwargs,
    ):
        super().__init__(
            split_criterion=split_criterion,
            handle=handle,
            verbose=verbose,
            output_type=output_type,
            **kwargs,
        )

    @nvtx.annotate(
        message="fit RF-Classifier @randomforestclassifier.pyx",
        domain="cuml_python",
    )
    @generate_docstring(
        skip_parameters_heading=True,
        y="dense_intdtype",
        convert_dtype_cast="np.float32",
    )
    def fit(self, X, y, *, convert_dtype=True) -> "RandomForestClassifier":
        """
        Perform Random Forest Classification on the input data

        Parameters
        ----------
        convert_dtype : bool, optional (default = True)
            When set to True, the fit method will, when necessary, convert
            y to be of dtype int32. This will increase memory used for
            the method.
        """
        X_m = input_to_cuml_array(
            X,
            convert_to_dtype=(np.float32 if convert_dtype else None),
            check_dtype=[np.float32, np.float64],
            order="F",
        ).array
        y_m = input_to_cuml_array(
            y,
            convert_to_dtype=(np.int32 if convert_dtype else None),
            check_dtype=np.int32,
            check_rows=X_m.shape[0],
            check_cols=1,
        ).array
        self.classes_ = cp.unique(y_m)
        self.n_classes_ = len(self.classes_)
        if not (self.classes_ == cp.arange(self.n_classes_)).all():
            y_m, _ = make_monotonic(y_m)

        return self._fit_forest(X_m, y_m)

    @nvtx.annotate(
        message="predict RF-Classifier @randomforestclassifier.pyx",
        domain="cuml_python",
    )
    @insert_into_docstring(
        parameters=[("dense", "(n_samples, n_features)")],
        return_values=[("dense", "(n_samples, 1)")],
    )
    @cuml.internals.api_base_return_array(get_output_dtype=True)
    def predict(
        self,
        X,
        *,
        threshold=0.5,
        convert_dtype=True,
        layout="depth_first",
        default_chunk_size=None,
        align_bytes=None,
    ) -> CumlArray:
        """
        Predicts the labels for X.

        Parameters
        ----------
        X : {}
        threshold : float (default = 0.5)
            Threshold used for classification.
        convert_dtype : bool (default = True)
            When True, automatically convert the input to the data type used
            to train the model. This may increase memory usage.
        layout : string (default = 'depth_first')
            Forest layout for GPU inference. Options: 'depth_first', 'layered',
            'breadth_first'.
        default_chunk_size : int, optional (default = None)
            Controls batch subdivision for parallel processing. Optimal value depends
            on hardware, model and batch size. If None, determined automatically.
        align_bytes : int, optional (default = None)
            If specified, trees will be padded to this byte alignment, which can
            improve performance. Typical values are 0 or 128 on GPU.

        Returns
        -------
        y : {}
        """
        fil = self._get_inference_fil_model(
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
        )
        preds = fil.predict(X, threshold=threshold)

        if not (self.classes_ == cp.arange(self.n_classes_)).all():
            preds = preds.to_output("cupy").astype(
                self.classes_.dtype, copy=False
            )
            preds = CumlArray(invert_labels(preds, self.classes_))
        return preds

    @insert_into_docstring(
        parameters=[("dense", "(n_samples, n_features)")],
        return_values=[("dense", "(n_samples, 1)")],
    )
    def predict_proba(
        self,
        X,
        *,
        convert_dtype=True,
        layout="depth_first",
        default_chunk_size=None,
        align_bytes=None,
    ) -> CumlArray:
        """
        Predicts class probabilities for X. This function uses the GPU
        implementation of predict.

        Parameters
        ----------
        X : {}
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
            Typical values are 0 or 128 on GPU and 0 or 64 on CPU.

        Returns
        -------
        y : {}
        """
        fil = self._get_inference_fil_model(
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
        )
        return fil.predict_proba(X)

    @nvtx.annotate(
        message="score RF-Classifier @randomforestclassifier.pyx",
        domain="cuml_python",
    )
    @insert_into_docstring(
        parameters=[
            ("dense", "(n_samples, n_features)"),
            ("dense_intdtype", "(n_samples, 1)"),
        ]
    )
    def score(
        self,
        X,
        y,
        *,
        threshold=0.5,
        convert_dtype=True,
        layout="depth_first",
        default_chunk_size=None,
        align_bytes=None,
    ):
        """
        Calculates the accuracy score of the model on test data.

        Parameters
        ----------
        X : {}
        y : {}
        threshold : float (default = 0.5)
            Threshold used for classification predictions
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
            Typical values are 0 or 128 on GPU and 0 or 64 on CPU.

        Returns
        -------
        accuracy : float
           Accuracy of the model [0.0 - 1.0]
        """
        y_pred = self.predict(
            X,
            threshold=threshold,
            convert_dtype=convert_dtype,
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
        )
        return accuracy_score(y, y_pred)
