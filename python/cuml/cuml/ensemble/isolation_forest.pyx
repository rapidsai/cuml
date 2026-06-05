#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

"""
Isolation Forest implementation for GPU-accelerated anomaly detection.

This module provides a GPU-accelerated implementation of the Isolation Forest
algorithm, which is an unsupervised learning method for detecting anomalies.
"""

import math
import warnings

import cupy as cp
import nvforest
import numpy as np
import treelite

from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray
from cuml.internals.base import Base, get_handle
from cuml.internals.input_utils import input_to_cuml_array
from cuml.internals.interop import InteropMixin, UnsupportedOnGPU
from cuml.internals.mixins import CMajorInputTagMixin
from cuml.internals.treelite import safe_treelite_call
from cuml.internals.validation import check_random_seed

from libc.stddef cimport size_t
from libc.stdint cimport uint64_t, uintptr_t
from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from pylibraft.common.handle cimport handle_t

from cuml.internals.logger cimport level_enum
from cuml.internals.treelite cimport (
    TreeliteFreeModel,
    TreeliteModelHandle,
    TreeliteSerializeModelToBytes,
)


# C++ declarations from isolation_forest.hpp
cdef extern from "cuml/ensemble/isolation_forest.hpp" namespace "ML" nogil:

    cdef struct IF_params:
        int n_estimators
        int max_samples
        int max_depth
        bool bootstrap
        uint64_t seed

    # C++ struct declaration with default constructor
    cdef cppclass IsolationForestModel[T]:
        IsolationForestModel() except +  # Default constructor
        int n_features
        int n_samples_per_tree
        T c_normalization

    ctypedef IsolationForestModel[float] IsolationForestF
    ctypedef IsolationForestModel[double] IsolationForestD

    cdef void build_treelite_isolation_forest[T](
        TreeliteModelHandle* model_handle,
        const handle_t& handle,
        const IsolationForestModel[T]* forest
    ) except +

    cdef void fit(
        const handle_t& handle,
        IsolationForestF* forest,
        const float* input,
        size_t n_rows,
        int n_cols,
        const IF_params& params,
        level_enum verbosity
    ) except +

    cdef void fit(
        const handle_t& handle,
        IsolationForestD* forest,
        const double* input,
        size_t n_rows,
        int n_cols,
        const IF_params& params,
        level_enum verbosity
    ) except +

    cdef void score_samples(
        const handle_t& handle,
        const IsolationForestF* forest,
        const float* input,
        size_t n_rows,
        int n_cols,
        float* scores,
        level_enum verbosity
    ) except +

    cdef void score_samples(
        const handle_t& handle,
        const IsolationForestD* forest,
        const double* input,
        size_t n_rows,
        int n_cols,
        double* scores,
        level_enum verbosity
    ) except +

    cdef void predict(
        const handle_t& handle,
        const IsolationForestF* forest,
        const float* input,
        size_t n_rows,
        int n_cols,
        int* predictions,
        float threshold,
        level_enum verbosity
    ) except +

    cdef void predict(
        const handle_t& handle,
        const IsolationForestD* forest,
        const double* input,
        size_t n_rows,
        int n_cols,
        int* predictions,
        double threshold,
        level_enum verbosity
    ) except +


class IsolationForest(Base, InteropMixin, CMajorInputTagMixin):
    """
    GPU-accelerated Isolation Forest for anomaly detection.

    Isolation Forest is an unsupervised learning algorithm for anomaly detection
    that works by isolating anomalies rather than profiling normal data points.
    It uses the concept that anomalies are few and different, so they are easier
    to isolate.

    The algorithm builds an ensemble of isolation trees where each tree is
    constructed by randomly selecting a feature and then randomly selecting a
    split value between the minimum and maximum values of the selected feature.
    Anomalies have shorter average path lengths in the trees because they are
    easier to isolate.

    Examples
    --------

    .. code-block:: python

        >>> import cupy as cp
        >>> from cuml.ensemble import IsolationForest

        >>> # Create synthetic data with some outliers
        >>> rng = cp.random.default_rng(42)
        >>> X_inliers = rng.standard_normal((100, 2), dtype=cp.float32)
        >>> X_outliers = rng.uniform(low=-4, high=4, size=(20, 2)).astype(cp.float32)
        >>> X = cp.vstack([X_inliers, X_outliers])

        >>> # Fit the model
        >>> clf = IsolationForest(n_estimators=100, random_state=42)
        >>> clf.fit(X)
        IsolationForest()

        >>> # Predict anomalies (-1 for anomaly, 1 for normal)
        >>> predictions = clf.predict(X)

        >>> # Get sklearn-compatible anomaly scores (lower = more anomalous)
        >>> scores = clf.score_samples(X)

    Parameters
    ----------
    n_estimators : int, default=100
        The number of isolation trees in the ensemble.
    max_samples : int or float, default=256
        The number of samples to draw from X to train each isolation tree.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * n_samples` samples.
        - If "auto", then `max_samples=min(256, n_samples)`.
    max_depth : int, default=None
        Maximum depth of each isolation tree. If None, depth is set to
        `ceil(log2(max_samples))`, which is the theoretical maximum depth
        needed to isolate any sample.
    max_features : float, default=1.0
        Accepted for sklearn API compatibility. The current GPU builder
        uses all features when selecting random split features.
    bootstrap : bool, default=False
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. Otherwise, sampling is without
        replacement.
    random_state : int, RandomState instance or None, default=None
        Controls random row sampling and split selection. Pass an int for
        reproducible results across runs.
    max_batch_size : int, default=4096
        Accepted for sklearn API compatibility. The current GPU builder
        builds each tree in a single CUDA block and does not batch nodes.
    contamination : float or "auto", default="auto"
        The proportion of outliers in the data set. The current implementation
        uses the sklearn ``"auto"`` offset (-0.5); float contamination values
        are accepted but do not yet compute a data-dependent offset.
    verbose : int or boolean, default=False
        Sets logging level.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \\
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    offset_ : float
        Offset used to compute `decision_function` from raw anomaly scores.

    Notes
    -----
    The implementation is based on the original Isolation Forest paper:
    Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest.
    In 2008 Eighth IEEE International Conference on Data Mining (pp. 413-422).

    **Original Paper Scoring (used internally by C++ backend):**

    The anomaly score is computed as: s(x) = 2^(-E[h(x)] / c(n))

    where:
    - h(x) is the path length of sample x in an isolation tree
    - E[h(x)] is the average path length over all trees
    - c(n) is the average path length in an unsuccessful search in a BST

    In this convention:
    - s ≈ 1.0: Anomaly (short path, easy to isolate)
    - s ≈ 0.5: Normal (average path length)
    - s ≈ 0.0: Very normal (long path, hard to isolate)

    **sklearn Convention (used by Python API):**

    For compatibility with sklearn, the Python method ``score_samples()``
    returns the opposite of the original paper score:
    ``sklearn_score = -paper_score``. ``decision_function()`` then subtracts
    ``offset_``. With the sklearn ``contamination="auto"`` offset of ``-0.5``,
    negative decision function values correspond to paper scores greater than
    0.5 and are predicted as anomalies.

    **Implementation Details:**

    The GPU builder constructs all trees in one CUDA launch with one block per
    tree. For each internal node it selects a random feature, computes the
    minimum and maximum value for that feature in the current node partition,
    and draws a random threshold uniformly between those values. Tree nodes,
    per-tree offsets, and per-tree metadata are stored in RMM-backed global
    memory, which supports both default and deeper non-default tree settings.
    Leaf nodes store pre-computed path lengths (`depth + c(n_leaf)`), which
    keeps inference to a simple tree traversal followed by the Isolation Forest
    score transform.

    Fitted models can be exported to Treelite with ``as_treelite()`` and loaded
    into nvForest with ``as_nvforest()``. The exported Treelite model predicts
    average path length; cuML applies the anomaly score transform separately.

    For additional docs, see `scikit-learn's IsolationForest
    <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html>`_.
    """

    _cpu_class_path = "sklearn.ensemble.IsolationForest"

    def __init__(
        self,
        *,
        n_estimators=100,
        max_samples=256,
        max_depth=None,
        max_features=1.0,
        bootstrap=False,
        random_state=None,
        max_batch_size=4096,
        contamination="auto",
        verbose=False,
        output_type=None,
    ):
        # Initialize internal pointers before Base init so cleanup is safe
        # even if constructor raises early.
        self._forest_float = None
        self._forest_double = None
        self._dtype = None
        self._treelite_model_bytes = None
        self._nvforest_model = None
        self._c_normalization = None

        super().__init__(verbose=verbose, output_type=output_type)

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.max_batch_size = max_batch_size
        self.contamination = contamination

    def __del__(self):
        """Clean up C++ model memory."""
        self._free_model()

    def _free_model(self):
        """Free the C++ model memory."""
        cdef IsolationForestF* f_ptr
        cdef IsolationForestD* d_ptr

        if hasattr(self, "_forest_float") and self._forest_float is not None:
            f_ptr = <IsolationForestF*><uintptr_t>self._forest_float
            del f_ptr
            self._forest_float = None

        if hasattr(self, "_forest_double") and self._forest_double is not None:
            d_ptr = <IsolationForestD*><uintptr_t>self._forest_double
            del d_ptr
            self._forest_double = None

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "n_estimators",
            "max_samples",
            "max_depth",
            "max_features",
            "bootstrap",
            "random_state",
            "max_batch_size",
            "contamination",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        """Convert sklearn model parameters to cuML parameters."""
        if model.warm_start:
            raise UnsupportedOnGPU("`warm_start=True` is not supported")

        return {
            "n_estimators": model.n_estimators,
            "max_samples": model.max_samples,
            "max_features": model.max_features,
            "bootstrap": model.bootstrap,
            "random_state": model.random_state,
            "contamination": model.contamination,
        }

    def _params_to_cpu(self):
        """Convert cuML parameters to sklearn parameters."""
        return {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
            "contamination": self.contamination,
        }

    def __getstate__(self):
        """Pickle support - serialize state."""
        state = self.__dict__.copy()
        # Cannot pickle C++ pointers directly - would need serialization
        state["_forest_float"] = None
        state["_forest_double"] = None
        state.pop("_nvforest_model", None)
        warnings.warn(
            "IsolationForest model serialization is not fully supported. "
            "The model will need to be re-fitted after unpickling."
        )
        return state

    def __setstate__(self, state):
        """Pickle support - restore state."""
        self.__dict__.update(state)

    def fit(self, X, y=None):
        """
        Fit the Isolation Forest model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to float32
            or float64.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : IsolationForest
            Fitted estimator.
        """
        # Free any existing model
        self._free_model()

        # Convert input to cuML array (column-major for fit)
        X_m = input_to_cuml_array(
            X,
            check_dtype=[np.float32, np.float64],
            order="F",  # Column-major for fit
        ).array

        cdef size_t n_rows = X_m.shape[0]
        cdef int n_cols = X_m.shape[1]
        cdef uintptr_t X_ptr = X_m.ptr
        self.n_features_in_ = n_cols
        self._dtype = X_m.dtype

        # Compute max_samples
        cdef int actual_max_samples
        if isinstance(self.max_samples, str) and self.max_samples == "auto":
            actual_max_samples = min(256, n_rows)
        elif isinstance(self.max_samples, float):
            actual_max_samples = int(self.max_samples * n_rows)
        else:
            actual_max_samples = min(self.max_samples, n_rows)

        # Compute max_depth (-1 means auto in C++)
        cdef int actual_max_depth
        if self.max_depth is None:
            actual_max_depth = -1  # C++ will compute ceil(log2(max_samples))
        else:
            actual_max_depth = self.max_depth

        # Get random seed
        cdef uint64_t seed = (
            0 if self.random_state is None
            else check_random_seed(self.random_state)
        )

        # Setup parameters
        cdef IF_params params
        params.n_estimators = self.n_estimators
        params.max_samples = actual_max_samples
        params.max_depth = actual_max_depth
        params.bootstrap = self.bootstrap
        params.seed = seed

        # Get handle and verbosity
        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><uintptr_t>handle.getHandle()
        cdef level_enum verbose = <level_enum>self._verbose_level

        cdef IsolationForestF* forest_f
        cdef IsolationForestD* forest_d
        cdef TreeliteModelHandle tl_handle
        cdef const char* tl_bytes = NULL
        cdef size_t tl_bytes_len

        if X_m.dtype == np.float32:
            forest_f = new IsolationForestF()
            with nogil:
                fit(handle_[0], forest_f, <float*>X_ptr, n_rows, n_cols,
                    params, verbose)
            self._forest_float = <uintptr_t>forest_f
            self._n_samples_per_tree = forest_f.n_samples_per_tree
            self._c_normalization = forest_f.c_normalization
            with nogil:
                build_treelite_isolation_forest[float](
                    &tl_handle, handle_[0], forest_f)
        else:
            forest_d = new IsolationForestD()
            with nogil:
                fit(handle_[0], forest_d, <double*>X_ptr, n_rows, n_cols,
                    params, verbose)
            self._forest_double = <uintptr_t>forest_d
            self._n_samples_per_tree = forest_d.n_samples_per_tree
            self._c_normalization = forest_d.c_normalization
            with nogil:
                build_treelite_isolation_forest[double](
                    &tl_handle, handle_[0], forest_d)

        # Serialize the Treelite handle immediately, following the RandomForest
        # ABI-safe pattern for Python wheels/conda environments.
        safe_treelite_call(
            TreeliteSerializeModelToBytes(tl_handle, &tl_bytes, &tl_bytes_len),
            "Failed to serialize Treelite model to bytes:"
        )
        safe_treelite_call(
            TreeliteFreeModel(tl_handle), "Failed to free Treelite model:"
        )
        self._treelite_model_bytes = <bytes>(tl_bytes[:tl_bytes_len])
        self._nvforest_model = None

        # Only the sklearn "auto" offset is implemented for now. Supporting a
        # float contamination requires scoring the training data and computing
        # the corresponding score quantile.
        if self.contamination == "auto":
            self.offset_ = -0.5
        else:
            self.offset_ = -0.5

        return self

    def as_treelite(self):
        """
        Converts this estimator to a Treelite model.

        The exported Treelite model predicts average path length across the
        isolation trees. cuML applies the Isolation Forest score transform
        separately to produce sklearn-compatible anomaly scores.

        Returns
        -------
        treelite.Model
        """
        if self._treelite_model_bytes is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        return treelite.Model.deserialize_bytes(self._treelite_model_bytes)

    def as_nvforest(
        self, layout="depth_first", default_chunk_size=None, align_bytes=None,
    ):
        """
        Create a nvForest model from the Treelite-exported Isolation Forest.

        Returns
        -------
        nvforest_model : nvforest.ForestInference
            A forest inference model that predicts average path length.
        """
        if self._treelite_model_bytes is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        return nvforest.load_from_treelite_model(
            tl_model=treelite.Model.deserialize_bytes(self._treelite_model_bytes),
            device="gpu",
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
            handle=get_handle(),
        )

    def _get_inference_nvforest_model(
        self,
        layout="depth_first",
        default_chunk_size=None,
        align_bytes=None,
    ):
        if (
            layout == "depth_first" and default_chunk_size is None
            and align_bytes is None
        ):
            if self._nvforest_model is None:
                self._nvforest_model = self.as_nvforest()
            return self._nvforest_model

        return self.as_nvforest(
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
        )

    def _score_samples_nvforest(
        self,
        X,
        layout="depth_first",
        default_chunk_size=None,
        align_bytes=None,
    ):
        """
        Compute sklearn-compatible anomaly scores through nvForest inference.

        This helper is intentionally private while parity and benchmark coverage
        are added. Public ``score_samples`` continues to use the existing C++
        scoring path.
        """
        if self._treelite_model_bytes is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        X_m = input_to_cuml_array(
            X,
            check_dtype=[np.float32, np.float64],
            convert_to_dtype=self._dtype,
            order="C",
        ).array

        if X_m.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_m.shape[1]} features, but IsolationForest was fitted "
                f"with {self.n_features_in_} features."
            )

        nvforest_model = self._get_inference_nvforest_model(
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
        )
        avg_path_lengths = nvforest_model.predict(X_m.to_output("cupy"))
        avg_path_lengths = cp.asarray(avg_path_lengths, dtype=self._dtype)
        if avg_path_lengths.ndim == 2 and avg_path_lengths.shape[1] == 1:
            avg_path_lengths = avg_path_lengths.reshape(-1)

        paper_scores = cp.power(2.0, -avg_path_lengths / self._c_normalization)
        scores_sklearn = -paper_scores

        return CumlArray(scores_sklearn).to_output(self._get_output_type(X))

    def score_samples(self, X):
        """
        Compute the anomaly score of X.

        Returns sklearn-compatible scores where **more negative = more anomalous**.

        .. note:: Score Convention Difference

            The **original paper** (Liu et al. 2008) defines:
                s(x) = 2^(-E[h(x)] / c(n))

            Where higher scores (close to 1) indicate anomalies.

            **sklearn** inverts this convention so that:
                - More negative scores → anomalies
                - Scores closer to 0 → more normal points

            This method follows sklearn's convention for drop-in compatibility.
            The C++ backend returns the original paper's scores, which are then
            transformed here as: sklearn_score = -paper_score

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly scores (sklearn convention: lower/more negative = more anomalous).
            Typical range is approximately [-1.0, 0.0], where values below
            ``offset_`` are predicted as anomalies.
        """
        if self._forest_float is None and self._forest_double is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        # Convert input to cuML array (row-major for inference)
        X_m = input_to_cuml_array(
            X,
            check_dtype=[np.float32, np.float64],
            convert_to_dtype=self._dtype,
            order="C",  # Row-major for inference
        ).array

        cdef size_t n_rows = X_m.shape[0]
        cdef int n_cols = X_m.shape[1]

        if n_cols != self.n_features_in_:
            raise ValueError(
                f"X has {n_cols} features, but IsolationForest was fitted "
                f"with {self.n_features_in_} features."
            )

        # Allocate output
        scores = CumlArray.zeros(n_rows, dtype=self._dtype, order="C")

        # Get handle and verbosity
        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><uintptr_t>handle.getHandle()
        cdef level_enum verbose = <level_enum>self._verbose_level

        cdef uintptr_t X_ptr = X_m.ptr
        cdef uintptr_t scores_ptr = scores.ptr
        cdef IsolationForestF* forest_f
        cdef IsolationForestD* forest_d

        if self._dtype == np.float32:
            forest_f = <IsolationForestF*><uintptr_t>self._forest_float
            with nogil:
                score_samples(handle_[0], forest_f, <float*>X_ptr, n_rows, n_cols,
                             <float*>scores_ptr, verbose)
        else:
            forest_d = <IsolationForestD*><uintptr_t>self._forest_double
            with nogil:
                score_samples(handle_[0], forest_d, <double*>X_ptr, n_rows, n_cols,
                             <double*>scores_ptr, verbose)

        # Transform from original paper convention to sklearn convention:
        #
        # Original paper (Liu et al. 2008):
        #   s(x) = 2^(-E[h(x)] / c(n))
        #   - Anomalies: s ≈ 1 (short paths, easy to isolate)
        #   - Normal:    s ≈ 0.5 (average path length)
        #   - Very normal: s ≈ 0 (long paths, hard to isolate)
        #
        # sklearn convention:
        #   - score_samples returns the opposite of the paper score
        #   - decision_function = score_samples - offset_
        #
        # Transformation: sklearn_score = -paper_score
        #   - paper_score=1.0 (anomaly) → sklearn_score=-1.0
        #   - paper_score=0.5 (normal threshold) → sklearn_score=-0.5
        #   - paper_score=0.0 (v.normal) → sklearn_score=0.0
        #
        scores_cp = scores.to_output("cupy")
        scores_sklearn = -scores_cp

        return CumlArray(scores_sklearn).to_output(self._get_output_type(X))

    def decision_function(self, X):
        """
        Compute the decision function of X.

        The decision function is ``score_samples(X) - offset_``.
        Negative values indicate anomalies.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The decision function. Negative values indicate anomalies.
        """
        return self.score_samples(X) - self.offset_

    def predict(self, X):
        """
        Predict if samples are anomalies or not.

        Returns -1 for anomalies and 1 for normal samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            1 for inliers, -1 for outliers.
        """
        if self._forest_float is None and self._forest_double is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        # Convert input to cuML array (row-major for inference)
        X_m = input_to_cuml_array(
            X,
            check_dtype=[np.float32, np.float64],
            convert_to_dtype=self._dtype,
            order="C",  # Row-major for inference
        ).array

        cdef size_t n_rows = X_m.shape[0]
        cdef int n_cols = X_m.shape[1]

        if n_cols != self.n_features_in_:
            raise ValueError(
                f"X has {n_cols} features, but IsolationForest was fitted "
                f"with {self.n_features_in_} features."
            )

        # Allocate output
        predictions = CumlArray.zeros(n_rows, dtype=np.int32, order="C")

        # Get handle and verbosity
        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><uintptr_t>handle.getHandle()
        cdef level_enum verbose = <level_enum>self._verbose_level

        cdef uintptr_t X_ptr = X_m.ptr
        cdef uintptr_t pred_ptr = predictions.ptr
        cdef IsolationForestF* forest_f
        cdef IsolationForestD* forest_d

        # Use threshold of 0.5 (scores > 0.5 are anomalies in our C++ impl)
        cdef float threshold_f = 0.5
        cdef double threshold_d = 0.5

        if self._dtype == np.float32:
            forest_f = <IsolationForestF*><uintptr_t>self._forest_float
            with nogil:
                predict(handle_[0], forest_f, <float*>X_ptr, n_rows, n_cols,
                       <int*>pred_ptr, threshold_f, verbose)
        else:
            forest_d = <IsolationForestD*><uintptr_t>self._forest_double
            with nogil:
                predict(handle_[0], forest_d, <double*>X_ptr, n_rows, n_cols,
                       <int*>pred_ptr, threshold_d, verbose)

        # Our C++ returns: 1 for anomaly, -1 for normal
        # sklearn returns: -1 for anomaly, 1 for normal
        # So we need to negate
        preds_cp = predictions.to_output("cupy")
        preds_sklearn = -preds_cp

        return CumlArray(preds_sklearn).to_output(self._get_output_type(X))

    def fit_predict(self, X, y=None):
        """
        Fit the model and predict on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            1 for inliers, -1 for outliers.
        """
        return self.fit(X).predict(X)
