#
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
import itertools
import pathlib
from time import perf_counter

import numpy as np
import treelite.sklearn

import cuml.internals.nvtx as nvtx
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.device_type import DeviceType, DeviceTypeError
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.input_utils import input_to_cuml_array
from cuml.internals.mem_type import MemoryType
from cuml.internals.mixins import CMajorInputTagMixin
from cuml.internals.treelite import safe_treelite_call

from libc.stdint cimport uint32_t, uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t as raft_handle_t

from cuml.fil.detail.raft_proto.cuda_stream cimport (
    cuda_stream as raft_proto_stream_t,
)
from cuml.fil.detail.raft_proto.device_type cimport (
    device_type as raft_proto_device_t,
)
from cuml.fil.detail.raft_proto.handle cimport handle_t as raft_proto_handle_t
from cuml.fil.detail.raft_proto.optional cimport nullopt, optional
from cuml.fil.infer_kind cimport infer_kind
from cuml.fil.postprocessing cimport element_op, row_op
from cuml.fil.tree_layout cimport tree_layout as fil_tree_layout
from cuml.internals.treelite cimport (
    TreeliteDeserializeModelFromBytes,
    TreeliteFreeModel,
    TreeliteModelHandle,
)

from cuda.bindings import runtime


cdef extern from "cuml/fil/forest_model.hpp" namespace "ML::fil" nogil:
    cdef cppclass forest_model:
        void predict[io_t](
            const raft_proto_handle_t&,
            io_t*,
            io_t*,
            size_t,
            raft_proto_device_t,
            raft_proto_device_t,
            infer_kind,
            optional[uint32_t]
        ) except +

        bool is_double_precision() except +
        size_t num_features() except +
        size_t num_outputs() except +
        size_t num_trees() except +
        bool has_vector_leaves() except +
        row_op row_postprocessing() except +
        element_op elem_postprocessing() except +

cdef extern from "cuml/fil/treelite_importer.hpp" namespace "ML::fil" nogil:
    forest_model import_from_treelite_handle(
        TreeliteModelHandle,
        fil_tree_layout,
        uint32_t,
        optional[bool],
        raft_proto_device_t,
        int,
        raft_proto_stream_t
    ) except +


class set_fil_device_type:
    """Set the device type used by FIL.

    May optionally be used as a context-manager to set the device type only
    within a context.

    Parameters
    ----------
    device_type : {'cpu', 'gpu'}
        The device type to use.

    Examples
    --------
    >>> from cuml.fil import set_fil_device_type  # doctest: +SKIP

    Set the device type globally to use CPU.

    >>> set_fil_device_type("cpu")  # doctest: +SKIP

    Set the device type globally to use GPU.

    >>> set_fil_device_type("gpu")  # doctest: +SKIP

    Set the device type to use CPU within a context.

    >>> with set_fil_device_type("cpu"):  # doctest: +SKIP
    ...     ...
    """
    def __init__(self, device_type):
        device_type = DeviceType.from_str(device_type)
        self._previous = GlobalSettings().fil_device_type
        GlobalSettings().fil_device_type = device_type

    def __enter__(self):
        return self

    def __exit__(self, *_):
        GlobalSettings().fil_device_type = self._previous


def get_fil_device_type() -> DeviceType:
    """Get the device type used by FIL."""
    return GlobalSettings().fil_device_type


cdef raft_proto_device_t get_fil_raft_proto_device_type(arr):
    """Get the current FIL device type as a raft_proto_device_t"""
    cdef raft_proto_device_t dev
    if arr.is_device_accessible:
        if arr.is_host_accessible and get_fil_device_type() is DeviceType.host:
            dev = raft_proto_device_t.cpu
        else:
            dev = raft_proto_device_t.gpu
    else:
        dev = raft_proto_device_t.cpu
    return dev


cdef class ForestInference_impl():
    cdef forest_model model
    cdef raft_proto_handle_t raft_proto_handle
    cdef object raft_handle

    def __cinit__(
        self,
        raft_handle,
        tl_model_bytes,
        *,
        layout='depth_first',
        align_bytes=0,
        use_double_precision=None,
        mem_type=None,
        device_id=None,
    ):
        # Store reference to RAFT handle to control lifetime, since raft_proto
        # handle keeps a pointer to it
        self.raft_handle = raft_handle
        self.raft_proto_handle = raft_proto_handle_t(
            <raft_handle_t*><size_t>self.raft_handle.getHandle()
        )
        if mem_type is None:
            mem_type = GlobalSettings().fil_memory_type
        else:
            mem_type = MemoryType.from_str(mem_type)

        cdef optional[bool] use_double_precision_c
        cdef bool use_double_precision_bool
        if use_double_precision is None:
            use_double_precision_c = nullopt
        else:
            use_double_precision_bool = use_double_precision
            use_double_precision_c = use_double_precision_bool

        cdef TreeliteModelHandle tl_handle = NULL
        safe_treelite_call(
            TreeliteDeserializeModelFromBytes(
                tl_model_bytes, len(tl_model_bytes), &tl_handle),
            "Failed to load Treelite model from bytes:"
        )

        cdef raft_proto_device_t dev_type
        if mem_type.is_device_accessible:
            dev_type = raft_proto_device_t.gpu
        else:
            dev_type = raft_proto_device_t.cpu
        cdef fil_tree_layout tree_layout
        if layout.lower() == "depth_first":
            tree_layout = fil_tree_layout.depth_first
        elif layout.lower() == "breadth_first":
            tree_layout = fil_tree_layout.breadth_first
        elif layout.lower() == "layered":
            tree_layout = fil_tree_layout.layered_children_together
        else:
            raise RuntimeError(f"Unrecognized tree layout {layout}")

        # Use assertion here, since device_id being None would indicate
        # a bug, not a user error. The outer ForestInference object
        # should set an integer device_id before passing it to
        # ForestInference_impl.
        assert device_id is not None, (
            "device_id should be set before building ForestInference_impl"
        )

        self.model = import_from_treelite_handle(
            tl_handle,
            tree_layout,
            align_bytes,
            use_double_precision_c,
            dev_type,
            device_id,
            self.raft_proto_handle.get_next_usable_stream()
        )

        safe_treelite_call(
            TreeliteFreeModel(tl_handle),
            "Failed to free Treelite model:"
        )

    def get_dtype(self):
        return [np.float32, np.float64][self.model.is_double_precision()]

    def num_features(self):
        return self.model.num_features()

    def num_outputs(self):
        return self.model.num_outputs()

    def num_trees(self):
        return self.model.num_trees()

    def row_postprocessing(self):
        enum_val = self.model.row_postprocessing()
        if enum_val == row_op.row_disable:
            return "disable"
        elif enum_val == row_op.softmax:
            return "softmax"
        elif enum_val == row_op.max_index:
            return "max_index"

    def elem_postprocessing(self):
        enum_val = self.model.elem_postprocessing()
        if enum_val == element_op.elem_disable:
            return "disable"
        elif enum_val == element_op.signed_square:
            return "signed_square"
        elif enum_val == element_op.hinge:
            return "hinge"
        elif enum_val == element_op.sigmoid:
            return "sigmoid"
        elif enum_val == element_op.exponential:
            return "exponential"
        elif enum_val == element_op.logarithm_one_plus_exp:
            return "logarithm_one_plus_exp"

    def _predict(self, X, *, predict_type="default", preds=None, chunk_size=None):
        model_dtype = self.get_dtype()

        cdef uintptr_t in_ptr
        in_arr, n_rows, _, _ = input_to_cuml_array(
            X,
            order='C',
            convert_to_dtype=model_dtype,
            convert_to_mem_type=GlobalSettings().fil_memory_type,
            check_dtype=model_dtype
        )
        cdef raft_proto_device_t in_dev
        in_dev = get_fil_raft_proto_device_type(in_arr)
        in_ptr = in_arr.ptr

        cdef uintptr_t out_ptr
        cdef infer_kind infer_type_enum
        if predict_type == "default":
            infer_type_enum = infer_kind.default_kind
            output_shape = (n_rows, self.model.num_outputs())
        elif predict_type == "per_tree":
            infer_type_enum = infer_kind.per_tree
            if self.model.has_vector_leaves():
                output_shape = (n_rows, self.model.num_trees(), self.model.num_outputs())
            else:
                output_shape = (n_rows, self.model.num_trees())
        elif predict_type == "leaf_id":
            infer_type_enum = infer_kind.leaf_id
            output_shape = (n_rows, self.model.num_trees())
        else:
            raise ValueError(f"Unrecognized predict_type: {predict_type}")
        if preds is None:
            preds = CumlArray.empty(
                output_shape,
                model_dtype,
                order='C',
                index=in_arr.index,
                mem_type=GlobalSettings().fil_memory_type,
            )
        else:
            # TODO(wphicks): Handle incorrect dtype/device/layout in C++
            if preds.shape != output_shape:
                raise ValueError(f"If supplied, preds argument must have shape {output_shape}")
            preds.index = in_arr.index
        cdef raft_proto_device_t out_dev
        out_dev = get_fil_raft_proto_device_type(preds)
        out_ptr = preds.ptr

        cdef optional[uint32_t] chunk_specification
        if chunk_size is None:
            chunk_specification = nullopt
        else:
            chunk_specification = <uint32_t> chunk_size

        if model_dtype == np.float32:
            self.model.predict[float](
                self.raft_proto_handle,
                <float*> out_ptr,
                <float*> in_ptr,
                n_rows,
                out_dev,
                in_dev,
                infer_type_enum,
                chunk_specification
            )
        else:
            self.model.predict[double](
                self.raft_proto_handle,
                <double*> out_ptr,
                <double*> in_ptr,
                n_rows,
                in_dev,
                out_dev,
                infer_type_enum,
                chunk_specification
            )

        if get_fil_device_type() is DeviceType.device:
            self.raft_proto_handle.synchronize()
        return preds

    def predict(
        self,
        X,
        *,
        predict_type="default",
        preds=None,
        chunk_size=None,
    ):
        return self._predict(
            X,
            predict_type=predict_type,
            preds=preds,
            chunk_size=chunk_size,
        )


class _AutoIterations:
    """Used to generate sequence of iterations (1, 2, 5, 10, 20, 50...) during
    FIL optimization"""

    def __init__(self):
        self.invocations = 0
        self.sequence = (1, 2, 5)

    def next(self):
        result = (
            (10 ** (
                self.invocations // len(self.sequence)
            )) * self.sequence[self.invocations % len(self.sequence)]
        )
        self.invocations += 1
        return result


class ForestInference(Base, CMajorInputTagMixin):
    """
    ForestInference provides accelerated inference for forest models on both
    CPU and GPU.

    **Performance Tuning**
    FIL offers a number of hyperparameters that can be tuned to obtain optimal
    performance for a given model, hardware, and batch size. The easiest way to
    optimize these parameters is using the automated `.optimize` method, which
    will find the optimum for an indicated batch size. For some use cases,
    manual adjustment of these parameters is preferred, so available
    performance hyperparameters are described in detail below.

    To obtain optimal performance with this implementation of FIL, the single
    most important value is the `chunk_size` parameter passed to the predict
    method. Essentially, `chunk_size` determines how many rows to evaluate
    together at once from a single batch. Larger values reduce global memory
    accesses on GPU and cache misses on CPU, but smaller values allow for
    finer-grained parallelism, improving usage of available processing power.
    The optimal value for this parameter is hard to predict a priori, but in
    general larger batch sizes benefit from larger chunk sizes and smaller
    batch sizes benefit from smaller chunk sizes. Having a chunk size larger
    than the batch size is never optimal.

    To determine the optimal chunk size on GPU, test powers of 2 from 1 to
    32. Values above 32 and values which are not powers of 2 are not supported.

    To determine the optimal chunk size on CPU, test powers of 2 from 1 to
    512. Values above 512 are supported, but RAPIDS developers have not yet
    seen a case where they yield improved performance.

    After chunk size, the most important performance parameter is `layout`,
    also described below. Testing available layouts is recommended to optimize
    performance, but the impact is likely to be substantially less than
    optimizing `chunk_size`. There is no universal rule for predicting which
    layout will produce the best performance. On both GPU and CPU, the
    `depth_first` layout can improve performance by increasing cache hits
    during tree traversal. This tends to be the strongest effect for most use
    cases, so `depth_first` is used as the default value.

    `align_bytes` is the final performance parameter. This parameter allows
    trees to be padded with empty nodes until their total in-memory size is a
    multiple of the given value. In general, if a non-default value is used, it
    should either be 0 or the cache line byte size for the device being used
    for execution (64 for CPU or 128 for GPU). If left unpadded, forest data
    remains more compact in memory, which can improve the frequency of cache
    hits. On the other hand, padding to the size of the cache line ensures that
    trees begin on cache line boundaries. It is difficult to predict for any
    given model which effect will be the greater determinant of performance. If
    left at the default value of `None`, trees will be unpadded for GPU
    execution and padded to 64 bytes for CPU execution. This value has no
    effect for the `layered` layout, since trees in this layout overlap in
    memory.

    Parameters
    ----------
    treelite_model : treelite.Model
        The model to be used for inference. This can be trained with XGBoost,
        LightGBM, cuML, Scikit-Learn, or any other forest model framework
        so long as it can be loaded into a treelite.Model object (See
        https://treelite.readthedocs.io/en/latest/treelite-api.html).
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
    layout : {'breadth_first', 'depth_first', 'layered'}, default='depth_first'
        The in-memory layout to be used during inference for nodes of the
        forest model. This parameter is available purely for runtime
        optimization. For performance-critical applications, it is
        recommended that each layout be tested with realistic batch sizes to
        determine the optimal value.
    align_bytes : int or None, default=None
        Pad each tree with empty nodes until its in-memory size is a multiple
        of the given value. If None, use 0 for GPU and 64 for CPU.
    precision : {'single', 'double', None}, default='single'
        Use the given floating point precision for evaluating the model. If
        None, use the native precision of the model. Note that
        single-precision execution is substantially faster than
        double-precision execution, so double-precision is recommended
        only for models trained and double precision and when exact
        conformance between results from FIL and the original training
        framework is of paramount importance.
    device_id : int or None, default=None
        For GPU execution, the device on which to load and execute this
        model. If set to None, use the currently active device.
        For CPU execution, this value is currently ignored.
    """

    def _reload_model(self):
        """Reload model on any device (CPU/GPU) where model has already been
        loaded"""
        if hasattr(self, '_gpu_forest'):
            with set_fil_device_type('gpu'):
                self._load_to_fil(device_id=self.device_id)
        if hasattr(self, '_cpu_forest'):
            with set_fil_device_type('cpu'):
                self._load_to_fil(device_id=self.device_id)

    @staticmethod
    def _get_default_align_bytes():
        if get_fil_device_type() is DeviceType.host:
            return 64
        else:
            return 0

    @property
    def align_bytes(self):
        try:
            return self._align_bytes_
        except AttributeError:
            return self._get_default_align_bytes()

    @align_bytes.setter
    def align_bytes(self, value):
        try:
            old_value = self._align_bytes_
        except AttributeError:
            old_value = None
        if value is None:
            if old_value is not None:
                del self._align_bytes_
                self._reload_model()
        else:
            self._align_bytes_ = value
            if old_value is None or value != old_value:
                self._reload_model()

    @property
    def precision(self):
        try:
            use_double_precision = \
                self._use_double_precision_
        except AttributeError:
            self._use_double_precision_ = False
            use_double_precision = \
                self._use_double_precision_
        if use_double_precision is None:
            return 'native'
        elif use_double_precision:
            return 'double'
        else:
            return 'single'

    @precision.setter
    def precision(self, value):
        try:
            old_value = self._use_double_precision_
        except AttributeError:
            self._use_double_precision_ = False
            old_value = self._use_double_precision_
        if value in ('native', None):
            self._use_double_precision_ = None
        elif value in ('double', 'float64'):
            self._use_double_precision_ = True
        else:
            self._use_double_precision_ = False
        if old_value != self._use_double_precision_:
            self._reload_model()

    @property
    def is_classifier(self):
        try:
            return self._is_classifier_
        except AttributeError:
            self._is_classifier_ = False
            return self._is_classifier_

    @is_classifier.setter
    def is_classifier(self, value):
        if not hasattr(self, '_is_classifier_'):
            self._is_classifier_ = value
        elif value is not None:
            self._is_classifier_ = value

    @property
    def device_id(self):
        try:
            return self._device_id_
        except AttributeError:
            self._device_id_ = None
            return self._device_id_

    @device_id.setter
    def device_id(self, value):
        try:
            old_value = self.device_id
        except AttributeError:
            old_value = None
        self._device_id_ = value
        if (
            self.treelite_model is not None
            and self.device_id != old_value
            and hasattr(self, '_gpu_forest')
        ):
            self._load_to_fil(device_id=self.device_id)

    @property
    def treelite_model(self):
        try:
            return self._treelite_model_
        except AttributeError:
            return None

    @treelite_model.setter
    def treelite_model(self, value):
        if value is not None:
            self._treelite_model_ = value
            self._reload_model()

    @property
    def layout(self):
        try:
            return self._layout_
        except AttributeError:
            self._layout_ = 'depth_first'
        return self._layout_

    @layout.setter
    def layout(self, value):
        try:
            old_value = self._layout_
        except AttributeError:
            old_value = None
        if value is not None:
            self._layout_ = value
        if old_value != value:
            self._reload_model()

    def __init__(
        self,
        *,
        treelite_model=None,
        handle=None,
        output_type=None,
        verbose=False,
        is_classifier=False,
        layout='depth_first',
        default_chunk_size=None,
        align_bytes=None,
        precision='single',
        device_id=None,
    ):
        super().__init__(
            handle=handle, verbose=verbose, output_type=output_type
        )
        self.is_classifier = is_classifier
        self.default_chunk_size = default_chunk_size
        self.align_bytes = align_bytes
        self.layout = layout
        self.precision = precision
        self.device_id = device_id
        self.treelite_model = treelite_model
        self._load_to_fil(device_id=self.device_id)

    def _load_to_fil(self, mem_type=None, device_id=None):
        if mem_type is None:
            mem_type = GlobalSettings().fil_memory_type
        else:
            mem_type = MemoryType.from_str(mem_type)

        if device_id is None:
            # If no device ID is explicitly given, use the currently
            # active device
            status, current_device_id = runtime.cudaGetDevice()
            if status != runtime.cudaError_t.cudaSuccess:
                _, name = runtime.cudaGetErrorName(status)
                _, msg = runtime.cudaGetErrorString(status)
                raise RuntimeError(f"Failed to run cudaGetDevice(). {name}: {msg}")
            device_id = current_device_id

        if mem_type.is_device_accessible:
            self.device_id = device_id

        if self.treelite_model is not None:
            if isinstance(self.treelite_model, treelite.Model):
                treelite_model_bytes = self.treelite_model.serialize_bytes()
            elif isinstance(self.treelite_model, bytes):
                treelite_model_bytes = self.treelite_model
            else:
                raise ValueError("treelite_model should be either treelite.Model or bytes")
            impl = ForestInference_impl(
                self.handle,
                treelite_model_bytes,
                layout=self.layout,
                align_bytes=self.align_bytes,
                use_double_precision=self._use_double_precision_,
                mem_type=mem_type,
                device_id=self.device_id
            )

            if mem_type.is_device_accessible:
                self._gpu_forest = impl

            if mem_type.is_host_accessible:
                self._cpu_forest = impl

    @property
    def gpu_forest(self):
        """The underlying FIL forest model loaded in GPU-accessible memory"""
        try:
            return self._gpu_forest
        except AttributeError:
            self._load_to_fil(mem_type=MemoryType.device)
            return self._gpu_forest

    @property
    def cpu_forest(self):
        """The underlying FIL forest model loaded in CPU-accessible memory"""
        try:
            return self._cpu_forest
        except AttributeError:
            self._load_to_fil(mem_type=MemoryType.host)
            return self._cpu_forest

    @property
    def forest(self):
        """The underlying FIL forest model loaded in memory compatible with the
        current global device_type setting"""
        device_type = get_fil_device_type()
        if device_type is DeviceType.device:
            return self.gpu_forest
        elif device_type is DeviceType.host:
            return self.cpu_forest
        else:
            raise DeviceTypeError("Unsupported device type for FIL")

    def num_outputs(self):
        return self.forest.num_outputs()

    def num_trees(self):
        return self.forest.num_trees()

    @classmethod
    def load(
        cls,
        path,
        *,
        is_classifier=False,
        precision='single',
        model_type=None,
        output_type=None,
        verbose=False,
        default_chunk_size=None,
        align_bytes=None,
        layout='depth_first',
        device_id=0,
        handle=None
    ):
        """Load a model into FIL from a serialized model file.

        Parameters
        ----------
        path : str
            The path to the serialized model file. This can be an XGBoost
            binary or JSON file, a LightGBM text file, or a Treelite checkpoint
            file. If the model_type parameter is not passed, an attempt will be
            made to load the file based on its extension.
        is_classifier : boolean, default=False
            True for classification models, False for regressors
        precision : {'single', 'double', None}, default='single'
            Use the given floating point precision for evaluating the model. If
            None, use the native precision of the model. Note that
            single-precision execution is substantially faster than
            double-precision execution, so double-precision is recommended
            only for models trained and double precision and when exact
            conformance between results from FIL and the original training
            framework is of paramount importance.
        model_type : {'xgboost_ubj', 'xgboost_json', 'xgboost', 'lightgbm',
            'treelite_checkpoint', None }, default=None
            The serialization format for the model file. If None, a best-effort
            guess will be made based on the file extension.
        output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
            'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
            Return results and set estimator attributes to the indicated output
            type. If None, the output type set at the module level
            (`cuml.global_settings.output_type`) will be used. See
            :ref:`output-data-type-configuration` for more info.
        verbose : int or boolean, default=False
            Sets logging level. It must be one of `cuml.common.logger.level_*`.
            See :ref:`verbosity-levels` for more info.
        default_chunk_size : int or None, default=None
            If set, predict calls without a specified chunk size will use
            this default value.
        align_bytes : int or None, default=None
            Pad each tree with empty nodes until its in-memory size is a multiple
            of the given value. If None, use 0 for GPU and 64 for CPU.
        layout : {'breadth_first', 'depth_first', 'layered'}, default='depth_first'
            The in-memory layout to be used during inference for nodes of the
            forest model. This parameter is available purely for runtime
            optimization. For performance-critical applications, it is
            recommended that available layouts be tested with realistic batch
            sizes to determine the optimal value.
        device_id : int, default=0
            For GPU execution, the device on which to load and execute this
            model. For CPU execution, this value is currently ignored.
        handle : pylibraft.common.handle or None
            For GPU execution, the RAFT handle containing the stream or stream
            pool to use during loading and inference.
        """
        if model_type is None:
            extension = pathlib.Path(path).suffix
            if extension == '.json':
                model_type = 'xgboost_json'
            elif extension == '.ubj':
                model_type = 'xgboost_ubj'
            elif extension == '.model':
                model_type = 'xgboost'
            elif extension == '.txt':
                model_type = 'lightgbm'
            else:
                model_type = 'treelite_checkpoint'
        if model_type == "treelite_checkpoint":
            tl_model = treelite.frontend.Model.deserialize(path)
        elif model_type == "xgboost_ubj":
            tl_model = treelite.frontend.load_xgboost_model(path, format_choice="ubjson")
        elif model_type == "xgboost_json":
            tl_model = treelite.frontend.load_xgboost_model(path, format_choice="json")
        elif model_type == "xgboost":
            tl_model = treelite.frontend.load_xgboost_model_legacy_binary(path)
        elif model_type == "lightgbm":
            tl_model = treelite.frontend.load_lightgbm_model(path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls(
            treelite_model=tl_model,
            handle=handle,
            output_type=output_type,
            verbose=verbose,
            is_classifier=is_classifier,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
            layout=layout,
            precision=precision,
            device_id=device_id
        )

    @classmethod
    def load_from_sklearn(
            cls,
            skl_model,
            *,
            is_classifier=False,
            precision='single',
            model_type=None,
            output_type=None,
            verbose=False,
            default_chunk_size=None,
            align_bytes=None,
            layout='depth_first',
            device_id=0,
            handle=None):
        """Load a Scikit-Learn forest model to FIL

        Parameters
        ----------
        skl_model
            The Scikit-Learn forest model to load.
        is_classifier : boolean, default=False
            True for classification models, False for regressors
        precision : {'single', 'double', None}, default='single'
            Use the given floating point precision for evaluating the model. If
            None, use the native precision of the model. Note that
            single-precision execution is substantially faster than
            double-precision execution, so double-precision is recommended
            only for models trained and double precision and when exact
            conformance between results from FIL and the original training
            framework is of paramount importance.
        model_type : {'xgboost', 'xgboost_json', 'lightgbm',
            'treelite_checkpoint', None }, default=None
            The serialization format for the model file. If None, a best-effort
            guess will be made based on the file extension.
        output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
            'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
            Return results and set estimator attributes to the indicated output
            type. If None, the output type set at the module level
            (`cuml.global_settings.output_type`) will be used. See
            :ref:`output-data-type-configuration` for more info.
        verbose : int or boolean, default=False
            Sets logging level. It must be one of `cuml.common.logger.level_*`.
            See :ref:`verbosity-levels` for more info.
        default_chunk_size : int or None, default=None
            If set, predict calls without a specified chunk size will use
            this default value.
        align_bytes : int or None, default=None
            Pad each tree with empty nodes until its in-memory size is a multiple
            of the given value. If None, use 0 for GPU and 64 for CPU.
        layout : {'breadth_first', 'depth_first', 'layered'}, default='depth_first'
            The in-memory layout to be used during inference for nodes of the
            forest model. This parameter is available purely for runtime
            optimization. For performance-critical applications, it is
            recommended that available layouts be tested with realistic batch
            sizes to determine the optimal value.
        mem_type : {'device', 'host', None}, default='single'
            The memory type to use for initially loading the model. If None,
            the current global memory type setting will be used. If the model
            is loaded with one memory type and inference is later requested
            with an incompatible device (e.g. device memory and CPU execution),
            the model will be lazily loaded to the correct location at that
            time. In general, it should not be necessary to set this parameter
            directly (rely instead on the `set_fil_device_type` context manager),
            but it can be a useful convenience for some hyperoptimization
            pipelines.
        device_id : int, default=0
            For GPU execution, the device on which to load and execute this
            model. For CPU execution, this value is currently ignored.
        handle : pylibraft.common.handle or None
            For GPU execution, the RAFT handle containing the stream or stream
            pool to use during loading and inference.
        """
        tl_model = treelite.sklearn.import_model(skl_model)
        result = cls(
            treelite_model=tl_model,
            handle=handle,
            output_type=output_type,
            verbose=verbose,
            is_classifier=is_classifier,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
            layout=layout,
            precision=precision,
            device_id=device_id
        )
        return result

    @classmethod
    def load_from_treelite_model(
            cls,
            tl_model,
            *,
            is_classifier=False,
            precision='single',
            model_type=None,
            output_type=None,
            verbose=False,
            default_chunk_size=None,
            align_bytes=None,
            layout='depth_first',
            device_id=0,
            handle=None):
        """Load a Treelite model to FIL

        Parameters
        ----------
        tl_model : treelite.Model
            The Treelite model to load.
        is_classifier : boolean, default=False
            True for classification models, False for regressors
        precision : {'single', 'double', None}, default='single'
            Use the given floating point precision for evaluating the model. If
            None, use the native precision of the model. Note that
            single-precision execution is substantially faster than
            double-precision execution, so double-precision is recommended
            only for models trained and double precision and when exact
            conformance between results from FIL and the original training
            framework is of paramount importance.
        model_type : {'xgboost', 'xgboost_json', 'lightgbm',
            'treelite_checkpoint', None }, default=None
            The serialization format for the model file. If None, a best-effort
            guess will be made based on the file extension.
        output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
            'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
            Return results and set estimator attributes to the indicated output
            type. If None, the output type set at the module level
            (`cuml.global_settings.output_type`) will be used. See
            :ref:`output-data-type-configuration` for more info.
        verbose : int or boolean, default=False
            Sets logging level. It must be one of `cuml.common.logger.level_*`.
            See :ref:`verbosity-levels` for more info.
        default_chunk_size : int or None, default=None
            If set, predict calls without a specified chunk size will use
            this default value.
        align_bytes : int or None, default=None
            Pad each tree with empty nodes until its in-memory size is a multiple
            of the given value. If None, use 0 for GPU and 64 for CPU.
        layout : {'breadth_first', 'depth_first', 'layered'}, default='depth_first'
            The in-memory layout to be used during inference for nodes of the
            forest model. This parameter is available purely for runtime
            optimization. For performance-critical applications, it is
            recommended that available layouts be tested with realistic batch
            sizes to determine the optimal value.
        mem_type : {'device', 'host', None}, default='single'
            The memory type to use for initially loading the model. If None,
            the current global memory type setting will be used. If the model
            is loaded with one memory type and inference is later requested
            with an incompatible device (e.g. device memory and CPU execution),
            the model will be lazily loaded to the correct location at that
            time. In general, it should not be necessary to set this parameter
            directly (rely instead on the `set_fil_device_type` context
            manager), but it can be a useful convenience for some
            hyperoptimization pipelines.
        device_id : int, default=0
            For GPU execution, the device on which to load and execute this
            model. For CPU execution, this value is currently ignored.
        handle : pylibraft.common.handle or None
            For GPU execution, the RAFT handle containing the stream or stream
            pool to use during loading and inference.
        """
        return cls(
            treelite_model=tl_model,
            handle=handle,
            output_type=output_type,
            verbose=verbose,
            is_classifier=is_classifier,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
            layout=layout,
            precision=precision,
            device_id=device_id
        )

    @nvtx.annotate(
        message='ForestInference.predict_proba',
        domain='cuml_python'
    )
    def predict_proba(
        self,
        X,
        *,
        preds=None,
        chunk_size=None,
    ) -> CumlArray:
        """
        Predict the class probabilities for each row in X.

        Parameters
        ----------
        X
            The input data of shape Rows X Features. This can be a numpy
            array, cupy array, Pandas/cuDF Dataframe or any other array type
            accepted by cuML. FIL is optimized for C-major arrays (e.g.
            numpy/cupy arrays). Inputs whose datatype does not match the
            precision of the loaded model (float/double) will be converted
            to the correct datatype before inference. If this input is in a
            memory location that is inaccessible to the current device type
            (as set with e.g. the `set_fil_device_type` context manager),
            it will be copied to the correct location. This copy will be
            distributed across as many CUDA streams as are available
            in the stream pool of the model's RAFT handle.
        preds
            If non-None, outputs will be written in-place to this array.
            Therefore, if given, this should be a C-major array of shape Rows x
            Classes with a datatype (float/double) corresponding to the
            precision of the model. If None, an output array of the correct
            shape and type will be allocated and returned.
        chunk_size : int
            The number of rows to simultaneously process in one iteration
            of the inference algorithm. Batches are further broken down into
            "chunks" of this size when assigning available threads to tasks.
            The choice of chunk size can have a substantial impact on
            performance, but the optimal choice depends on model and
            hardware and is difficult to predict a priori. In general,
            larger batch sizes benefit from larger chunk sizes, and smaller
            batch sizes benefit from small chunk sizes. On GPU, valid
            values are powers of 2 from 1 to 32. On CPU, valid values are
            any power of 2, but little benefit is expected above a chunk size
            of 512.
        """
        if not self.is_classifier:
            raise RuntimeError(
                "predict_proba is not available for regression models. Load"
                " with is_classifier=True if this is a classifier."
            )
        return self.forest.predict(
            X, preds=preds, chunk_size=(chunk_size or self.default_chunk_size)
        )

    @nvtx.annotate(
        message='ForestInference.predict',
        domain='cuml_python'
    )
    def predict(
        self,
        X,
        *,
        preds=None,
        chunk_size=None,
        threshold=None,
    ) -> CumlArray:
        """
        For classification models, predict the class for each row. For
        regression models, predict the output for each row.

        Parameters
        ----------
        X
            The input data of shape Rows X Features. This can be a numpy
            array, cupy array, Pandas/cuDF Dataframe or any other array type
            accepted by cuML. FIL is optimized for C-major arrays (e.g.
            numpy/cupy arrays). Inputs whose datatype does not match the
            precision of the loaded model (float/double) will be converted
            to the correct datatype before inference. If this input is in a
            memory location that is inaccessible to the current device type
            (as set with e.g. the `set_fil_device_type` context manager),
            it will be copied to the correct location. This copy will be
            distributed across as many CUDA streams as are available
            in the stream pool of the model's RAFT handle.
        preds
            If non-None, outputs will be written in-place to this array.
            Therefore, if given, this should be a C-major array of shape Rows x
            1 with a datatype (float/double) corresponding to the precision of
            the model. If None, an output array of the correct shape and
            type will be allocated and returned. For classifiers, in-place
            prediction offers no performance or memory benefit. For regressors,
            in-place prediction offers both a performance and memory
            benefit.
        chunk_size : int
            The number of rows to simultaneously process in one iteration
            of the inference algorithm. Batches are further broken down into
            "chunks" of this size when assigning available threads to tasks.
            The choice of chunk size can have a substantial impact on
            performance, but the optimal choice depends on model and
            hardware and is difficult to predict a priori. In general,
            larger batch sizes benefit from larger chunk sizes, and smaller
            batch sizes benefit from small chunk sizes. On GPU, valid
            values are powers of 2 from 1 to 32. On CPU, valid values are
            any power of 2, but little benefit is expected above a chunk size
            of 512.
        threshold : float
            For binary classifiers, output probabilities above this threshold
            will be considered positive detections. If None, a threshold
            of 0.5 will be used for binary classifiers. For multiclass
            classifiers, the highest probability class is chosen regardless
            of threshold.
        """
        chunk_size = (chunk_size or self.default_chunk_size)
        if self.forest.row_postprocessing() == 'max_index':
            raw_out = self.forest.predict(X, chunk_size=chunk_size)
            result = raw_out[:, 0]
            if preds is None:
                return result
            else:
                preds[:] = result
                return preds
        elif self.is_classifier:
            proba = self.forest.predict(X, chunk_size=chunk_size)
            if len(proba.shape) < 2 or proba.shape[1] == 1:
                if threshold is None:
                    threshold = 0.5
                result = (
                    proba.to_output(output_type='array') > threshold
                ).astype('int')
            else:
                result = GlobalSettings().fil_xpy.argmax(
                    proba.to_output(output_type='array'), axis=1
                )
            if preds is None:
                return result
            else:
                preds[:] = result
                return preds
        else:
            return self.forest.predict(
                X, predict_type="default", preds=preds, chunk_size=chunk_size
            )

    @nvtx.annotate(
        message='ForestInference.predict_per_tree',
        domain='cuml_python'
    )
    def predict_per_tree(
            self,
            X,
            *,
            preds=None,
            chunk_size=None) -> CumlArray:
        """
        Output prediction of each tree.
        This function computes one or more margin scores per tree.

        Parameters
        ----------
        X
            The input data of shape Rows X Features. This can be a numpy
            array, cupy array, Pandas/cuDF Dataframe or any other array type
            accepted by cuML. FIL is optimized for C-major arrays (e.g.
            numpy/cupy arrays). Inputs whose datatype does not match the
            precision of the loaded model (float/double) will be converted
            to the correct datatype before inference. If this input is in a
            memory location that is inaccessible to the current device type
            (as set with e.g. the `set_fil_device_type` context manager),
            it will be copied to the correct location. This copy will be
            distributed across as many CUDA streams as are available
            in the stream pool of the model's RAFT handle.
        preds
            If non-None, outputs will be written in-place to this array.
            Therefore, if given, this should be a C-major array of shape
            n_rows * n_trees * n_outputs (if vector leaf is used) or
            shape n_rows * n_trees (if scalar leaf is used).
            Classes with a datatype (float/double) corresponding to the
            precision of the model. If None, an output array of the correct
            shape and type will be allocated and returned.
        chunk_size : int
            The number of rows to simultaneously process in one iteration
            of the inference algorithm. Batches are further broken down into
            "chunks" of this size when assigning available threads to tasks.
            The choice of chunk size can have a substantial impact on
            performance, but the optimal choice depends on model and
            hardware and is difficult to predict a priori. In general,
            larger batch sizes benefit from larger chunk sizes, and smaller
            batch sizes benefit from small chunk sizes. On GPU, valid
            values are powers of 2 from 1 to 32. On CPU, valid values are
            any power of 2, but little benefit is expected above a chunk size
            of 512.
        """
        chunk_size = (chunk_size or self.default_chunk_size)
        return self.forest.predict(
            X, predict_type="per_tree", preds=preds, chunk_size=chunk_size
        )

    @nvtx.annotate(
        message='ForestInference.apply',
        domain='cuml_python'
    )
    def apply(
            self,
            X,
            *,
            preds=None,
            chunk_size=None) -> CumlArray:
        """
        Output the ID of the leaf node for each tree.

        Parameters
        ----------
        X
            The input data of shape Rows X Features. This can be a numpy
            array, cupy array, Pandas/cuDF Dataframe or any other array type
            accepted by cuML. FIL is optimized for C-major arrays (e.g.
            numpy/cupy arrays). Inputs whose datatype does not match the
            precision of the loaded model (float/double) will be converted
            to the correct datatype before inference. If this input is in a
            memory location that is inaccessible to the current device type
            (as set with e.g. the `set_fil_device_type` context manager),
            it will be copied to the correct location. This copy will be
            distributed across as many CUDA streams as are available
            in the stream pool of the model's RAFT handle.
        preds
            If non-None, outputs will be written in-place to this array.
            Therefore, if given, this should be a C-major array of shape
            n_rows * n_trees.
            Classes with a datatype (float/double) corresponding to the
            precision of the model. If None, an output array of the correct
            shape and type will be allocated and returned.
        chunk_size : int
            The number of rows to simultaneously process in one iteration
            of the inference algorithm. Batches are further broken down into
            "chunks" of this size when assigning available threads to tasks.
            The choice of chunk size can have a substantial impact on
            performance, but the optimal choice depends on model and
            hardware and is difficult to predict a priori. In general,
            larger batch sizes benefit from larger chunk sizes, and smaller
            batch sizes benefit from small chunk sizes. On GPU, valid
            values are powers of 2 from 1 to 32. On CPU, valid values are
            any power of 2, but little benefit is expected above a chunk size
            of 512.
        """
        return self.forest.predict(
            X, predict_type="leaf_id", preds=preds, chunk_size=chunk_size
        )

    def optimize(
        self,
        *,
        data=None,
        batch_size=1024,
        unique_batches=10,
        timeout=0.2,
        predict_method='predict',
        max_chunk_size=None,
        seed=0
    ):
        """
        Find the optimal layout and chunk size for this model

        The optimal value for layout and chunk size depends on the model,
        batch size, and available hardware. In order to get the most
        realistic performance distribution, example data can be provided. If
        it is not, random data will be generated based on the indicated batch
        size. After finding the optimal layout, the model will be reloaded if
        necessary. The optimal chunk size will be used to set the default chunk
        size used if none is passed to the predict call.

        Parameters
        ----------
        data
            Example data either of shape unique_batches x batch size x features
            or batch_size x features or None. If None, random data will be
            generated instead.
        batch_size : int
            If example data is not provided, random data with this many rows
            per batch will be used.
        unique_batches : int
            The number of unique batches to generate if random data are used.
            Increasing this number decreases the chance that the optimal
            configuration will be skewed by a single batch with unusual
            performance characteristics.
        timeout : float
            Time in seconds to target for optimization. The optimization loop
            will be repeatedly run a number of times increasing in the sequence
            1, 2, 5, 10, 20, 50, ... until the time taken is at least the given
            value. Note that for very large batch sizes and large models, the
            total elapsed time may exceed this timeout; it is a soft target for
            elapsed time. Setting the timeout to zero will run through the
            indicated number of unique batches exactly once. Defaults to 0.2s.
        predict_method : str
            If desired, optimization can occur over one of the prediction
            method variants (e.g. "predict_per_tree") rather than the
            default `predict` method. To do so, pass the name of the method
            here.
        max_chunk_size : int or None
            The maximum chunk size to explore during optimization. If not
            set, a value will be picked based on the current device type.
            Setting this to a lower value will reduce the optimization search
            time but may not result in optimal performance.
        seed : int
            The random seed used for generating example data if none is
            provided.
        """
        if data is None:
            xpy = GlobalSettings().fil_xpy
            dtype = self.forest.get_dtype()
            data = xpy.random.uniform(
                xpy.finfo(dtype).min / 2,
                xpy.finfo(dtype).max / 2,
                (unique_batches, batch_size, self.forest.num_features())
            )
        else:
            data = CumlArray.from_input(
                data,
                order='K',
                convert_to_mem_type=GlobalSettings().fil_memory_type,
            ).to_output('array')
        try:
            unique_batches, batch_size, features = data.shape
        except ValueError:
            unique_batches = 1
            batch_size, features = data.shape
            data = [data]

        if max_chunk_size is None:
            max_chunk_size = 512
        if get_fil_device_type() is DeviceType.device:
            max_chunk_size = min(max_chunk_size, 32)

        max_chunk_size = min(max_chunk_size, batch_size)

        infer = getattr(self, predict_method)

        optimal_layout = 'depth_first'
        optimal_chunk_size = 1

        valid_layouts = ('depth_first', 'breadth_first', 'layered')
        chunk_size = 1
        valid_chunk_sizes = []
        while chunk_size <= max_chunk_size:
            valid_chunk_sizes.append(chunk_size)
            chunk_size *= 2

        all_params = list(itertools.product(valid_layouts, valid_chunk_sizes))
        auto_iterator = _AutoIterations()
        loop_start = perf_counter()
        while True:
            optimal_time = float('inf')
            iterations = auto_iterator.next()
            for layout, chunk_size in all_params:
                self.layout = layout
                infer(data[0], chunk_size=chunk_size)
                elapsed = float('inf')
                for _ in range(iterations):
                    start = perf_counter()
                    for iter_index in range(unique_batches):
                        infer(
                            data[iter_index], chunk_size=chunk_size
                        )
                    elapsed = min(elapsed, perf_counter() - start)
                if elapsed < optimal_time:
                    optimal_time = elapsed
                    optimal_layout = layout
                    optimal_chunk_size = chunk_size
            if (perf_counter() - loop_start > timeout):
                break

        self.layout = optimal_layout
        self.default_chunk_size = optimal_chunk_size

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "treelite_model",
            "handle",
            "output_type",
            "verbose",
            "is_classifier",
            "layout",
            "default_chunk_size",
            "align_bytes",
            "precision",
            "device_id",
        ]

    def set_params(self, **params):
        super().set_params(**params)
        return self
