import cupy as cp
import numpy as np
import nvtx
import pathlib
import treelite.sklearn
import warnings
from libcpp cimport bool
from libc.stdint cimport uint32_t, uintptr_t

from cuml.internals.input_utils import input_to_cuml_array
from cuml.internals.array import CumlArray
from cuml.common.mixins import CMajorInputTagMixin
from cuml.experimental.kayak.cuda_stream cimport cuda_stream as kayak_stream_t
from cuml.experimental.kayak.device_type cimport device_type as kayak_device_t
from cuml.experimental.kayak.handle cimport handle_t as kayak_handle_t
from cuml.experimental.kayak.optional cimport optional, nullopt
from cuml.internals import set_api_output_dtype
from cuml.internals.base import UniversalBase
from cuml.internals.device_type import DeviceType
from cuml.internals.global_settings import global_settings
from cuml.internals.mem_type import MemoryType
from pylibraft.common.handle cimport handle_t as raft_handle_t

cdef extern from "treelite/c_api.h":
    ctypedef void* ModelHandle
    cdef int TreeliteLoadXGBoostModel(const char* filename,
                                      ModelHandle* out) except +
    cdef int TreeliteLoadXGBoostJSON(const char* filename,
                                     ModelHandle* out) except +
    cdef int TreeliteFreeModel(ModelHandle handle) except +
    cdef int TreeliteQueryNumTree(ModelHandle handle, size_t* out) except +
    cdef int TreeliteQueryNumFeature(ModelHandle handle, size_t* out) except +
    cdef int TreeliteQueryNumClass(ModelHandle handle, size_t* out) except +
    cdef int TreeliteLoadLightGBMModel(const char* filename,
                                       ModelHandle* out) except +
    cdef int TreeliteSerializeModel(const char* filename,
                                    ModelHandle handle) except +
    cdef int TreeliteDeserializeModel(const char* filename,
                                      ModelHandle handle) except +
    cdef const char* TreeliteGetLastError()


cdef class TreeliteModel():
    """
    Wrapper for Treelite-loaded forest

    .. note:: This is only used for loading saved models into ForestInference,
    it does not actually perform inference. Users typically do
    not need to access TreeliteModel instances directly.

    Attributes
    ----------

    handle : ModelHandle
        Opaque pointer to Treelite model
    """
    cpdef ModelHandle handle
    cpdef bool owns_handle

    def __cinit__(self, owns_handle=True):
        """If owns_handle is True, free the handle's model in destructor.
        Set this to False if another owner will free the model."""
        self.handle = <ModelHandle>NULL
        self.owns_handle = owns_handle

    cdef set_handle(self, ModelHandle new_handle):
        self.handle = new_handle

    cdef ModelHandle get_handle(self):
        return self.handle

    @property
    def handle(self):
        return <uintptr_t>(self.handle)

    def __dealloc__(self):
        if self.handle != NULL and self.owns_handle:
            TreeliteFreeModel(self.handle)

    @property
    def num_trees(self):
        assert self.handle != NULL
        cdef size_t out
        TreeliteQueryNumTree(self.handle, &out)
        return out

    @property
    def num_features(self):
        assert self.handle != NULL
        cdef size_t out
        TreeliteQueryNumFeature(self.handle, &out)
        return out

    @staticmethod
    def free_treelite_model(model_handle):
        cdef uintptr_t model_ptr = <uintptr_t>model_handle
        TreeliteFreeModel(<ModelHandle> model_ptr)

    @staticmethod
    def from_filename(filename, model_type="xgboost"):
        """
        Returns a TreeliteModel object loaded from `filename`

        Parameters
        ----------
        filename : string
            Path to treelite model file to load

        model_type : string
            Type of model: 'xgboost', 'xgboost_json', or 'lightgbm'
        """
        filename_bytes = filename.encode("UTF-8")
        cdef ModelHandle handle

        if model_type == "xgboost":
            res = TreeliteLoadXGBoostModel(filename_bytes, &handle)
        elif model_type == "xgboost_json":
            res = TreeliteLoadXGBoostJSON(filename_bytes, &handle)
        elif model_type == "lightgbm":
            res = TreeliteLoadLightGBMModel(filename_bytes, &handle)
        elif model_type == "treelite_checkpoint":
            res = TreeliteDeserializeModel(filename_bytes, &handle)
        else:
            raise ValueError("Unknown model type %s" % model_type)

        if res < 0:
            err = TreeliteGetLastError()
            raise RuntimeError("Failed to load %s (%s)" % (filename, err))
        model = TreeliteModel()
        model.set_handle(handle)
        return model

    def to_treelite_checkpoint(self, filename):
        """
        Serialize to a Treelite binary checkpoint

        Parameters
        ----------
        filename : string
            Path to Treelite binary checkpoint
        """
        assert self.handle != NULL
        filename_bytes = filename.encode("UTF-8")
        TreeliteSerializeModel(filename_bytes, self.handle)

    @staticmethod
    def from_treelite_model_handle(treelite_handle,
                                   take_handle_ownership=False):
        cdef ModelHandle handle = <ModelHandle> <size_t> treelite_handle
        model = TreeliteModel(owns_handle=take_handle_ownership)
        model.set_handle(handle)
        return model


cdef extern from "cuml/experimental/fil/forest_model.hpp" namespace "ML::experimental::fil":
    cdef cppclass forest_model:
        void predict[io_t](
            const kayak_handle_t&,
            io_t*,
            io_t*,
            size_t,
            kayak_device_t,
            kayak_device_t,
            optional[uint32_t]
        )

        bool is_double_precision()
        size_t num_outputs()

cdef extern from "cuml/experimental/fil/treelite_importer.hpp" namespace "ML::experimental::fil":
    forest_model import_from_treelite_handle(
        ModelHandle,
        uint32_t,
        optional[bool],
        kayak_device_t,
        int,
        kayak_stream_t
    )

cdef class ForestInference_impl():
    cdef forest_model model
    cdef kayak_handle_t kayak_handle
    cdef object raft_handle

    def __cinit__(
            self,
            raft_handle,
            tl_model,
            *,
            align_bytes=0,
            use_double_precision=None,
            mem_type=None,
            device_id=0):
        # Store reference to RAFT handle to control lifetime, since kayak
        # handle keeps a pointer to it
        self.raft_handle = raft_handle
        self.kayak_handle = kayak_handle_t(
            <raft_handle_t*><size_t>self.raft_handle.getHandle()
        )
        if mem_type is None:
            mem_type = global_settings.memory_type
        else:
            mem_type = MemoryType.from_str(mem_type)

        cdef optional[bool] use_double_precision_c
        if use_double_precision is None:
            use_double_precision_c = nullopt
        else:
            use_double_precision_c = use_double_precision

        try:
            model_handle = tl_model.handle
        except AttributeError:
            model_handle = tl_model

        cdef kayak_device_t dev_type
        if mem_type.is_device_accessible:
            dev_type = kayak_device_t.gpu
        else:
            dev_type = kayak_device_t.cpu

        self.model = import_from_treelite_handle(
            <ModelHandle><uintptr_t>model_handle,
            align_bytes,
            use_double_precision_c,
            dev_type,
            device_id,
            self.kayak_handle.get_next_usable_stream()
        )

    def get_dtype(self):
        return [np.float32, np.float64][self.model.is_double_precision()]

    def predict(
            self,
            X,
            *,
            preds=None,
            chunk_size=None,
            output_dtype=None):
        set_api_output_dtype(output_dtype)
        model_dtype = self.get_dtype()

        cdef uintptr_t in_ptr
        in_arr, n_rows, n_cols, dtype = input_to_cuml_array(
            X,
            order='C',
            convert_to_dtype=model_dtype,
            check_dtype=model_dtype
        )
        cdef kayak_device_t in_dev
        if in_arr.is_device_accessible:
            if (
                global_settings.device_type == DeviceType.host
                and in_arr.is_host_accessible
            ):
                in_dev = kayak_device_t.cpu
            else:
                in_dev = kayak_device_t.gpu
        else:
            in_dev = kayak_device_t.cpu

        in_ptr = in_arr.ptr

        cdef uintptr_t out_ptr
        if preds is None:
            preds = CumlArray.empty(
                (n_rows, self.model.num_outputs()),
                model_dtype,
                order='C',
                index=in_arr.index
            )
        else:
            # TODO(wphicks): Handle incorrect dtype/device/layout in C++
            preds.index = in_arr.index
        cdef kayak_device_t out_dev
        if preds.is_device_accessible:
            if (
                global_settings.device_type == DeviceType.host
                and preds.is_host_accessible
            ):
                out_dev = kayak_device_t.cpu
            else:
                out_dev = kayak_device_t.gpu
        else:
            out_dev = kayak_device_t.cpu

        out_ptr = preds.ptr

        cdef optional[uint32_t] chunk_specification
        if chunk_size is None:
            chunk_specification = nullopt
        else:
            chunk_specification = <uint32_t> chunk_size

        if model_dtype == np.float32:
            self.model.predict[float](
                self.kayak_handle,
                <float*> out_ptr,
                <float*> in_ptr,
                n_rows,
                out_dev,
                in_dev,
                chunk_specification
            )
        else:
            self.model.predict[double](
                self.kayak_handle,
                <double*> out_ptr,
                <double*> in_ptr,
                n_rows,
                in_dev,
                out_dev,
                chunk_specification
            )

        self.kayak_handle.synchronize()

        return preds

def _handle_legacy_args(
        threshold=None,
        algo=None,
        storage_type=None,
        blocks_per_sm=None,
        threads_per_tree=None,
        n_items=None,
        compute_shape_str=None):
    if threshold is not None:
        raise DeprecationWarning(
            'Parameter "threshold" has been deprecated.'
            ' To use a threshold for binary classification, pass'
            ' the "threshold" keyword directly to the predict method.'
        )
    if algo is not None:
        warnings.warn(
            'Parameter "algo" has been deprecated. Its use is no longer'
            ' necessary to achieve optimal performance with FIL.',
            DeprecationWarning
        )
    if storage_type is not None:
        warnings.warn(
            'Parameter "storage_type" has been deprecated. The correct'
            ' storage type will be used automatically.',
            DeprecationWarning
        )
    if blocks_per_sm is not None:
        warnings.warn(
            'Parameter "blocks_per_sm" has been deprecated. Its use is no'
            ' longer necessary to achieve optimal performance with FIL.',
            DeprecationWarning
        )
    if threads_per_tree is not None:
        warnings.warn(
            'Parameter "threads_per_tree" has been deprecated. Pass'
            ' the "chunk_size" keyword argument to the predict method for'
            ' equivalent functionality.',
            DeprecationWarning
        )
    if n_items is not None:
        warnings.warn(
            'Parameter "n_items" has been deprecated. Its use is no'
            ' longer necessary to achieve optimal performance with FIL.',
            DeprecationWarning
        )
    if compute_shape_str is not None:
        warnings.warn(
            'Parameter "compute_shape_str" has been deprecated.',
            DeprecationWarning
        )


class ForestInference(UniversalBase, CMajorInputTagMixin):
    """
    ForestInference provides accelerated inference for forest models on both
    CPU and GPU.
    """

    def __init__(
            self,
            *,
            treelite_model=None,
            handle=None,
            output_type=None,
            verbose=False,
            output_class=False,
            align_bytes=None,
            precision='single',
            mem_type=None,
            device_id=0):
        super().__init__(
            handle=handle, verbose=verbose, output_type=output_type
        )
        if mem_type is None:
            mem_type = global_settings.memory_type
        else:
            mem_type = MemoryType.from_str(mem_type)

        if align_bytes is None:
            self.align_bytes = 0
        else:
            self.align_bytes = align_bytes
        if precision in ('native', None):
            self.use_double_precision = None
        else:
            self.use_double_precision = (precision in ('double', 'float32'))

        self.is_classifier = output_class

        if treelite_model is not None:
            self.treelite_model = treelite_model
            self._load_to_fil(mem_type=mem_type, device_id=device_id)
        else:
            self.treelite_model = None

    def _load_to_fil(self, mem_type=None, device_id=0):
        if mem_type is None:
            mem_type = global_settings.memory_type
        else:
            mem_type = MemoryType.from_str(mem_type)

        impl = ForestInference_impl(
            self.handle,
            self.treelite_model,
            align_bytes=self.align_bytes,
            use_double_precision=self.use_double_precision,
            mem_type=mem_type,
            device_id=device_id
        )

        if mem_type.is_device_accessible:
            self._gpu_forest = impl

        if mem_type.is_host_accessible:
            self._cpu_forest = impl

    @property
    def gpu_forest(self):
        try:
            return self._gpu_forest
        except AttributeError:
            self._load_to_fil(mem_type=MemoryType.device)
            return self._gpu_forest

    @property
    def cpu_forest(self):
        try:
            return self._cpu_forest
        except AttributeError:
            self._load_to_fil(mem_type=MemoryType.host)
            return self._cpu_forest

    @property
    def forest(self):
        if global_settings.device_type == DeviceType.device:
            return self.gpu_forest
        else:
            return self.cpu_forest

    @classmethod
    def load(
            cls,
            path,
            *,
            output_class=False,
            threshold=None,
            algo=None,
            storage_type=None,
            blocks_per_sm=None,
            threads_per_tree=None,
            n_items=None,
            compute_shape_str=None,
            precision='single',
            model_type=None,
            output_type=None,
            verbose=False,
            align_bytes=None,
            mem_type=None,
            device_id=0,
            handle=None):
        _handle_legacy_args(
            threshold=threshold,
            algo=algo,
            storage_type=storage_type,
            blocks_per_sm=blocks_per_sm,
            threads_per_tree=threads_per_tree,
            n_items=n_items,
            compute_shape_str=compute_shape_str
        )
        if model_type is None:
            extension = pathlib.Path(path).suffix
            if extension == '.json':
                model_type = 'xgboost_json'
            elif extension == '.model':
                model_type = 'xgboost'
            elif extension == '.txt':
                model_type = 'lightgbm'
            else:
                model_type = 'treelite_checkpoint'
        tl_model = TreeliteModel.from_filename(path, model_type)
        return cls(
            treelite_model=tl_model,
            handle=handle,
            output_type=output_type,
            verbose=verbose,
            output_class=output_class,
            align_bytes=align_bytes,
            precision=precision,
            mem_type=mem_type,
            device_id=device_id
        )

    @classmethod
    def load_from_sklearn(
            cls,
            skl_model,
            *,
            output_class=False,
            threshold=None,
            algo=None,
            storage_type=None,
            blocks_per_sm=None,
            threads_per_tree=None,
            n_items=None,
            compute_shape_str=None,
            precision='single',
            model_type=None,
            output_type=None,
            verbose=False,
            align_bytes=None,
            mem_type=None,
            device_id=0,
            handle=None):
        _handle_legacy_args(
            threshold=threshold,
            algo=algo,
            storage_type=storage_type,
            blocks_per_sm=blocks_per_sm,
            threads_per_tree=threads_per_tree,
            n_items=n_items,
            compute_shape_str=compute_shape_str
        )
        tl_frontend_model = treelite.sklearn.import_model(skl_model)
        tl_model = TreeliteModel.from_treelite_model_handle(
            tl_frontend_model.handle.value
        )
        result = cls(
            treelite_model=tl_model,
            handle=handle,
            output_type=output_type,
            verbose=verbose,
            output_class=output_class,
            align_bytes=align_bytes,
            precision=precision,
            mem_type=mem_type,
            device_id=device_id
        )
        result._tl_frontend_model = tl_frontend_model
        return result

    @classmethod
    def load_from_treelite_model(
            cls,
            tl_model,
            *,
            output_class=False,
            threshold=None,
            algo=None,
            storage_type=None,
            blocks_per_sm=None,
            threads_per_tree=None,
            n_items=None,
            compute_shape_str=None,
            precision='single',
            model_type=None,
            output_type=None,
            verbose=False,
            align_bytes=None,
            mem_type=None,
            device_id=0,
            handle=None):
        _handle_legacy_args(
            threshold=threshold,
            algo=algo,
            storage_type=storage_type,
            blocks_per_sm=blocks_per_sm,
            threads_per_tree=threads_per_tree,
            n_items=n_items,
            compute_shape_str=compute_shape_str
        )
        return cls(
            treelite_model=tl_model,
            handle=handle,
            output_type=output_type,
            verbose=verbose,
            output_class=output_class,
            align_bytes=align_bytes,
            precision=precision,
            mem_type=mem_type,
            device_id=device_id
        )

    @nvtx.annotate(
        message='ForestInference.predict_proba',
        domain='cuml_python'
    )
    def predict_proba(self, X, *, preds=None, chunk_size=None) -> CumlArray:
        if not self.is_classifier:
            raise RuntimeError(
                "predict_proba is not available for regression models. Load"
                " with output_class=True if this is a classifier."
            )
        return self.forest.predict(X, preds=preds, chunk_size=chunk_size)

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
            threshold=None) -> CumlArray:
        proba = self.forest.predict(X, preds=preds, chunk_size=chunk_size)
        if self.is_classifier:
            if len(proba.shape) < 2 or proba.shape[1] == 1:
                if threshold is None:
                    threshold = 0.5
                return (
                    proba.to_output(output_type='array') > threshold
                ).astype('int')
            else:
                return global_settings.xpy.argmax(
                    proba.to_output(output_type='array'), axis=1
                )
        else:
            return proba
