#
# Copyright (c) 2025, NVIDIA CORPORATION.
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

from cuml.experimental.fil.fil import (
    ForestInference as ExperimentalForestInference,
)
from cuml.internals.array import CumlArray
from cuml.internals.global_settings import GlobalSettings


class ForestInference(ExperimentalForestInference):
    def __init__(
        self,
        *,
        treelite_model=None,
        handle=None,
        output_type=None,
        verbose=False,
        is_classifier=False,
        output_class=None,
        layout="depth_first",
        default_chunk_size=None,
        align_bytes=None,
        precision="single",
        device_id=0,
    ):
        """
        ForestInference provides accelerated inference for forest models on both
        CPU and GPU.

        Parameters
        ----------
        treelite_model : treelite.Model
            The model to be used for inference. This can be trained with XGBoost,
            LightGBM, cuML, Scikit-Learn, or any other forest model framework
            so long as it can be loaded into a treelite.Model object (See
            https://treelite.readthedocs.io/en/latest/treelite-api.html).
        handle : pylibraft.common.handle or None
            For GPU execution, the RAFT handle containing the stream or stream
            pool to use during loading and inference. If input is provide to
            this model in the wrong memory location (e.g. host memory input but
            GPU execution), the input will be copied to the correct location
            using as many streams as are available in the handle. It is therefore
            recommended that a handle with a stream pool be used for models where
            it is expected that large input arrays will be coming from the host but
            evaluated on device.
        output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
            'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
            Return results and set estimator attributes to the indicated output
            type. If None, the output type set at the module level
            (`cuml.global_settings.output_type`) will be used. See
            :ref:`output-data-type-configuration` for more info.
        verbose : int or boolean, default=False
            Sets logging level. It must be one of `cuml.common.logger.level_*`.
            See :ref:`verbosity-levels` for more info.
        is_classifier : boolean
            True for classifier models, False for regressors.
        output_class : boolean
            Deprecated parameter. Please use is_classifier instead.
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
        device_id : int, default=0
            For GPU execution, the device on which to load and execute this
            model. For CPU execution, this value is currently ignored.
        """
        super().__init__(
            treelite_model=treelite_model,
            handle=handle,
            output_type=output_type,
            verbose=verbose,
            is_classifier=is_classifier,
            output_class=output_class,
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
            precision=precision,
            device_id=device_id,
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
            (as set with e.g. the `using_device_type` context manager),
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
        results = super().predict_proba(X, preds=preds, chunk_size=chunk_size)
        if len(results.shape) == 2 and results.shape[-1] == 1:
            results = results.to_output("array").flatten()
            results = GlobalSettings().xpy.stack(
                [1 - results, results], axis=1
            )
        return results

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
            (as set with e.g. the `using_device_type` context manager),
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
        results = super().predict(
            X, preds=preds, chunk_size=chunk_size, threshold=threshold
        )
        if (
            self.is_classifier
            and len(results.shape) == 2
            and results.shape[-1] == 1
        ):
            results = results.to_output("array").flatten()
        return results
