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
import warnings

from cuml.fil.fil import ForestInference as NewForestInference


class ForestInference(NewForestInference):
    """
    A compatibility wrapper for cuml.fil.ForestInference.
    New code should use cuml.fil.ForestInference directly.

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

    def __init__(
        self,
        *,
        treelite_model=None,
        handle=None,
        output_type=None,
        verbose=False,
        is_classifier=False,
        layout="depth_first",
        default_chunk_size=None,
        align_bytes=None,
        precision="single",
        device_id=0,
    ):
        warnings.warn(
            "cuml.experimental.fil module has been migrated to "
            "cuml.fil. Starting 25.08, cuml.experimental.fil will "
            "no longer be available. Please use cuml.fil instead.",
            FutureWarning,
        )
        super().__init__(
            treelite_model=treelite_model,
            handle=handle,
            output_type=output_type,
            verbose=verbose,
            is_classifier=is_classifier,
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
            precision=precision,
            device_id=device_id,
        )
