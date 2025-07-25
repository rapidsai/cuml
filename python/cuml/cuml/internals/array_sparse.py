#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
from collections import namedtuple

import cupyx.scipy.sparse as cpx_sparse
import scipy.sparse as scipy_sparse

import cuml.internals.nvtx as nvtx
from cuml.internals.array import CumlArray
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.logger import debug
from cuml.internals.mem_type import MemoryType
from cuml.internals.memory_utils import class_with_cupy_rmm

sparse_matrix_classes = (
    cpx_sparse.csr_matrix,
    scipy_sparse.csr_matrix,
)

SparseCumlArrayInput = namedtuple(
    "SparseCumlArrayInput",
    ["indptr", "indices", "data", "nnz", "dtype", "shape"],
)


@class_with_cupy_rmm()
class SparseCumlArray:
    """
    SparseCumlArray abstracts sparse array allocations. This will
    accept either a Scipy or Cupy sparse array and construct CumlArrays
    out of the underlying index and data arrays. Currently, this class
    only supports the CSR array format and input in any other sparse
    format will be converted to CSR by default. Set `convert_format=False`
    to disable automatic conversion to CSR.

    Parameters
    ----------

    data : scipy.sparse.spmatrix or cupyx.scipy.sparse.spmatrix
        A Scipy or Cupy sparse matrix
    convert_to_dtype : data-type or False, optional
        Any object that can be interpreted as a numpy or cupy data type.
        Specifies whether to convert the data array to a different dtype.
    convert_index : data-type or False (default: np.int32), optional
        Any object that can be interpreted as a numpy or cupy data type.
        Specifies whether to convert the indices to a different dtype. By
        default, it is preferred to use 32-bit indexing.
    convert_format : bool, optional (default: True)
        Specifies whether to convert any non-CSR inputs to CSR. If False,
        an exception is thrown.


    Attributes
    ----------

    indptr : CumlArray
        Compressed row index array
    indices : CumlArray
        Column indices array
    data : CumlArray
        Data array
    dtype : dtype
        Data type of data array
    shape : tuple of ints
        Shape of the array
    nnz : int
        Number of nonzeros in underlying arrays
    """

    @nvtx.annotate(
        message="common.SparseCumlArray.__init__",
        category="utils",
        domain="cuml_python",
    )
    def __init__(
        self,
        data=None,
        convert_to_dtype=False,
        convert_to_mem_type=None,
        convert_index=None,
        convert_format=True,
    ):
        if not isinstance(data, SparseCumlArrayInput):
            if cpx_sparse.isspmatrix(data):
                from_mem_type = MemoryType.device
            elif scipy_sparse.isspmatrix(data):
                from_mem_type = MemoryType.host
            else:
                raise ValueError(
                    "A sparse matrix is expected as input. "
                    "Received %s" % type(data)
                )

            if not isinstance(data, sparse_matrix_classes):
                if convert_format:
                    debug(
                        "Received sparse matrix in {} format but CSR is "
                        "expected. Data will be converted to CSR, but this "
                        "will require additional memory copies. If this "
                        "conversion is not desired, set "
                        "set_convert_format=False to raise an exception "
                        "instead.".format(type(data))
                    )
                    data = data.tocsr()  # currently only CSR is supported
                else:
                    raise ValueError(
                        "Expected CSR matrix but received {}".format(
                            type(data)
                        )
                    )

        if not convert_to_dtype:
            convert_to_dtype = data.dtype

        if convert_to_mem_type:
            convert_to_mem_type = MemoryType.from_str(convert_to_mem_type)
        else:
            convert_to_mem_type = GlobalSettings().memory_type

        if convert_to_mem_type is MemoryType.mirror or not convert_to_mem_type:
            convert_to_mem_type = from_mem_type

        self._mem_type = convert_to_mem_type

        if convert_index is None:
            convert_index = GlobalSettings().xpy.int32
        if not convert_index:
            convert_index = data.indptr.dtype

        # Note: Only 32-bit indexing is supported currently.
        # Since CUDA11, Cusparse provides 64-bit function calls
        # but these are not yet used in RAFT/Cuml
        self.indptr = CumlArray.from_input(
            data.indptr,
            convert_to_dtype=convert_index,
            convert_to_mem_type=convert_to_mem_type,
        )

        self.indices = CumlArray.from_input(
            data.indices,
            convert_to_dtype=convert_index,
            convert_to_mem_type=convert_to_mem_type,
        )

        self.data = CumlArray.from_input(
            data.data,
            convert_to_dtype=convert_to_dtype,
            convert_to_mem_type=convert_to_mem_type,
        )

        self.shape = data.shape
        self.dtype = self.data.dtype
        self.nnz = data.nnz
        self.index = None

    @nvtx.annotate(
        message="common.SparseCumlArray.to_output",
        category="utils",
        domain="cuml_python",
    )
    def to_output(
        self,
        output_type="cupy",
        output_format=None,
        output_dtype=None,
        output_mem_type=None,
    ):
        """
        Convert array to output format

        Parameters
        ----------
        output_type : string
            Format to convert the array to. Acceptable formats are:

            - 'cupy' - to cupy array
            - 'scipy' - to scipy (host) array
            - 'numpy' - to scipy (host) array
            - 'array' - to cupy or scipy array depending on
              output_mem_type

        output_format : string, optional { 'coo', 'csc' }
            Optionally convert the output to the specified format.
        output_dtype : string, optional
            Optionally cast the array to a specified dtype, creating
            a copy if necessary.
        output_mem_type : {'host, 'device'}, optional
            Optionally convert array to given memory type. If `output_type`
            already indicates a specific memory type, `output_type` takes
            precedence. If the memory type is not otherwise indicated, the data
            are kept on their current device.
        """
        if output_mem_type is None:
            output_mem_type = GlobalSettings().memory_type
        else:
            output_mem_type = MemoryType.from_str(output_mem_type)
        # Treat numpy and scipy as the same
        if output_type in ("numpy", "scipy"):
            if GlobalSettings().memory_type.is_host_accessible:
                output_mem_type = GlobalSettings().memory_type
            else:
                output_mem_type = MemoryType.host
        elif output_type == "cupy":
            if GlobalSettings().memory_type.is_device_accessible:
                output_mem_type = GlobalSettings().memory_type
            else:
                output_mem_type = MemoryType.device
        elif output_mem_type is MemoryType.mirror:
            output_mem_type = self.mem_type

        data = self.data.to_output(
            "array", output_dtype=output_dtype, output_mem_type=output_mem_type
        )
        indices = self.indices.to_output(
            "array", output_mem_type=output_mem_type
        )
        indptr = self.indptr.to_output(
            "array", output_mem_type=output_mem_type
        )

        if output_type in ("scipy", "numpy"):
            constructor = scipy_sparse.csr_matrix
        elif output_mem_type.is_device_accessible:
            constructor = cpx_sparse.csr_matrix
        else:
            constructor = scipy_sparse.csr_matrix

        ret = constructor(
            (data, indices, indptr), dtype=output_dtype, shape=self.shape
        )

        if output_format is not None:
            if output_format == "coo":
                ret = ret.tocoo()
            elif output_format == "csc":
                ret = ret.tocsc()
            else:
                raise ValueError(
                    "Output format %s not supported" % output_format
                )

        return ret
