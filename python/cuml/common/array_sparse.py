#
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
import cupyx as cpx
import numpy as np
import nvtx
from cuml.common.import_utils import has_scipy
from cuml.common.memory_utils import class_with_cupy_rmm
from cuml.common.logger import debug

import cuml.common

if has_scipy():
    import scipy.sparse


@class_with_cupy_rmm()
class SparseCumlArray():
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

    @nvtx.annotate(message="common.SparseCumlArray.__init__", category="utils",
                   domain="cuml_python")
    def __init__(self, data=None,
                 convert_to_dtype=False,
                 convert_index=np.int32,
                 convert_format=True):
        if not cpx.scipy.sparse.isspmatrix(data) and \
                not (has_scipy() and scipy.sparse.isspmatrix(data)):
            raise ValueError("A sparse matrix is expected as input. "
                             "Received %s" % type(data))

        check_classes = [cpx.scipy.sparse.csr_matrix]
        if has_scipy():
            check_classes.append(scipy.sparse.csr_matrix)

        if not isinstance(data, tuple(check_classes)):
            if convert_format:
                debug('Received sparse matrix in %s format but CSR is '
                      'expected. Data will be converted to CSR, but this '
                      'will require additional memory copies. If this '
                      'conversion is not desired, set '
                      'set_convert_format=False to raise an exception '
                      'instead.' % type(data))
                data = data.tocsr()  # currently only CSR is supported
            else:
                raise ValueError("Expected CSR matrix but received %s"
                                 % type(data))

        if not convert_to_dtype:
            convert_to_dtype = data.dtype

        if not convert_index:
            convert_index = data.indptr.dtype

        # Note: Only 32-bit indexing is supported currently.
        # In CUDA11, Cusparse provides 64-bit function calls
        # but these are not yet used in RAFT/Cuml
        self.indptr, _, _, _ = cuml.common.input_to_cuml_array(
            data.indptr, convert_to_dtype=convert_index)

        self.indices, _, _, _ = cuml.common.input_to_cuml_array(
            data.indices, convert_to_dtype=convert_index)

        self.data, _, _, _ = cuml.common.input_to_cuml_array(
            data.data, convert_to_dtype=convert_to_dtype)

        self.shape = data.shape
        self.dtype = self.data.dtype
        self.nnz = data.nnz
        self.index = None

    @nvtx.annotate(message="common.SparseCumlArray.to_output",
                   category="utils", domain="cuml_python")
    def to_output(self, output_type='cupy',
                  output_format=None,
                  output_dtype=None):
        """
        Convert array to output format

        Parameters
        ----------
        output_type : string
            Format to convert the array to. Acceptable formats are:

            - 'cupy' - to cupy array
            - 'scipy' - to scipy (host) array
            - 'numpy' - to scipy (host) array

        output_format : string, optional { 'coo', 'csc' }
            Optionally convert the output to the specified format.
        output_dtype : string, optional
            Optionally cast the array to a specified dtype, creating
            a copy if necessary.
        """
        # Treat numpy and scipy as the same
        if (output_type == "numpy"):
            output_type = "scipy"

        output_dtype = self.data.dtype \
            if output_dtype is None else output_dtype

        if output_type not in ['cupy', 'scipy']:
            # raise ValueError("Unsupported output_type: %s" % output_type)
            # Default if output_type doesn't support sparse arrays
            output_type = 'cupy'

        cuml_arr_output_type = 'numpy' \
            if output_type in ('scipy', 'numpy') else 'cupy'

        data = self.data.to_output(cuml_arr_output_type, output_dtype)
        indices = self.indices.to_output(cuml_arr_output_type)
        indptr = self.indptr.to_output(cuml_arr_output_type)

        if output_type == 'cupy':
            constructor = cpx.scipy.sparse.csr_matrix
        elif output_type == 'scipy' and has_scipy(raise_if_unavailable=True):
            constructor = scipy.sparse.csr_matrix
        else:
            raise ValueError("Unsupported output_type: %s" % output_type)

        ret = constructor((data, indices, indptr),
                          dtype=output_dtype, shape=self.shape)

        if output_format is not None:
            if output_format == 'coo':
                ret = ret.tocoo()
            elif output_format == 'csc':
                ret = ret.tocsc()
            else:
                raise ValueError("Output format %s not supported"
                                 % output_format)

        return ret
