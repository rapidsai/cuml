#
# Copyright (c) 2020, NVIDIA CORPORATION.
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

from cuml.common.import_utils import has_scipy

import cupyx as cpx
import numpy as np

from cuml.common.memory_utils import with_cupy_rmm

from cuml.common.input_utils import input_to_cuml_array

if has_scipy():
    import scipy.sparse


class SparseCumlArray:

    @with_cupy_rmm
    def __init__(self, data=None, dtype=None):
        """
        SparseCumlArray abstracts sparse array allocations. This will
        accept either a Scipy or Cupy sparse array and construct CumlArrays
        out of the underlying index and data arrays. Currently, this class
        only supports the CSR array format and input in any other sparse
        format will be converted to CSR.

        Parameters
        ----------

        data : scipy.sparse.csr_matrix or cupyx.scipy.sparse.csr_matrix
            A Scipy or Cupy sparse csr_matrix
        dtype : data-type, optional
            Any object that can be interpreted as a numpy or cupy data type.

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

        if not cpx.scipy.sparse.isspmatrix(data) and \
                not (has_scipy() and scipy.sparse.isspmatrix(data)):
            raise ValueError("A sparse matrix is expected as input. "
                             "Received %s" % type(data))

        data = data.tocsr()  # currently only CSR is supported

        # Note: Only 32-bit indexing is supported currently.
        # In CUDA11, Cusparse provides 64-bit function calls
        # but these are not yet used in RAFT/Cuml
        self.indptr, _, _, _ = input_to_cuml_array(data.indptr,
                                                   check_dtype=False,
                                                   convert_to_dtype=np.int32)

        self.indices, _, _, _ = input_to_cuml_array(data.indices,
                                                    check_dtype=False,
                                                    convert_to_dtype=np.int32)

        self.data, _, _, _ = input_to_cuml_array(data.data,
                                                 check_dtype=False,
                                                 convert_to_dtype=dtype)

        self.shape = data.shape
        self.dtype = self.data.dtype
        self.nnz = data.nnz

    @with_cupy_rmm
    def to_output(self, output_type='cupy', output_dtype=None):
        """
        Convert array to output format

        Parameters
        ----------
        output_type : string
            Format to convert the array to. Acceptable formats are:

            - 'cupy' - to cupy array
            - 'scipy' - to scipy (host) array

        output_dtype : string, optional
            Optionally cast the array to a specified dtype, creating
            a copy if necessary.
        """
        output_dtype = self.data.dtype \
            if output_dtype is None else output_dtype

        if output_type not in ['cupy', 'scipy']:
            raise ValueError("Unsupported output_type: %s" % output_dtype)

        cuml_arr_output_type = 'numpy' if output_type == 'scipy' else 'cupy'

        data = self.data.to_output(cuml_arr_output_type, output_dtype)
        indices = self.indices.to_output(cuml_arr_output_type)
        indptr = self.indptr.to_output(cuml_arr_output_type)

        if output_type == 'cupy':
            constructor = cpx.scipy.sparse.csr_matrix

        elif output_type == 'scipy' and has_scipy():
            if has_scipy():
                constructor = scipy.sparse.csr_matrix
            else:
                raise ValueError("Scipy library is not available.")

        return constructor((data, indices, indptr),
                           dtype=output_dtype, shape=self.shape)
