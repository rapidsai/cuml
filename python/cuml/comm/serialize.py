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


import copyreg

import cupy as cp
import cuml
import cudf.comm.serialize  # noqa: F401


def serialize_mat_descriptor(m):
    return cp.cusparse.MatDescriptor.create, ()


try:
    from distributed.protocol import dask_deserialize, dask_serialize
    from distributed.protocol.cuda import cuda_deserialize, cuda_serialize

    from distributed.protocol import register_generic

    from cuml.naive_bayes.naive_bayes import MultinomialNB

    register_generic(MultinomialNB, 'cuda',
                     cuda_serialize, cuda_deserialize)

    register_generic(cuml.Base, 'cuda',
                     cuda_serialize, cuda_deserialize)

    register_generic(MultinomialNB, 'dask',
                     dask_serialize, dask_deserialize)

    register_generic(cuml.Base, 'dask',
                     dask_serialize, dask_deserialize)

    copyreg.pickle(cp.cusparse.MatDescriptor, serialize_mat_descriptor)


except ImportError:
    # distributed is probably not installed on the system
    pass
