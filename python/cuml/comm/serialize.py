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
import pickle

import cupy as cp
import cudf
import cuml


# all (de-)serializtion are attached to cuML Objects
serializable_classes = (cuml.common.CumlArray,)


def serialize_mat_descriptor(m):
    return cp.cusparse.MatDescriptor.create, ()


try:
    from distributed.protocol import dask_deserialize, dask_serialize
    from distributed.protocol.serialize import pickle_dumps, pickle_loads
    from distributed.protocol.cuda import cuda_deserialize, cuda_serialize
    from distributed.utils import log_errors

    from distributed.protocol import register_generic

    from cuml.naive_bayes.naive_bayes import MultinomialNB
    from cuml.ensemble import RandomForestRegressor
    from cuml.ensemble import RandomForestClassifier

    # Registering RF Regressor and Classifier to use pickling even when
    # Base is serialized with Dask or CUDA serializations
    @dask_serialize.register(RandomForestRegressor)
    @cuda_serialize.register(RandomForestRegressor)
    def rfr_serialize(rf):
        return pickle_dumps(rf)

    @dask_deserialize.register(RandomForestRegressor)
    @cuda_deserialize.register(RandomForestRegressor)
    def rfr_deserialize(header, frames):
        return pickle_loads(header, frames)

    @dask_serialize.register(RandomForestClassifier)
    @cuda_serialize.register(RandomForestClassifier)
    def rfc_serialize(rf):
        return pickle_dumps(rf)

    @dask_deserialize.register(RandomForestClassifier)
    @cuda_deserialize.register(RandomForestClassifier)
    def rfc_deserialize(header, frames):
        return pickle_loads(header, frames)

    register_generic(MultinomialNB, 'cuda',
                     cuda_serialize, cuda_deserialize)

    register_generic(cuml.Base, 'cuda',
                     cuda_serialize, cuda_deserialize)

    register_generic(MultinomialNB, 'dask',
                     dask_serialize, dask_deserialize)

    register_generic(cuml.Base, 'dask',
                     dask_serialize, dask_deserialize)

    copyreg.pickle(cp.cusparse.MatDescriptor, serialize_mat_descriptor)

    @cuda_serialize.register(serializable_classes)
    def cuda_serialize_cuml_object(x):
        with log_errors():
            header, frames = x.serialize()
            assert all(isinstance(f, cudf.core.buffer.Buffer) for f in frames)
            return header, frames

    # all (de-)serializtion are attached to cuML Objects
    @dask_serialize.register(serializable_classes)
    def dask_serialize_cuml_object(x):
        with log_errors():
            header, frames = x.serialize()
            frames = [f.to_host_array().data for f in frames]
            return header, frames

    @cuda_deserialize.register(serializable_classes)
    @dask_deserialize.register(serializable_classes)
    def deserialize_cuml_object(header, frames):
        with log_errors():
            cuml_typ = pickle.loads(header["type-serialized"])
            cuml_obj = cuml_typ.deserialize(header, frames)
            return cuml_obj


except ImportError:
    # distributed is probably not installed on the system
    pass
