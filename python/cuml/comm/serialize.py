#
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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


import cuml
import cudf.comm.serialize  # noqa: F401


try:
    from distributed.protocol import dask_deserialize, dask_serialize
    from distributed.protocol.serialize import pickle_dumps, pickle_loads
    from distributed.protocol.cuda import cuda_deserialize, cuda_serialize

    from distributed.protocol import register_generic

    from cuml.internals.array_sparse import SparseCumlArray

    from cuml.ensemble import RandomForestRegressor
    from cuml.ensemble import RandomForestClassifier

    from cuml.naive_bayes import MultinomialNB

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

    register_generic(SparseCumlArray, "cuda", cuda_serialize, cuda_deserialize)

    register_generic(SparseCumlArray, "dask", dask_serialize, dask_deserialize)

    register_generic(cuml.Base, "cuda", cuda_serialize, cuda_deserialize)

    register_generic(cuml.Base, "dask", dask_serialize, dask_deserialize)

    register_generic(MultinomialNB, "cuda", cuda_serialize, cuda_deserialize)

    register_generic(MultinomialNB, "dask", dask_serialize, dask_deserialize)

except ImportError:
    # distributed is probably not installed on the system
    pass
