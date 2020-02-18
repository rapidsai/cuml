import pickle

import cudf


# all (de-)serializtion are attached to cuML Objects
serializable_classes = ()


try:
    from distributed.protocol import dask_deserialize, dask_serialize
    from distributed.protocol.cuda import cuda_deserialize, cuda_serialize
    from distributed.utils import log_errors

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
