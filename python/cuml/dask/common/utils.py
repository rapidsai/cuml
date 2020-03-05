# Copyright (c) 2019, NVIDIA CORPORATION.
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

import logging
import os
import numba.cuda

from cuml.naive_bayes.naive_bayes import MultinomialNB
from cuml.utils import device_of_gpu_matrix

from distributed.protocol.cuda import cuda_deserialize
from distributed.protocol.cuda import cuda_serialize
from distributed.protocol.serialize import dask_deserialize
from distributed.protocol.serialize import dask_serialize
from distributed.protocol.serialize import register_generic


def get_visible_devices():
    """
    Return a list of the CUDA_VISIBLE_DEVICES
    :return: list[int] visible devices
    """
    # TODO: Shouldn't have to split on every call
    return os.environ["CUDA_VISIBLE_DEVICES"].split(",")


def device_of_devicendarray(devicendarray):
    """
    Returns the device that backs memory allocated on the given
    deviceNDArray
    :param devicendarray: devicendarray array to check
    :return: int device id
    """
    dev = device_of_gpu_matrix(devicendarray)
    return get_visible_devices()[dev]


def get_device_id(canonical_name):
    """
    Given a local device id, find the actual "global" id
    :param canonical_name: the local device name in CUDA_VISIBLE_DEVICES
    :return: the global device id for the system
    """
    dev_order = get_visible_devices()
    idx = 0
    for dev in dev_order:
        if dev == canonical_name:
            return idx
        idx += 1

    return -1


def select_device(dev, close=True):
    """
    Use numbas numba to select the given device, optionally
    closing and opening up a new cuda context if it fails.
    :param dev: int device to select
    :param close: bool close the cuda context and create new one?
    """
    if numba.cuda.get_current_device().id != dev:
        logging.warn("Selecting device " + str(dev))
        if close:
            numba.cuda.close()
        numba.cuda.select_device(dev)
        if dev != numba.cuda.get_current_device().id:
            logging.warn("Current device " +
                         str(numba.cuda.get_current_device()) +
                         " does not match expected " + str(dev))


def parse_host_port(address):
    """
    Given a string address with host/port, build a tuple(host, port)
    :param address: string address to parse
    :return: tuple(host, port)
    """
    if '://' in address:
        address = address.rsplit('://', 1)[1]
    host, port = address.split(':')
    port = int(port)
    return host, port


def build_host_dict(workers):
    """
    Builds a dict to map the set of ports running on each host to
    the hostname.
    :param workers: list(tuple(host, port)) list of worker addresses
    :return: dict(host, set(port))
    """
    hosts = set(map(lambda x: parse_host_port(x), workers))
    hosts_dict = {}
    for host, port in hosts:
        if host not in hosts_dict:
            hosts_dict[host] = set([port])
        else:
            hosts_dict[host].add(port)

    return hosts_dict


def persist_across_workers(client, objects, workers=None):
    """
    Calls persist on the 'objects' ensuring they are spread
    across the workers on 'workers'.

    Parameters
    ----------
    client : dask.distributed.Client
    objects : list
        Dask distributed objects to be persisted
    workers : list or None
        List of workers across which to persist objects
        If None, then all workers attached to 'client' will be used
    """
    if workers is None:
        workers = client.has_what().keys()  # Default to all workers
    return client.persist(objects, workers={o: workers for o in objects})


def raise_exception_from_futures(futures):
    """Raises a RuntimeError if any of the futures indicates an exception"""
    errs = [f.exception() for f in futures if f.exception()]
    if errs:
        raise RuntimeError("%d of %d worker jobs failed: %s" % (
            len(errs), len(futures), ", ".join(map(str, errs))
            ))


def raise_mg_import_exception():
    raise Exception("cuML has not been built with multiGPU support "
                    "enabled. Build with the --multigpu flag to"
                    " enable multiGPU support.")


def register_serialization(client):
    """
    This function provides a temporary fix for a bug
    in CuPy that doesn't properly serialize cuSPARSE handles.

    Reference: https://github.com/cupy/cupy/issues/3061

    Parameters
    ----------

    client : dask.distributed.Client client to use
    """
    def patch_func():
        def serialize_mat_descriptor(m):
            return cp.cupy.cusparse.MatDescriptor.create, ()

        register_generic(MultinomialNB, "cuda",
                         cuda_serialize, cuda_deserialize)
        register_generic(MultinomialNB, "dask",
                         dask_serialize, dask_deserialize)

        copyreg.pickle(cp.cupy.cusparse.MatDescriptor,
                       serialize_mat_descriptor)

    patch_func()
    client.run(patch_func)
