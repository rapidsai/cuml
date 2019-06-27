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
import logging
import random
import os
import numba.cuda

from cuml.utils import device_of_gpu_matrix
from dask.distributed import wait, default_client


def get_visible_devices():
    # TODO: Shouldn't have to split on every call
    return os.environ["CUDA_VISIBLE_DEVICES"].split(",")


def device_of_devicendarray(devicendarray):
    dev = device_of_gpu_matrix(devicendarray)
    return get_visible_devices()[dev]


def get_device_id(canonical_name):
    dev_order = get_visible_devices()
    idx = 0
    for dev in dev_order:
        if dev == canonical_name:
            return idx
        idx += 1

    return -1


def select_device(dev, close=True):
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
    if '://' in address:
        address = address.rsplit('://', 1)[1]
    host, port = address.split(':')
    port = int(port)
    return host, port


def build_host_dict(workers):
    hosts = set(map(lambda x: parse_host_port(x), workers))
    hosts_dict = {}
    for host, port in hosts:
        if host not in hosts_dict:
            hosts_dict[host] = set([port])
        else:
            hosts_dict[host].add(port)

    return hosts_dict


def assign_gpus():
    client = default_client()

    """
    Supports a multi-GPU & multi-Node environment by assigning a single local
    GPU to each worker in the cluster. This is necessary due to Numba's
    restriction that only a single CUDA context (and thus a single device)
    can be active on a thread at a time.

    The GPU assignments are valid as long as the future returned from this
    function is held in scope. This allows any functions that need to allocate
    GPU data to utilize the CUDA context on the same device, otherwise data
    could be lost.
    """

    workers = list(client.has_what().keys())
    hosts_dict = build_host_dict(workers)

    print(str(hosts_dict))

    def get_gpu_info():
        import numba.cuda
        return [x.id for x in numba.cuda.gpus]

    gpu_info = dict([(host,
                      client.submit(get_gpu_info,
                                    workers=[(host,
                                              random.sample(hosts_dict[host],
                                                            1)[0])]))
                     for host in hosts_dict])
    wait(list(gpu_info.values()))

    # Scatter out a GPU device ID to workers
    f = []
    for host, future in gpu_info.items():
        gpu_ids = future.result()
        ports = random.sample(hosts_dict[host],
                              min(len(gpu_ids), len(hosts_dict[host])))

        f.extend([client.scatter(device_id, workers=[(host, port)])
                 for device_id, port in zip(gpu_ids, ports)])

    wait(f)

    return f, workers

