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
import os
import numba.cuda

from cuml.utils import device_of_gpu_matrix


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
