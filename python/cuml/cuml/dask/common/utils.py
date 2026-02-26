# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import dask
from dask.distributed import default_client, wait


def get_client(client=None):
    return default_client() if client is None else client


def parse_host_port(address):
    """
    Given a string address with host/port, build a tuple(host, port)
    :param address: string address to parse
    :return: tuple(host, port)
    """
    if "://" in address:
        address = address.rsplit("://", 1)[1]
    host, port = address.split(":")
    port = int(port)
    return host, port


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

    with dask.annotate(workers=set(workers)):
        return client.persist(objects)


def raise_exception_from_futures(futures):
    """Raises a RuntimeError if any of the futures indicates an exception"""
    errs = [f.exception() for f in futures if f.exception()]
    if errs:
        raise RuntimeError(
            "%d of %d worker jobs failed: %s"
            % (len(errs), len(futures), ", ".join(map(str, errs)))
        )


def wait_and_raise_from_futures(futures):
    """
    Returns the collected futures after all the futures
    have finished and do not indicate any exceptions.
    """
    wait(futures)
    raise_exception_from_futures(futures)
    return futures


def raise_mg_import_exception():
    raise Exception(
        "cuML has not been built with multiGPU support "
        "enabled. Build with the --multigpu flag to"
        " enable multiGPU support."
    )
