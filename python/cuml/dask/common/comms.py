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

import time
from collections import OrderedDict
import warnings

from dask.distributed import get_worker

from .comms_utils import inject_comms_on_handle, \
    inject_comms_on_handle_coll_only
from cuml.common.handle import Handle

from cuml.raft.dask.common.comms import Comms
from cuml.raft.dask.common.comms import get_ucx
from cuml.raft.dask.common.nccl import nccl
from cuml.raft.dask.common.comms import _func_init_nccl
from cuml.raft.dask.common.comms import _func_ucp_create_endpoints


async def _func_init_all(sessionId, uniqueId, comms_p2p,
                         worker_info, verbose, streams_per_handle):

    session_state = worker_state(sessionId)
    session_state["nccl_uid"] = uniqueId
    session_state["wid"] = worker_info[get_worker().address]["rank"]
    session_state["nworkers"] = len(worker_info)

    if verbose:
        print("Initializing NCCL")
        start = time.time()

    _func_init_nccl(sessionId, uniqueId)

    if verbose:
        elapsed = time.time() - start
        print("NCCL Initialization took: %f seconds." % elapsed)

    if comms_p2p:
        if verbose:
            print("Initializing UCX Endpoints")

        if verbose:
            start = time.time()
        await _func_ucp_create_endpoints(sessionId, worker_info)

        if verbose:
            elapsed = time.time() - start
            print("Done initializing UCX endpoints. Took: %f seconds." %
                  elapsed)
            print("Building handle")

        _func_build_handle_p2p(sessionId, streams_per_handle, verbose)

        if verbose:
            print("Done building handle.")

    else:
        _func_build_handle(sessionId, streams_per_handle, verbose)


def _func_build_handle_p2p(sessionId, streams_per_handle, verbose):
    """
    Builds a handle_t on the current worker given the initialized comms
    Parameters
    ----------
    sessionId : str id to reference state for current comms instance.
    streams_per_handle : int number of internal streams to create
    verbose : bool print verbose logging output
    """
    ucp_worker = get_ucx().get_worker()
    session_state = worker_state(sessionId)

    handle = Handle(streams_per_handle)
    nccl_comm = session_state["nccl"]
    eps = session_state["ucp_eps"]
    nWorkers = session_state["nworkers"]
    workerId = session_state["wid"]

    inject_comms_on_handle(handle, nccl_comm, ucp_worker, eps,
                           nWorkers, workerId)

    worker_state(sessionId)["handle"] = handle


def _func_build_handle(sessionId, streams_per_handle, verbose):
    """
    Builds a handle_t on the current worker given the initialized comms
    Parameters
    ----------
    sessionId : str id to reference state for current comms instance.
    streams_per_handle : int number of internal streams to create
    verbose : bool print verbose logging output
    """
    handle = Handle(streams_per_handle)

    session_state = worker_state(sessionId)

    workerId = session_state["wid"]
    nWorkers = session_state["nworkers"]

    nccl_comm = session_state["nccl"]
    inject_comms_on_handle_coll_only(handle, nccl_comm, nWorkers,
                                     workerId)
    session_state["handle"] = handle


def worker_state(sessionId=None):
    """
    Retrieves cuML comms state on local worker for the given
    sessionId, creating a new session if it does not exist.
    If no session id is given, returns the state dict for all
    sessions.
    :param sessionId:
    :return:
    """
    worker = get_worker()
    if not hasattr(worker, "_raft_comm_state"):
        worker._raft_comm_state = {}
    if sessionId is not None and sessionId not in worker._raft_comm_state:
        # Build state for new session and mark session creation time
        worker._raft_comm_state[sessionId] = {"ts": time.time()}

    if sessionId is not None:
        return worker._raft_comm_state[sessionId]
    return worker._raft_comm_state


class CommsContext(Comms):

    def __init__(self, comms_p2p=False, client=None, verbose=False,
                 streams_per_handle=0):
        super().__init__(comms_p2p=comms_p2p,
                         client=client,
                         verbose=verbose,
                         streams_per_handle=streams_per_handle)

    def init(self, workers=None):
        """
        Initializes the underlying comms. NCCL is required but
        UCX is only initialized if `comms_p2p == True`
        Parameters
        ----------
        workers : Sequence
                  Unique collection of workers for initializing comms.
        """

        self.worker_addresses = list(OrderedDict.fromkeys(
            self.client.scheduler_info()["workers"].keys()
            if workers is None else workers))

        if self.nccl_initialized or self.ucx_initialized:
            warnings.warn("Comms have already been initialized.")
            return

        worker_info = self.worker_info(self.worker_addresses)
        worker_info = {w: worker_info[w] for w in self.worker_addresses}

        self.uniqueId = nccl.get_unique_id()

        self.client.run(_func_init_all,
                        self.sessionId,
                        self.uniqueId,
                        self.comms_p2p,
                        worker_info,
                        self.verbose,
                        self.streams_per_handle,
                        workers=self.worker_addresses,
                        wait=True)

        self.nccl_initialized = True

        if self.comms_p2p:
            self.ucx_initialized = True

        if self.verbose:
            print("Initialization complete.")
