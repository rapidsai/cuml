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

from cuml.nccl import nccl

import threading

import weakref

from .comms_utils import inject_comms_on_handle, \
    inject_comms_on_handle_coll_only, is_ucx_enabled
from .utils import parse_host_port
from cuml.common.handle import Handle

from dask.distributed import wait, get_worker, default_client

from cuml.utils.import_utils import has_ucp
import warnings

import time

import random
import asyncio
import uuid


_global_comms = weakref.WeakValueDictionary()
_global_comms_index = [0]


def _set_global_comms(c):
    if c is not None:
        _global_comms[_global_comms_index[0]] = c
        _global_comms_index[0] += 1


def _del_global_comms(c):
    for k in list(_global_comms):
        try:
            if _global_comms[k] is c:
                del _global_comms[k]
        except KeyError:
            pass


if is_ucx_enabled() and has_ucp():
    import ucp


async def connection_func(ep, listener):
    pass


def worker_state(sessionId = None):
    """
    Retrieves cuML comms state on local worker for the given
    sessionId, creating a new session if it does not exist.
    If no session id is given, returns the state dict for all
    sessions.
    :param sessionId:
    :return:
    """
    worker = get_worker()
    if not hasattr(worker, "_cuml_comm_state"):
        worker._cuml_comm_state = {}
    if sessionId is not None and sessionId not in worker._cuml_comm_state:
        # Build state for new session and mark session creation time
        worker._cuml_comm_state[sessionId] = {"ts": time.time()}

    if sessionId is not None:
        return worker._cuml_comm_state[sessionId]
    return worker._cuml_comm_state


def _get_global_comms():
    L = sorted(list(_global_comms), reverse=True)
    for k in L:
        c = _global_comms[k]
        if c.nccl_initialized and (not c.comms_p2p or c.ucx_initialized):
            return c
        else:
            del _global_comms[k]
    del L
    return None


def _del_default_comms(ref):

    print("Deleting default_comms")
    global _default_comms
    _default_comms = None


def default_comms(c=None):
    """ Return a comms instance if one has been initialized """
    c = c or _get_global_comms()
    if c:
        return c
    else:
        raise ValueError(
            "No comms instances found\n"
            "Start a comms instance and initialize it\n"
        )



class CommsBase:

    """
    A base class to initialize and manage underlying NCCL and UCX
    comms handles across a Dask cluster. Classes extending CommsBase
    are responsible for calling `self.init()` to initialize the comms.
    Classes that extend or use the CommsBase are also responsible for
    calling `destroy()` to clean up the underlying comms.

    This class is not meant to be thread-safe and each instance
    """

    def __init__(self, comms_p2p=False, client = None):
        """
        Construct a new BaseComms instance
        :param comms_p2p: bool Should p2p comms be initialized?
        """
        self.client = client if client is not None else default_client()
        self.comms_p2p = comms_p2p

        self.sessionId = uuid.uuid4().bytes

        self.worker_addresses = self.get_workers_()
        self.workers = list(map(lambda x: parse_host_port(x),
                                self.worker_addresses))

        self.nccl_initialized = False
        self.ucx_initialized = False

        if comms_p2p and (not is_ucx_enabled() or not has_ucp()):
            warnings.warn("ucx-py not found. UCP Integration will "
                          "be disabled.")
            self.comms_p2p = False

    def get_workers_(self):
        """
        Return the list of workers parsed as [(address, port)]
        """
        return list(self.client.has_what().keys())

    def worker_ranks(self):
        """
        Builds a dictionary of { (worker_address, worker_port) : worker_rank }
        """
        return dict(list(zip(self.workers, range(len(self.workers)))))

    @staticmethod
    def func_ucp_listener_port(sessionId, r):
        return worker_state(sessionId)["ucp_listener"].port

    def ucp_ports(self):
        return [(w, self.client.submit(CommsBase.func_ucp_listener_port,
                                       self.sessionId,
                                       random.random(),
                                       workers=[w]).result()) for w in self.workers]

    def worker_ports(self):
        """
        Builds a dictionary of { (worker_address, worker_port) : worker_port }
        """
        return dict(list(self.ucp_ports()))

    def worker_info(self):
        """
        Builds a dictionary of { (worker_address, worker_port) :
                                (worker_rank, worker_port ) }
        """
        ranks = self.worker_ranks()
        ports = self.worker_ports() if self.comms_p2p else None

        if self.comms_p2p:
            output = {}
            for k in self.worker_ranks().keys():
                output[k] = (ranks[k], ports[k])
            return output

        else:
            return ranks

    @staticmethod
    async def func_init_nccl(sessionId):
        """
        Initialize ncclComm_t on worker
        :param workerId: int ID of the current worker running the function
        :param nWorkers: int Number of workers in the cluster
        :param uniqueId: array[byte] The NCCL unique Id generated from the
                         client.
        """
        wid = worker_state(sessionId)["wid"]
        uniqueId = worker_state(sessionId)["nccl_uid"]
        nWorkers = worker_state(sessionId)["nworkers"]

        n = nccl()
        n.init(nWorkers, uniqueId, wid)
        worker_state(sessionId)["nccl"] = n

    @staticmethod
    async def func_ucp_create_listener(sessionId, r):
        """
        Creates a UCP listener for incoming endpoint connections.
        This function runs in a loop asynchronously in the background
        on the worker
        :param sessionId: uuid Unique id for current instance
        :param r: float a random number to stop the function from being cached
        """
        if "ucp_listener" in worker_state(sessionId):
            print("Listener already started for sessionId=" +
                  str(sessionId))
        else:
            ucp.init()
            listener = ucp.start_listener(connection_func, 0,
                                          is_coroutine=True)

            worker_state(sessionId)["ucp_listener"] = listener
            task = asyncio.create_task(listener.coroutine)

            while not task.done():
                await task
                await asyncio.sleep(1)

            ucp.fin()

    @staticmethod
    async def func_ucp_stop_listener(sessionId):
        """
        Stops the listener running in the background on the current worker.
        :param sessionId: uuid Unique id for current instance
        :param r: float a random number to stop the function from being cached
        """
        if "ucp_listener" in worker_state(sessionId):
            listener = worker_state(sessionId)["ucp_listener"]
            ucp.stop_listener(listener)

            del worker_state(sessionId)["ucp_listener"]
        else:
            print("Listener not found with sessionId=" + str(sessionId))

    def create_ucp_listeners(self):
        """
        Build a UCP listener on each worker. Since this async
        function is long-running, the listener is
        placed in the worker's data dict.

        NOTE: This is not the most ideal design because the worker's
        data dict could be serialized at any point, which would cause
        an error. Need to sync w/ the Dask team to see if there's a better
        way to do this.
        Ref: https://github.com/rapidsai/cuml/issues/841
        """
        [self.client.run(CommsBase.func_ucp_create_listener, self.sessionId,
                         random.random(), workers=[w], wait=False)
         for w in self.worker_addresses]

        self.block_for_init("ucp_listener")

    def stop_ucp_listeners(self):
        """
        Stops the UCP listeners attached to this session
        """
        self.client.run(CommsBase.func_ucp_stop_listener,
                        self.sessionId,
                        wait=True)

    @staticmethod
    async def func_build_handle_p2p(sessionId):
        """
        Builds a cumlHandle on the current worker given the initialized comms
        :param nccl_comm: ncclComm_t Initialized NCCL comm
        :param eps: size_t initialized endpoints
        :param nWorkers: int number of workers in cluster
        :param workerId: int Rank of current worker
        :return:
        """
        ucp_worker = ucp.get_ucp_worker()

        session_state = worker_state(sessionId)

        handle = Handle()
        nccl_comm = session_state["nccl"]
        eps = session_state["ucp_eps"]
        nWorkers = session_state["nworkers"]
        workerId = session_state["wid"]

        inject_comms_on_handle(handle, nccl_comm, ucp_worker, eps,
                               nWorkers, workerId)

        worker_state(sessionId)["handle"] = handle

    @staticmethod
    async def func_build_handle(sessionId):
        """
        Builds a cumlHandle on the current worker given the initialized comms
        :param nccl_comm: ncclComm_t Initialized NCCL comm
        :param nWorkers: int number of workers in cluster
        :param workerId: int Rank of current worker
        :return:
        """
        handle = Handle()

        session_state = worker_state(sessionId)

        workerId = session_state["wid"]
        nWorkers = session_state["nworkers"]

        nccl_comm = session_state["nccl"]
        inject_comms_on_handle_coll_only(handle, nccl_comm, nWorkers, workerId)
        session_state["handle"] = handle


    @staticmethod
    def func_wait_for_key(sessionId, key):

        # TODO: Check for errors as well...
        while key not in worker_state(sessionId):
            time.sleep(0.01)

    @staticmethod
    def _func_store_initial_state(nworkers, sessionId, uniqueId, wid):
        session_state = worker_state(sessionId)
        session_state["nccl_uid"] = uniqueId
        session_state["wid"] = wid
        session_state["nworkers"] = nworkers

    def block_for_init(self, key):

        # TODO: Add appropriate error handling here to eliminate endless loops

        [self.client.run(CommsBase.func_wait_for_key,
                         self.sessionId,
                         key,
                         wait=True)]

    def init_nccl(self):
        """
        Use nccl-py to initialize ncclComm_t on each worker and
        store the futures for this instance.
        """
        self.uniqueId = nccl.get_unique_id()

        for worker, idx in zip(self.worker_addresses, range(len(self.workers))):
            self.client.run(CommsBase._func_store_initial_state,
                            len(self.workers),
                            self.sessionId,
                            self.uniqueId,
                            idx,
                            workers=[worker])

        self.client.run(CommsBase.func_init_nccl,
                        self.sessionId,
                        wait=False)

        self.block_for_init("nccl")

        self.nccl_initialized = True

    def init_ucp(self):
        """
        Use ucx-py to initialize ucp endpoints so that every
        worker can communicate, point-to-point, with every other worker
        """
        self.create_ucp_listeners()
        self.ucp_create_endpoints()

        self.ucx_initialized = True

    def init(self):
        """
        Initializes the underlying comms. NCCL is required but
        UCX is only initialized if `comms_p2p == True`
        """
        self.init_nccl()

        if self.comms_p2p:
            self.init_ucp()

            self.client.run(CommsBase.func_build_handle_p2p,
                             self.sessionId,
                             wait=False)

        else:
            self.client.run(CommsBase.func_build_handle,
                              self.sessionId,
                              wait=False)

        self.block_for_init("handle")

        _set_global_comms(self)

    @staticmethod
    async def func_ucp_create_endpoints(sessionId, worker_info):
        """
        Runs on each worker to create ucp endpoints to all other workers
        :param sessionId: uuid unique id for this instance
        :param worker_info: dict Maps worker address to rank & UCX port
        :param r: float a random number to stop the function from being cached
        """
        dask_worker = get_worker()
        local_address = parse_host_port(dask_worker.address)

        eps = [None] * len(worker_info)

        count = 1

        for k in worker_info:
            if k != local_address:
                ip, port = k
                rank, ucp_port = worker_info[k]
                ep = await ucp.get_endpoint(ip.encode(), ucp_port, timeout=1)
                eps[rank] = ep
                count += 1

        worker_state(sessionId)["ucp_eps"] = eps

    def ucp_create_endpoints(self):
        """
        Creates UCX endpoints for each worker in the Dask cluster and
        connects them to every other worker.
        """
        worker_info = self.worker_info()

        [self.client.run(CommsBase.func_ucp_create_endpoints, self.sessionId,
                         worker_info, workers=[w],
                         wait=False)
         for w in self.worker_addresses]

        self.block_for_init("ucp_eps")

    @staticmethod
    def func_destroy_nccl(sessionId):
        """
        Destroys NCCL communicator on worker
        :param nccl_comm: ncclComm_t Initialized NCCL comm
        :param r: float a random number to stop the function from being cached
        """
        worker_state(sessionId)["nccl"].destroy()
        del worker_state(sessionId)["nccl"]

    def destroy_nccl(self):
        """
        Destroys all NCCL communicators on workers
        """
        self.client.run(CommsBase.func_destroy_nccl, self.sessionId, wait=True)

    @staticmethod
    def func_destroy_ep(sessionId):
        """
        Destroys UCP endpoints on worker
        :param r: float a random number to stop the function from being cached
        """
        for ep in worker_state(sessionId)["ucp_eps"]:
            if ep is not None:
                ucp.destroy_ep(ep)
        del worker_state(sessionId)["ucp_eps"]

    def destroy_eps(self):
        """
        Destroys all UCP endpoints on all workers
        """
        self.client.run(CommsBase.func_destroy_ep,
                         self.sessionId,
                         wait=True)

    def destroy_ucp(self):
        """
        Stops initialized UCP endpoints and listers on the Dask workers
        """
        self.destroy_eps()
        self.stop_ucp_listeners()

    def destroy(self):
        """
        Shuts down initialized comms and cleans up resources.
        """
        if self.comms_p2p:
            self.destroy_ucp()

        self.destroy_nccl()
