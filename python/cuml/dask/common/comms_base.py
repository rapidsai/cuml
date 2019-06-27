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
import ucp

from cuml.nccl import nccl

from nccl_example import nccl, inject_comms_on_handle, SimpleReduce
from cuml.common.handle import Handle

from dask.distributed import wait
from dask.distributed import get_worker

import random

import asyncio
import ucp

import uuid

from dask.distributed import default_client


async def connection_func(ep, listener):
    print("connection received from " + str(ep))


class CommsBase:

    def __init__(self, comms_coll=True, comms_p2p=False):
        self.client = default_client()
        self.comms_coll = comms_coll
        self.comms_p2p = comms_p2p

        # Used to identify this distinct session on workers
        self.sessionId = uuid.uuid4().bytes

        self.worker_addresses = self.get_workers_()
        self.workers = list(map(lambda x: parse_host_port(x), self.worker_addresses))

    def __dealloc__(self):
        self.destroy()

    def get_workers_(self, parse_address=True):
        """
        Return the list of workers parsed as [(address, port)]
        """
        return list(self.client.has_what().keys())

    def worker_ranks(self):
        """
        Builds a dictionary of { (worker_address, worker_port) : worker_rank }
        """
        return dict(list(map(lambda x: (x[1], x[0]), self.nccl_clique)))

    def worker_ports(self):
        """
        Builds a dictionary of { (worker_address, worker_port) : worker_port }
        """
        return dict(list(self.ucp_ports))

    def worker_info(self):
        """
        Builds a dictionary of { (worker_address, worker_port) : (worker_rank, worker_port ) }
        """
        ranks = self.worker_ranks() if self.comms_coll else None
        ports = self.worker_ports() if self.comms_p2p else None

        if self.comms_coll and self.comms_p2p:
            output = {}
            for k in self.worker_ranks().keys():
                output[k] = (ranks[k], ports[k])
            return output

        elif self.comms_coll:
            return ranks
        elif self.comms_p2p:
            return ports

    @staticmethod
    def func_init_nccl(workerId, nWorkers, uniqueId):
        """
        Initialize ncclComm_t on worker
        """
        n = nccl()
        n.init(nWorkers, uniqueId, workerId)
        return n

    @staticmethod
    def func_get_ucp_port(sessionId, r):
        """
        Return the port assigned to a UCP listener on worker
        """
        dask_worker = get_worker()
        port = dask_worker.data[sessionId].port
        return port

    @staticmethod
    async def ucp_create_listener(sessionId, r):
        dask_worker = get_worker()
        if sessionId in dask_worker.data:
            print("Listener already started for sessionId=" + str(sessionId))
        else:
            ucp.init()
            listener = ucp.start_listener(connection_func, 0, is_coroutine=True)

            dask_worker.data[sessionId] = listener
            task = asyncio.create_task(listener.coroutine)

            while not task.done():
                await task
                await asyncio.sleep(1)

            ucp.fin()
            del dask_worker.data[sessionId]

    @staticmethod
    def ucp_stop_listener(sessionId, r):
        dask_worker = get_worker()
        if sessionId in dask_worker.data:
            listener = dask_worker.data[sessionId]
            ucp.stop_listener(listener)
        else:
            print("Listener not found with sessionId=" + str(sessionId))

    def create_ucp_listeners(self):
        """
        Build a UCP listener on each worker. Since this async function is long-running, the listener is
        placed in the worker's data dict.

        NOTE: This is not the most ideal design because the worker's data dict could be serialized at
        any point, which would cause an error. Need to sync w/ the Dask team to see if there's a better
        way to do this.
        """
        [self.client.run(CommsBase.ucp_create_listener, self.sessionId, random.random(), workers=[w], wait=False) for w
         in
         self.worker_addresses]

    def get_ucp_ports(self):
        """
        Return the UCP listener ports attached to this session
        """
        self.ucp_ports = [
            (w, self.client.submit(CommsBase.func_get_ucp_port, self.sessionId, random.random(), workers=[w]).result())
            for w in self.workers]

    def stop_ucp_listeners(self):
        """
        Stops the UCP listeners attached to this session
        """
        a = [c.submit(CommsBase.ucp_stop_listener, self.sessionId, random.random(), workers=[w])
             for w in self.workers]
        wait(a)

    @staticmethod
    def func_build_handle(nccl_comm, eps, nWorkers, workerId):

        ucp_worker = ucp.get_ucp_worker()

        handle = Handle()
        inject_comms_on_handle(handle, nccl_comm, ucp_worker, eps, nWorkers, workerId)
        return handle

    def init_nccl(self):
        """
        Use nccl-py to initialize ncclComm_t on each worker and
        store the futures for this instance.
        """
        self.uniqueId = nccl.get_unique_id()

        workers_indices = list(zip(self.workers, range(len(self.workers))))

        self.nccl_clique = [(idx, worker, self.client.submit(CommsBase.func_init_nccl,
                                                             idx,
                                                             len(self.workers),
                                                             self.uniqueId,
                                                             workers=[worker]))
                            for worker, idx in workers_indices]

    def init_ucp(self):
        """
        Use ucx-py to initialize ucp endpoints so that every
        worker can communicate, point-to-point, with every other worker
        """
        self.create_ucp_listeners()
        self.get_ucp_ports()
        self.ucp_create_endpoints()

    def init(self):
        if self.comms_coll:
            self.init_nccl()

        if self.comms_p2p:
            self.init_ucp()

        # Combine ucp ports w/ nccl ranks

        eps_futures = dict(self.ucp_endpoints)

        self.handles = [(wid, w,
                         self.client.submit(CommsBase.func_build_handle, f, eps_futures[w], len(self.workers), wid,
                                            workers=[w]))
                        for wid, w, f in self.nccl_clique]

    @staticmethod
    async def func_ucp_create_endpoints(sessionId, worker_info, r):
        """
        Runs on each worker to create ucp endpoints to all other workers
        """
        dask_worker = get_worker()
        local_address = parse_host_port(dask_worker.address)

        eps = [None] * len(worker_info)

        count = 1
        size = len(worker_info) - 1

        for k in worker_info:
            if k != local_address:
                ip, port = k
                rank, ucp_port = worker_info[k]
                ep = await ucp.get_endpoint(ip.encode(), ucp_port, timeout=1)
                eps[rank] = ep
                count += 1
        dask_worker.data[str(sessionId) + "_eps"] = eps

    @staticmethod
    def func_get_endpoints(sessionId, r):
        """
        Fetches (and removes) the endpoints from the worker's data dict
        """
        dask_worker = get_worker()
        eps = dask_worker.data[str(sessionId) + "_eps"]
        del dask_worker.data[str(sessionId) + "_eps"]
        return eps

    def ucp_create_endpoints(self):

        worker_info = self.worker_info()

        [self.client.run(CommsBase.func_ucp_create_endpoints, self.sessionId, worker_info, random.random(), workers=[w],
                         wait=True)
         for w in self.worker_addresses]

        ret = [(w, self.client.submit(CommsBase.func_get_endpoints, self.sessionId, random.random(), workers=[w])) for w
               in self.workers]
        wait(ret)

        self.ucp_endpoints = ret

    @staticmethod
    def func_destroy_nccl(nccl_comm, r):
        """
        Destroys NCCL communicator on worker
        """
        nccl_comm.destroy()

    def destroy_nccl(self):
        """
        Destroys all NCCL communicators on workers
        """
        a = [self.client.submit(CommsBase.func_destroy_nccl, f, random.random(), workers=[w]) for wid, w, f in
             self.nccl_clique]
        wait(a)

    def func_destroy_ep(eps, r):
        """
        Destroys UCP endpoints on worker
        """
        for ep in eps:
            if ep is not None:
                ucp.destroy_ep(ep)

    def destroy_eps(self):
        """
        Destroys all UCP endpoints on all workers
        """
        a = [self.client.submit(CommsBase.func_destroy_ep, f, random.random(), workers=[w]) for w, f in
             self.ucp_endpoints]
        wait(a)

    def destroy_ucp(self):
        self.destroy_eps()
        self.stop_ucp_listeners()

    def destroy(self):

        self.handles = None

        if self.comms_p2p:
            self.destroy_ucp()
            self.ucp_ports = None
            self.ucp_endpoints = None

        if self.comms_coll:
            # TODO: Figure out why this fails when UCP + NCCL are both used
            #             self.destroy_nccl()
            self.nccl_clique = None
