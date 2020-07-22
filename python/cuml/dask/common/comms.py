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

from cuml.raft.dask.common.comms import Comms
from cuml.raft.dask.common.comms import worker_state
from cuml.raft.dask.common.comms import local_handle
from cuml.raft.dask.common.comms_utils import *
from cuml.common.handle import Handle

class CommsContext(Comms):
    def __init__(self, comms_p2p=False, client=None, verbose=False,
                 streams_per_handle=0):
        print("In comms init")
        super().__init__(comms_p2p=comms_p2p, client=client,
                         verbose=verbose,
                         streams_per_handle=streams_per_handle)
    
    def init(self, workers=None):
        super().init(workers=workers)
        self.client.run(_wrap_cuml_handle,
                        self.sessionId,
                        workers=self.worker_addresses,
                        wait=True)

async def _wrap_cuml_handle(sessionId):
    handle = local_handle(sessionId)
    if handle:
        worker_state(sessionId)["handle"] = Handle(raftHandle=handle)