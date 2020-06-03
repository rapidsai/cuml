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
from cuml.dask.common.ucx import UCX

from dask.distributed import Client
from dask.distributed import get_worker

from cuml.dask.common.utils import parse_host_port


def test_listener(cluster):

    c = Client(cluster)

    multiple_workers = len(c.scheduler_info()["workers"]) > 1

    # Test only runs when multiple workers are present
    if multiple_workers:

        def build_ucx():
            # Create listener and cache on worker
            get_worker()._callback_invoked = False

            def mock_callback(ep):
                get_worker()._callback_invoked = True

            ucx = UCX.get(mock_callback)

            get_worker()._ucx = ucx
            return get_worker().address, ucx.listener_port()

        ports = c.run(build_ucx)

        def get_endpoints(addr_ports):
            # Create endpoints to all other workers
            ucx = get_worker()._ucx

            for address, port in addr_ports:
                if address != get_worker().address:
                    host, p = parse_host_port(address)
                    ucx.get_endpoint(host, port)

        c.run(get_endpoints, [ap for ap in ports.values()])

        def callback_invoked():
            # Return True if listener callback was invoked
            return get_worker()._callback_invoked

        invoked = c.run(callback_invoked)

        assert all(invoked)
