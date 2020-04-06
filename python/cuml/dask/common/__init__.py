#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

from cuml.dask.common.comms import CommsContext, worker_state, default_comms

from cuml.dask.common.comms_utils import inject_comms_on_handle, \
    perform_test_comms_allreduce, perform_test_comms_send_recv, \
    perform_test_comms_recv_any_rank, \
    inject_comms_on_handle_coll_only, is_ucx_enabled

from cuml.dask.common.dask_arr_utils import to_sp_dask_array # NOQA

from cuml.dask.common.dask_df_utils import get_meta  # NOQA
from cuml.dask.common.dask_df_utils import to_dask_cudf  # NOQA
from cuml.dask.common.dask_df_utils import to_dask_df  # NOQA

from cuml.dask.common.part_utils import *

from cuml.dask.common.utils import raise_exception_from_futures  # NOQA
from cuml.dask.common.utils import raise_mg_import_exception  # NOQA

# TODO: The following will be going away soon
from cuml.dask.common.dask_arr_utils import extract_arr_partitions
from cuml.dask.common.dask_df_utils import extract_ddf_partitions
from cuml.dask.common.dask_df_utils import extract_colocated_ddf_partitions

