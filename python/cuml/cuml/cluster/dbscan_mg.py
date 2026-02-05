#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from cuml.cluster import DBSCAN


class DBSCANMG(DBSCAN):
    """
    A Multi-Node Multi-GPU implementation of DBSCAN
    NOTE: This implementation of DBSCAN is meant to be used with an
    initialized cumlCommunicator instance inside an existing distributed
    system. Refer to the Dask DBSCAN implementation in
    `cuml.dask.cluster.dbscan`.
    """

    _multi_gpu = True

    def __init__(self, *, handle, **kwargs):
        self.handle = handle
        super().__init__(**kwargs)
