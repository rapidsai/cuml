#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from cuml.cluster import KMeans


def _required_random_init_rows(n_clusters, n_workers, rank):
    n_sampling_workers = min(n_workers, n_clusters)
    if rank >= n_sampling_workers:
        return 0

    required_rows = n_clusters // n_sampling_workers
    if rank == 0:
        required_rows += n_clusters % n_sampling_workers
    return required_rows


class KMeansMG(KMeans):
    """
    A Multi-Node Multi-GPU implementation of KMeans
    """

    _multi_gpu = True

    def __init__(
        self,
        *,
        handle,
        rank=None,
        n_workers=None,
        **kwargs,
    ):
        self.handle = handle
        self.rank = rank
        self.n_workers = n_workers
        super().__init__(**kwargs)

    def _validate_fit_row_constraints(self, n_rows):
        super()._validate_fit_row_constraints(n_rows)
        if isinstance(self.init, str):
            if self.init == "k-means++":
                raise ValueError(
                    "init='k-means++' is not supported for KMeansMG. "
                    "Use init='k-means||' or init='scalable-k-means++'."
                )
            if self.init not in {
                "scalable-k-means++",
                "k-means||",
                "random",
            }:
                raise ValueError(
                    f"init={self.init!r} is not supported for KMeansMG."
                )

        if self.oversampling_factor == 0:
            raise ValueError(
                "oversampling_factor=0 is not supported for KMeansMG."
            )

        if (
            not isinstance(self.init, str)
            or self.init != "random"
            or self.rank is None
            or not self.n_workers
        ):
            return

        required_rows = _required_random_init_rows(
            self.n_clusters, self.n_workers, self.rank
        )
        if n_rows < required_rows:
            raise ValueError(
                f"init='random' requires rank {self.rank} to sample up to "
                f"{required_rows} initial centroid(s), but this rank only "
                f"has {n_rows} row(s). Repartition the data so each rank "
                f"has enough rows for initialization, reduce n_clusters, "
                f"or provide explicit initial centers."
            )
