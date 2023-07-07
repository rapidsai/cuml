# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

from cuml.internals.safe_imports import cpu_only_import
from cuml.datasets.classification import _generate_hypercube
from cuml.datasets.classification import (
    make_classification as sg_make_classification,
)
from cuml.datasets.utils import _create_rs_generator
from cuml.dask.datasets.utils import _get_X
from cuml.dask.datasets.utils import _get_labels
from cuml.dask.datasets.utils import _create_delayed
from cuml.dask.common.utils import get_client
from cuml.common import with_cupy_rmm

import dask.array as da

from cuml.internals.safe_imports import gpu_only_import

cp = gpu_only_import("cupy")
np = cpu_only_import("numpy")


def _create_covariance(dims, seed, dtype="float32"):
    local_rs = cp.random.RandomState(seed=seed)
    return 2 * local_rs.rand(*dims, dtype=dtype) - 1


@with_cupy_rmm
def make_classification(
    n_samples=100,
    n_features=20,
    n_informative=2,
    n_redundant=2,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=2,
    weights=None,
    flip_y=0.01,
    class_sep=1.0,
    hypercube=True,
    shift=0.0,
    scale=1.0,
    shuffle=True,
    random_state=None,
    order="F",
    dtype="float32",
    n_parts=None,
    client=None,
):
    """
    Generate a random n-class classification problem.

    This initially creates clusters of points normally distributed (std=1)
    about vertices of an `n_informative`-dimensional hypercube with sides of
    length :py:`2 * class_sep` and assigns an equal number of clusters to each
    class. It introduces interdependence between these features and adds
    various types of further noise to the data.

    Without shuffling, `X` horizontally stacks features in the following
    order: the primary `n_informative` features, followed by `n_redundant`
    linear combinations of the informative features, followed by `n_repeated`
    duplicates, drawn randomly with replacement from the informative and
    redundant features. The remaining features are filled with random noise.
    Thus, without shuffling, all useful features are contained in the columns
    :py:`X[:, :n_informative + n_redundant + n_repeated]`.

    Examples
    --------
    .. code-block:: python

        >>> from dask.distributed import Client
        >>> from dask_cuda import LocalCUDACluster
        >>> from cuml.dask.datasets.classification import make_classification
        >>> cluster = LocalCUDACluster()
        >>> client = Client(cluster)
        >>> X, y = make_classification(n_samples=10, n_features=4,
        ...                            random_state=1, n_informative=2,
        ...                            n_classes=2)
        >>> print(X.compute()) # doctest: +SKIP
        [[-1.1273878   1.2844919  -0.32349187  0.1595734 ]
        [ 0.80521786 -0.65946865 -0.40753683  0.15538901]
        [ 1.0404129  -1.481386    1.4241115   1.2664981 ]
        [-0.92821544 -0.6805706  -0.26001272  0.36004275]
        [-1.0392245  -1.1977317   0.16345565 -0.21848428]
        [ 1.2273135  -0.529214    2.4799604   0.44108105]
        [-1.9163864  -0.39505136 -1.9588828  -1.8881643 ]
        [-0.9788184  -0.89851004 -0.08339313  0.1130247 ]
        [-1.0549078  -0.8993015  -0.11921967  0.04821599]
        [-1.8388828  -1.4063598  -0.02838472 -1.0874642 ]]
        >>> print(y.compute()) # doctest: +SKIP
        [1 0 0 0 0 1 0 0 0 0]
        >>> client.close()
        >>> cluster.close()

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.
    n_features : int, optional (default=20)
        The total number of features. These comprise `n_informative`
        informative features, `n_redundant` redundant features,
        `n_repeated` duplicated features and
        :py:`n_features-n_informative-n_redundant-n_repeated` useless features
        drawn at random.
    n_informative : int, optional (default=2)
        The number of informative features. Each class is composed of a number
        of gaussian clusters each located around the vertices of a hypercube
        in a subspace of dimension `n_informative`. For each cluster,
        informative features are drawn independently from  N(0, 1) and then
        randomly linearly combined within each cluster in order to add
        covariance. The clusters are then placed on the vertices of the
        hypercube.
    n_redundant : int, optional (default=2)
        The number of redundant features. These features are generated as
        random linear combinations of the informative features.
    n_repeated : int, optional (default=0)
        The number of duplicated features, drawn randomly from the informative
        and the redundant features.
    n_classes : int, optional (default=2)
        The number of classes (or labels) of the classification problem.
    n_clusters_per_class : int, optional (default=2)
        The number of clusters per class.
    weights : array-like of shape :py:`(n_classes,)` or :py:`(n_classes - 1,)`\
        , (default=None)
        The proportions of samples assigned to each class. If None, then
        classes are balanced. Note that if :py:`len(weights) == n_classes - 1`,
        then the last class weight is automatically inferred.
        More than `n_samples` samples may be returned if the sum of
        `weights` exceeds 1.
    flip_y : float, optional (default=0.01)
        The fraction of samples whose class is assigned randomly. Larger
        values introduce noise in the labels and make the classification
        task harder.
    class_sep : float, optional (default=1.0)
        The factor multiplying the hypercube size.  Larger values spread
        out the clusters/classes and make the classification task easier.
    hypercube : boolean, optional (default=True)
        If True, the clusters are put on the vertices of a hypercube. If
        False, the clusters are put on the vertices of a random polytope.
    shift : float, array of shape [n_features] or None, optional (default=0.0)
        Shift features by the specified value. If None, then features
        are shifted by a random value drawn in [-class_sep, class_sep].
    scale : float, array of shape [n_features] or None, optional (default=1.0)
        Multiply features by the specified value. If None, then features
        are scaled by a random value drawn in [1, 100]. Note that scaling
        happens after shifting.
    shuffle : boolean, optional (default=True)
        Shuffle the samples and the features.
    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    order: str, optional (default='F')
        The order of the generated samples
    dtype : str, optional (default='float32')
        Dtype of the generated samples
    n_parts : int (default = None)
        number of partitions to generate (this can be greater
        than the number of workers)

    Returns
    -------
    X : dask.array backed by CuPy array of shape [n_samples, n_features]
        The generated samples.
    y : dask.array backed by CuPy array of shape [n_samples]
        The integer labels for class membership of each sample.

    Notes
    -----
    How we extended the dask MNMG version from the single GPU version:

    1. We generate centroids of shape ``(n_centroids, n_informative)``
    2. We generate an informative covariance of shape \
        ``(n_centroids, n_informative, n_informative)``
    3. We generate a redundant covariance of shape \
        ``(n_informative, n_redundant)``
    4. We generate the indices for the repeated features \
    We pass along the references to the futures of the above arrays \
    with each part to the single GPU \
    `cuml.datasets.classification.make_classification` so that each \
    part (and worker) has access to the correct values to generate \
    data from the same covariances

    """

    client = get_client(client=client)

    rs = _create_rs_generator(random_state)

    workers = list(client.scheduler_info()["workers"].keys())

    n_parts = n_parts if n_parts is not None else len(workers)
    parts_workers = (workers * n_parts)[:n_parts]

    n_clusters = n_classes * n_clusters_per_class

    # create centroids
    centroids = cp.array(
        _generate_hypercube(n_clusters, n_informative, rs)
    ).astype(dtype, copy=False)

    covariance_seeds = rs.randint(n_features, size=2)
    informative_covariance = client.submit(
        _create_covariance,
        (n_clusters, n_informative, n_informative),
        int(covariance_seeds[0]),
        pure=False,
    )

    redundant_covariance = client.submit(
        _create_covariance,
        (n_informative, n_redundant),
        int(covariance_seeds[1]),
        pure=False,
    )

    # repeated indices
    n = n_informative + n_redundant
    repeated_indices = (
        (n - 1) * rs.rand(n_repeated, dtype=dtype) + 0.5
    ).astype(np.intp)

    # scale and shift
    if shift is None:
        shift = (2 * rs.rand(n_features, dtype=dtype) - 1) * class_sep

    if scale is None:
        scale = 1 + 100 * rs.rand(n_features, dtype=dtype)

    # Create arrays on each worker (gpu)
    rows_per_part = max(1, int(n_samples / n_parts))

    worker_rows = [rows_per_part] * n_parts

    worker_rows[-1] += n_samples % n_parts

    worker_rows = tuple(worker_rows)

    part_seeds = rs.permutation(n_parts)
    parts = [
        client.submit(
            sg_make_classification,
            worker_rows[i],
            n_features,
            n_informative,
            n_redundant,
            n_repeated,
            n_classes,
            n_clusters_per_class,
            weights,
            flip_y,
            class_sep,
            hypercube,
            shift,
            scale,
            shuffle,
            int(part_seeds[i]),
            order,
            dtype,
            centroids,
            informative_covariance,
            redundant_covariance,
            repeated_indices,
            pure=False,
            workers=[parts_workers[i]],
        )
        for i in range(len(parts_workers))
    ]

    X_parts = [
        client.submit(_get_X, f, pure=False) for idx, f in enumerate(parts)
    ]
    y_parts = [
        client.submit(_get_labels, f, pure=False)
        for idx, f in enumerate(parts)
    ]

    X_dela = _create_delayed(X_parts, dtype, worker_rows, n_features)
    y_dela = _create_delayed(y_parts, np.int64, worker_rows)

    X = da.concatenate(X_dela)
    y = da.concatenate(y_dela)

    return X, y
