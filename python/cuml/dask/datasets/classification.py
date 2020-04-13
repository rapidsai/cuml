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

from cuml.datasets.classification import _generate_hypercube
from cuml.datasets.classification import make_classification as sg_make_classification
from cuml.datasets.utils import _create_rs_generator
from cuml.utils import with_cupy_rmm

from dask.distributed import default_client, wait

import cupy as cp
import numpy as np
import math

from time import sleep

def _create_covariance(*args, rs, dtype='float32'):
    return 2 * rs.rand(*args, dtype=dtype) - 1


@with_cupy_rmm
def make_classification(n_samples=100, n_features=20, n_informative=2,
                        n_redundant=2, n_repeated=0, n_classes=2,
                        n_clusters_per_class=2, weights=None, flip_y=0.01,
                        class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
                        shuffle=True, random_state=None, order='F',
                        dtype='float32', n_parts=None):
    """Generate a random n-class classification problem.
    This initially creates clusters of points normally distributed (std=1)
    about vertices of an ``n_informative``-dimensional hypercube with sides of
    length ``2*class_sep`` and assigns an equal number of clusters to each
    class. It introduces interdependence between these features and adds
    various types of further noise to the data.
    Without shuffling, ``X`` horizontally stacks features in the following
    order: the primary ``n_informative`` features, followed by ``n_redundant``
    linear combinations of the informative features, followed by ``n_repeated``
    duplicates, drawn randomly with replacement from the informative and
    redundant features. The remaining features are filled with random noise.
    Thus, without shuffling, all useful features are contained in the columns
    ``X[:, :n_informative + n_redundant + n_repeated]``.
    Read more in the :ref:`User Guide <sample_generators>`.
    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.
    n_features : int, optional (default=20)
        The total number of features. These comprise ``n_informative``
        informative features, ``n_redundant`` redundant features,
        ``n_repeated`` duplicated features and
        ``n_features-n_informative-n_redundant-n_repeated`` useless features
        drawn at random.
    n_informative : int, optional (default=2)
        The number of informative features. Each class is composed of a number
        of gaussian clusters each located around the vertices of a hypercube
        in a subspace of dimension ``n_informative``. For each cluster,
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
    weights : array-like of shape (n_classes,) or (n_classes - 1,),\
              (default=None)
        The proportions of samples assigned to each class. If None, then
        classes are balanced. Note that if ``len(weights) == n_classes - 1``,
        then the last class weight is automatically inferred.
        More than ``n_samples`` samples may be returned if the sum of
        ``weights`` exceeds 1.
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
    X : device array of shape [n_samples, n_features]
        The generated samples.
    y : device array of shape [n_samples]
        The integer labels for class membership of each sample.
    Notes
    -----
    The algorithm is adapted from Guyon [1] and was designed to generate
    the "Madelon" dataset.
    References
    ----------
    .. [1] I. Guyon, "Design of experiments for the NIPS 2003 variable
           selection benchmark", 2003.
    """

    client = default_client()

    rs = _create_rs_generator(random_state)

    workers = list(client.has_what().keys())

    n_parts = n_parts if n_parts is not None else len(workers)
    parts_workers = (workers * n_parts)[:n_parts]
    print(parts_workers)
    rows_per_part = math.ceil(n_samples / n_parts)

    n_clusters = n_classes * n_clusters_per_class

    # create centroids
    centroids = _generate_hypercube(n_clusters, n_informative, rs, dtype)

    # # create covariance matrices
    informative_covariance_local = rs.rand(n_clusters, n_informative, n_informative, dtype=dtype)
    informative_covariance = client.scatter(informative_covariance_local, workers=workers)
    del informative_covariance_local

    redundant_covariance_local = rs.rand(n_informative, n_redundant, dtype=dtype)
    redundant_covariance = client.scatter(redundant_covariance_local, workers=workers)
    del redundant_covariance_local

    wait([informative_covariance, redundant_covariance])
    print(client.has_what())

    # repeated indices
    n = n_informative + n_redundant
    repeated_indices = ((n - 1) * rs.rand(n_repeated, dtype=dtype) + 0.5).astype(np.intp)

    # scale and shift
    if shift is None:
        shift = (2 * rs.rand(n_features, dtype=dtype) - 1) * class_sep

    if scale is None:
        scale = 1 + 100 * rs.rand(n_features, dtype=dtype)

    # Create arrays on each worker (gpu)
    parts = []
    worker_rows = []
    rows_so_far = 0
    for idx, worker in enumerate(parts_workers):
        if rows_so_far + rows_per_part <= n_samples:
            rows_so_far += rows_per_part
            worker_rows.append(rows_per_part)
        else:
            worker_rows.append((int(n_samples) - rows_so_far))

    print(parts_workers)
    parts = [client.submit(sg_make_classification, worker_rows[i], n_features,
                                   n_informative, n_redundant, n_repeated, n_classes,
                                   n_clusters_per_class, weights, flip_y,
                                   class_sep, hypercube, shift, scale,
                                   shuffle, random_state, order, dtype,
                                   centroids, informative_covariance, redundant_covariance,
                                   repeated_indices,
                                   pure=False,
                                   workers=[parts_workers[i]]) for i in range(len(parts_workers))]

    wait(parts)