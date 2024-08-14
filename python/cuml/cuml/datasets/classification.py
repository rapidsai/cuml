# Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

import cuml.internals
from cuml.internals.import_utils import has_sklearn
from cuml.datasets.utils import _create_rs_generator

from cuml.internals.safe_imports import gpu_only_import

from cuml.internals.safe_imports import (
    cpu_only_import,
    gpu_only_import_from,
    null_decorator,
)

nvtx_annotate = gpu_only_import_from("nvtx", "annotate", alt=null_decorator)

cp = gpu_only_import("cupy")
np = cpu_only_import("numpy")


def _generate_hypercube(samples, dimensions, rng):
    """Returns distinct binary samples of length dimensions"""
    if not has_sklearn():
        raise RuntimeError(
            "Scikit-learn is needed to run \
                           make_classification."
        )

    from sklearn.utils.random import sample_without_replacement

    if dimensions > 30:
        return np.hstack(
            [
                np.random.randint(2, size=(samples, dimensions - 30)),
                _generate_hypercube(samples, 30, rng),
            ]
        )
    random_state = int(rng.randint(dimensions))
    out = sample_without_replacement(
        2**dimensions, samples, random_state=random_state
    ).astype(dtype=">u4", copy=False)
    out = np.unpackbits(out.view(">u1")).reshape((-1, 32))[:, -dimensions:]
    return out


@nvtx_annotate(message="datasets.make_classification", domain="cuml_python")
@cuml.internals.api_return_generic()
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
    _centroids=None,
    _informative_covariance=None,
    _redundant_covariance=None,
    _repeated_indices=None,
):
    """
    Generate a random n-class classification problem.
    This initially creates clusters of points normally distributed (std=1)
    about vertices of an `n_informative`-dimensional hypercube with sides of
    length :py:`2*class_sep` and assigns an equal number of clusters to each
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

        >>> from cuml.datasets.classification import make_classification

        >>> X, y = make_classification(n_samples=10, n_features=4,
        ...                            n_informative=2, n_classes=2,
        ...                            random_state=10)

        >>> print(X) # doctest: +SKIP
        [[-1.7974224   0.24425316  0.39062843 -0.38293394]
        [ 0.6358963   1.4161923   0.06970507 -0.16085647]
        [-0.22802866 -1.1827322   0.3525861   0.276615  ]
        [ 1.7308872   0.43080002  0.05048406  0.29837844]
        [-1.9465544   0.5704457  -0.8997551  -0.27898186]
        [ 1.0575483  -0.9171263   0.09529338  0.01173469]
        [ 0.7917619  -1.0638094  -0.17599393 -0.06420116]
        [-0.6686142  -0.13951421 -0.6074711   0.21645583]
        [-0.88968956 -0.914443    0.1302423   0.02924336]
        [-0.8817671  -0.84549576  0.1845096   0.02556021]]

        >>> print(y)
        [1 0 1 1 1 1 1 1 1 0]

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
    weights : array-like of shape (n_classes,) or (n_classes - 1,),\
              (default=None)
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
    _centroids: array of centroids of shape (n_clusters, n_informative)
    _informative_covariance: array for covariance between informative features
        of shape (n_clusters, n_informative, n_informative)
    _redundant_covariance: array for covariance between redundant features
        of shape (n_informative, n_redundant)
    _repeated_indices: array of indices for the repeated features
        of shape (n_repeated, )

    Returns
    -------
    X : device array of shape [n_samples, n_features]
        The generated samples.
    y : device array of shape [n_samples]
        The integer labels for class membership of each sample.

    Notes
    -----
    The algorithm is adapted from Guyon [1]_ and was designed to generate
    the "Madelon" dataset. How we optimized for GPUs:

        1. Firstly, we generate X from a standard univariate instead of zeros.
           This saves memory as we don't need to generate univariates each
           time for each feature class (informative, repeated, etc.) while
           also providing the added speedup of generating a big matrix
           on GPU
        2. We generate :py:`order=F` construction. We exploit the
           fact that X is a generated from a univariate normal, and
           covariance is introduced with matrix multiplications. Which means,
           we can generate X as a 1D array and just reshape it to the
           desired order, which only updates the metadata and eliminates
           copies
        3. Lastly, we also shuffle by construction. Centroid indices are
           permuted for each sample, and then we construct the data for
           each centroid. This shuffle works for both :py:`order=C` and
           :py:`order=F` and eliminates any need for secondary copies

    References
    ----------
    .. [1] I. Guyon, "Design of experiments for the NIPS 2003 variable
           selection benchmark", 2003.

    """
    cuml.internals.set_api_output_type("cupy")

    generator = _create_rs_generator(random_state)

    # Count features, clusters and samples
    if n_informative + n_redundant + n_repeated > n_features:
        raise ValueError(
            "Number of informative, redundant and repeated "
            "features must sum to less than the number of total"
            " features"
        )
    # Use log2 to avoid overflow errors
    if n_informative < np.log2(n_classes * n_clusters_per_class):
        msg = "n_classes({}) * n_clusters_per_class({}) must be"
        msg += " smaller or equal 2**n_informative({})={}"
        raise ValueError(
            msg.format(
                n_classes,
                n_clusters_per_class,
                n_informative,
                2**n_informative,
            )
        )

    if weights is not None:
        if len(weights) not in [n_classes, n_classes - 1]:
            raise ValueError(
                "Weights specified but incompatible with number " "of classes."
            )
        if len(weights) == n_classes - 1:
            if isinstance(weights, list):
                weights = weights + [1.0 - sum(weights)]
            else:
                weights = np.resize(weights, n_classes)
                weights[-1] = 1.0 - sum(weights[:-1])
    else:
        weights = [1.0 / n_classes] * n_classes

    n_clusters = n_classes * n_clusters_per_class

    # Distribute samples among clusters by weight
    n_samples_per_cluster = [
        int(n_samples * weights[k % n_classes] / n_clusters_per_class)
        for k in range(n_clusters)
    ]

    for i in range(n_samples - sum(n_samples_per_cluster)):
        n_samples_per_cluster[i % n_clusters] += 1

    # Initialize X and y
    X = generator.randn(n_samples * n_features, dtype=dtype)
    X = X.reshape((n_samples, n_features), order=order)
    y = cp.zeros(n_samples, dtype=np.int64)

    # Build the polytope whose vertices become cluster centroids
    if _centroids is None:
        centroids = cp.array(
            _generate_hypercube(n_clusters, n_informative, generator)
        ).astype(dtype, copy=False)
    else:
        centroids = _centroids
    centroids *= 2 * class_sep
    centroids -= class_sep
    if not hypercube:
        centroids *= generator.rand(n_clusters, 1, dtype=dtype)
        centroids *= generator.rand(1, n_informative, dtype=dtype)

    # Create redundant features
    if n_redundant > 0:
        if _redundant_covariance is None:
            B = 2 * generator.rand(n_informative, n_redundant, dtype=dtype) - 1
        else:
            B = _redundant_covariance

    # Create each cluster; a variant of make_blobs
    if shuffle:
        proba_samples_per_cluster = np.array(n_samples_per_cluster) / np.sum(
            n_samples_per_cluster
        )
        shuffled_sample_indices = generator.choice(
            n_clusters, n_samples, replace=True, p=proba_samples_per_cluster
        )
        for k, centroid in enumerate(centroids):
            centroid_indices = cp.where(shuffled_sample_indices == k)
            y[centroid_indices[0]] = k % n_classes

            X_k = X[centroid_indices[0], :n_informative]

            if _informative_covariance is None:
                A = (
                    2
                    * generator.rand(n_informative, n_informative, dtype=dtype)
                    - 1
                )
            else:
                A = _informative_covariance[k]
            X_k = cp.dot(X_k, A)

            # NOTE: This could be done outside the loop, but a current
            # cupy bug does not allow that
            # https://github.com/cupy/cupy/issues/3284
            if n_redundant > 0:
                X[
                    centroid_indices[0],
                    n_informative : n_informative + n_redundant,
                ] = cp.dot(X_k, B)

            X_k += centroid  # shift the cluster to a vertex
            X[centroid_indices[0], :n_informative] = X_k
    else:
        stop = 0
        for k, centroid in enumerate(centroids):
            start, stop = stop, stop + n_samples_per_cluster[k]
            y[start:stop] = k % n_classes  # assign labels
            X_k = X[start:stop, :n_informative]  # slice a view of the cluster

            if _informative_covariance is None:
                A = (
                    2
                    * generator.rand(n_informative, n_informative, dtype=dtype)
                    - 1
                )
            else:
                A = _informative_covariance[k]
            X_k = cp.dot(X_k, A)  # introduce random covariance

            if n_redundant > 0:
                X[
                    start:stop, n_informative : n_informative + n_redundant
                ] = cp.dot(X_k, B)

            X_k += centroid  # shift the cluster to a vertex
            X[start:stop, :n_informative] = X_k

    # Repeat some features
    if n_repeated > 0:
        n = n_informative + n_redundant
        if _repeated_indices is None:
            indices = (
                (n - 1) * generator.rand(n_repeated, dtype=dtype) + 0.5
            ).astype(np.intp)
        else:
            indices = _repeated_indices
        X[:, n : n + n_repeated] = X[:, indices]

    # Randomly replace labels
    if flip_y >= 0.0:
        flip_mask = generator.rand(n_samples, dtype=dtype) < flip_y
        y[flip_mask] = generator.randint(n_classes, size=int(flip_mask.sum()))

    # Randomly shift and scale
    if shift is None:
        shift = (2 * generator.rand(n_features, dtype=dtype) - 1) * class_sep
    X += shift

    if scale is None:
        scale = 1 + 100 * generator.rand(n_features, dtype=dtype)
    X *= scale

    return X, y
