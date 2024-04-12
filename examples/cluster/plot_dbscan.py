"""
===================================
Demo of DBSCAN clustering algorithm
===================================

Example adapted from `the scikit-learn gallery <https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html>`_.

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) finds core
samples in regions of high density and expands clusters from them. This
algorithm is good for data which contains clusters of similar density.

"""

# %%
# Data generation
# ---------------
#
# We use :class:`~sklearn.datasets.make_blobs` to create 3 synthetic clusters.

import cupy as cp
from cuml.datasets import make_blobs
from cuml.preprocessing import StandardScaler

centers = cp.array([[1, 1], [-1, -1], [1, -1]])
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)

X = StandardScaler().fit_transform(X)

# %%
# We can visualize the resulting data:

import matplotlib.pyplot as plt

X_ = X.get()
plt.scatter(X_[:, 0], X_[:, 1])
plt.show()

# %%
# Compute DBSCAN
# --------------
#
# One can access the labels assigned by :class:`~sklearn.cluster.DBSCAN` using
# the `labels_` attribute. Noisy samples are given the label math:`-1`.

import numpy as np

from cuml import metrics
from sklearn import metrics as sk_metrics
from cuml.cluster import DBSCAN

db = DBSCAN(eps=0.3, min_samples=10).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(cp.unique(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels.get()).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

# %%
# Clustering algorithms are fundamentally unsupervised learning methods.
# However, since :class:`~sklearn.datasets.make_blobs` gives access to the true
# labels of the synthetic clusters, it is possible to use evaluation metrics
# that leverage this "supervised" ground truth information to quantify the
# quality of the resulting clusters. Examples of such metrics are the
# homogeneity, completeness, V-measure, Rand-Index, Adjusted Rand-Index and
# Adjusted Mutual Information (AMI).
#
# If the ground truth labels are not known, evaluation can only be performed
# using the model results itself. In that case, the Silhouette Coefficient comes
# in handy.

print(f"Homogeneity: {metrics.homogeneity_score(labels_true.astype(cp.int32), labels):.3f}")
print(f"Completeness: {metrics.completeness_score(labels_true.astype(cp.int32), labels):.3f}")
print(f"V-measure: {metrics.v_measure_score(labels_true.astype(cp.int32), labels):.3f}")
print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(labels_true.astype(cp.int32), labels):.3f}")
print(
    "Adjusted Mutual Information:"
    f" {sk_metrics.adjusted_mutual_info_score(labels_true.astype(cp.int32).get(), labels.get()):.3f}"
)
print(f"Silhouette Coefficient: {sk_metrics.silhouette_score(X_, labels.get()):.3f}")

# %%
# Plot results
# ------------
#
# Core samples (large dots) and non-core samples (small dots) are color-coded
# according to the assigned cluster. Samples tagged as noise are represented in
# black.

unique_labels = cp.unique(labels)
core_samples_mask = cp.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
core_samples_mask = core_samples_mask.get()

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k).get()

    xy = X_[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = X_[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title(f"Estimated number of clusters: {n_clusters_}")
plt.show()
