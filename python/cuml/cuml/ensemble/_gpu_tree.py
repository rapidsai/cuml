# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""GPU-backed proxy objects that mimic sklearn DecisionTree estimators.

These are exposed via ``RandomForest{Classifier,Regressor}.estimators_``
and route inference through FIL while exposing the full tree structure
(topology, node values, impurity) extracted from the treelite model.
"""

from __future__ import annotations

import cupy as cp
import numpy as np
import sklearn.tree


def _compute_max_depth(children_left, children_right):
    """Compute the max depth of a tree from its children arrays."""
    n_nodes = len(children_left)
    if n_nodes == 0:
        return 0
    depths = np.zeros(n_nodes, dtype=np.intp)
    for i in range(n_nodes):
        left = children_left[i]
        if left != -1:
            depths[left] = depths[i] + 1
            depths[children_right[i]] = depths[i] + 1
    return int(depths.max())


def _extract_tree_from_treelite(tl_model, tree_idx, n_classes, is_classifier):
    """Extract sklearn-compatible tree arrays from a treelite tree accessor.

    Returns a dict with keys matching sklearn Tree attributes.
    """
    ta = tl_model.get_tree_accessor(tree_idx)
    header = tl_model.get_header_accessor()

    node_type = ta.get_field("node_type")  # 1=internal, 0=leaf
    n_nodes = int(ta.get_field("num_nodes")[0])
    cleft = ta.get_field("cleft").astype(np.intp)
    cright = ta.get_field("cright").astype(np.intp)
    split_index = ta.get_field("split_index").astype(np.intp)
    threshold_raw = ta.get_field("threshold")
    data_count = ta.get_field("data_count").astype(np.intp)
    leaf_vector = ta.get_field("leaf_vector")

    n_features = int(header.get_field("num_feature")[0])
    is_leaf = node_type == 0

    # Convert to sklearn conventions
    children_left = cleft.copy()
    children_right = cright.copy()
    children_left[is_leaf] = -1
    children_right[is_leaf] = -1

    feature = split_index.copy()
    feature[is_leaf] = -2

    threshold = threshold_raw.copy().astype(np.float64)
    threshold[is_leaf] = -2.0

    n_node_samples = data_count.copy()

    n_outputs = 1
    if is_classifier:
        leaf_probs = leaf_vector.reshape(-1, n_classes)

        value = np.zeros((n_nodes, n_outputs, n_classes), dtype=np.float64)
        leaf_counter = 0
        for i in range(n_nodes):
            if is_leaf[i]:
                value[i, 0, :] = leaf_probs[leaf_counter] * data_count[i]
                leaf_counter += 1

        for i in range(n_nodes - 1, -1, -1):
            if not is_leaf[i]:
                value[i, 0, :] = value[cleft[i], 0, :] + value[cright[i], 0, :]
    else:
        n_classes = 1
        leaf_vals = ta.get_field("leaf_value").astype(np.float64)

        value = np.zeros((n_nodes, n_outputs, 1), dtype=np.float64)
        for i in range(n_nodes):
            if is_leaf[i]:
                value[i, 0, 0] = leaf_vals[i] * data_count[i]

        for i in range(n_nodes - 1, -1, -1):
            if not is_leaf[i]:
                value[i, 0, 0] = value[cleft[i], 0, 0] + value[cright[i], 0, 0]

    # Impurity: Gini for classification, variance for regression
    impurity = np.zeros(n_nodes, dtype=np.float64)
    if is_classifier:
        for i in range(n_nodes):
            total = value[i, 0, :].sum()
            if total > 0:
                p = value[i, 0, :] / total
                impurity[i] = 1.0 - np.sum(p**2)
    else:
        pass  # TODO: compute variance-based impurity for regression

    max_depth = _compute_max_depth(children_left, children_right)

    return {
        "children_left": children_left,
        "children_right": children_right,
        "feature": feature,
        "threshold": threshold,
        "value": value,
        "impurity": impurity,
        "n_node_samples": n_node_samples,
        "weighted_n_node_samples": n_node_samples.astype(np.float64),
        "node_count": n_nodes,
        "n_features": n_features,
        "n_classes": np.array(
            [n_classes] if is_classifier else [1], dtype=np.intp
        ),
        "n_outputs": n_outputs,
        "max_depth": max_depth,
    }


class GPUTree:
    """Duck-typed proxy for ``sklearn.tree._tree.Tree``.

    Exposes tree structure arrays and routes ``predict``/``apply`` through FIL.
    """

    def __init__(self, tree_data, fil_model, tree_idx):
        self._fil_model = fil_model
        self._tree_idx = tree_idx

        self.children_left = tree_data["children_left"]
        self.children_right = tree_data["children_right"]
        self.feature = tree_data["feature"]
        self.threshold = tree_data["threshold"]
        self.value = tree_data["value"]
        self.impurity = tree_data["impurity"]
        self.n_node_samples = tree_data["n_node_samples"]
        self.weighted_n_node_samples = tree_data["weighted_n_node_samples"]
        self.node_count = tree_data["node_count"]
        self.capacity = tree_data["node_count"]
        self.n_features = tree_data["n_features"]
        self.n_classes = tree_data["n_classes"]
        self.n_outputs = tree_data["n_outputs"]
        self.max_depth = tree_data["max_depth"]
        self.max_n_classes = int(self.n_classes.max())

    @property
    def n_leaves(self):
        return int(np.sum(self.children_left == -1))

    def predict(self, X):
        """Return per-class probabilities via FIL predict_per_tree."""
        X_gpu = cp.asarray(X, dtype=cp.float32)
        per_tree = self._fil_model.predict_per_tree(X_gpu)
        result = cp.asnumpy(per_tree[:, self._tree_idx])
        if result.ndim == 1:
            result = result[:, np.newaxis]
        return result

    def apply(self, X):
        """Return leaf node IDs via FIL apply."""
        X_gpu = cp.asarray(X, dtype=cp.float32)
        leaf_ids = self._fil_model.apply(X_gpu)
        return cp.asnumpy(leaf_ids[:, self._tree_idx]).astype(np.intp)

    def decision_path(self, X):
        """CPU fallback: traverse the stored tree structure."""
        from scipy.sparse import csr_matrix

        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]
        indptr = [0]
        indices = []

        for sample_idx in range(n_samples):
            node = 0
            while node != -1:
                indices.append(node)
                if self.children_left[node] == -1:
                    break
                if X[sample_idx, self.feature[node]] <= self.threshold[node]:
                    node = self.children_left[node]
                else:
                    node = self.children_right[node]
            indptr.append(len(indices))

        return csr_matrix(
            (np.ones(len(indices), dtype=np.uint8), indices, indptr),
            shape=(n_samples, self.node_count),
        )

    def compute_feature_importances(self, normalize=True):
        importances = np.zeros(self.n_features, dtype=np.float64)
        is_leaf = self.children_left == -1
        for i in range(self.node_count):
            if is_leaf[i]:
                continue
            left = self.children_left[i]
            right = self.children_right[i]
            w = self.weighted_n_node_samples
            importances[self.feature[i]] += (
                w[i] * self.impurity[i]
                - w[left] * self.impurity[left]
                - w[right] * self.impurity[right]
            )
        if normalize:
            total = importances.sum()
            if total > 0:
                importances /= total
        return importances


def _build_gpu_estimators(rf_model):
    """Build list of GPU-backed DecisionTree proxy objects from a fitted RF."""
    import treelite

    is_classifier = rf_model._estimator_type == "classifier"
    tl_model = treelite.Model.deserialize_bytes(rf_model._treelite_model_bytes)
    fil_model = rf_model._get_inference_fil_model()

    header = tl_model.get_header_accessor()
    n_classes = int(header.get_field("num_class")[0])
    n_features = int(header.get_field("num_feature")[0])
    n_trees = tl_model.num_tree

    estimators = []
    for tree_idx in range(n_trees):
        tree_data = _extract_tree_from_treelite(
            tl_model,
            tree_idx,
            n_classes,
            is_classifier,
        )
        gpu_tree = GPUTree(tree_data, fil_model, tree_idx)

        if is_classifier:
            est = GPUDecisionTreeClassifier.__new__(GPUDecisionTreeClassifier)
            est.classes_ = rf_model.classes_
            est.n_classes_ = rf_model.n_classes_
        else:
            est = GPUDecisionTreeRegressor.__new__(GPUDecisionTreeRegressor)

        est.tree_ = gpu_tree
        est.n_features_in_ = n_features
        est.n_outputs_ = 1
        est.max_features_ = n_features
        est._fil_model = fil_model
        est._tree_idx = tree_idx
        estimators.append(est)

    return estimators


class GPUDecisionTreeClassifier(sklearn.tree.DecisionTreeClassifier):
    """GPU-backed proxy for a single decision tree classifier in a forest."""

    def predict(self, X, check_input=True):
        X_gpu = cp.asarray(X, dtype=cp.float32)
        per_tree = self._fil_model.predict_per_tree(X_gpu)
        proba = cp.asnumpy(per_tree[:, self._tree_idx])
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)

    def predict_proba(self, X, check_input=True):
        X_gpu = cp.asarray(X, dtype=cp.float32)
        per_tree = self._fil_model.predict_per_tree(X_gpu)
        return cp.asnumpy(per_tree[:, self._tree_idx])

    def apply(self, X, check_input=True):
        X_gpu = cp.asarray(X, dtype=cp.float32)
        leaf_ids = self._fil_model.apply(X_gpu)
        return cp.asnumpy(leaf_ids[:, self._tree_idx]).astype(np.intp)

    @property
    def feature_importances_(self):
        return self.tree_.compute_feature_importances()


class GPUDecisionTreeRegressor(sklearn.tree.DecisionTreeRegressor):
    """GPU-backed proxy for a single decision tree regressor in a forest."""

    def predict(self, X, check_input=True):
        X_gpu = cp.asarray(X, dtype=cp.float32)
        per_tree = self._fil_model.predict_per_tree(X_gpu)
        result = cp.asnumpy(per_tree[:, self._tree_idx])
        if result.ndim > 1:
            result = result.squeeze(axis=-1)
        return result

    def apply(self, X, check_input=True):
        X_gpu = cp.asarray(X, dtype=cp.float32)
        leaf_ids = self._fil_model.apply(X_gpu)
        return cp.asnumpy(leaf_ids[:, self._tree_idx]).astype(np.intp)

    @property
    def feature_importances_(self):
        return self.tree_.compute_feature_importances()
