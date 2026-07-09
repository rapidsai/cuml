# Modified cuml.metrics module to fix label-permutation invariance issue
import numpy as np
from cuml.common import utils
from cuml.common import logger
from cuml.common.exceptions import CumlValueError

def _check_labels(labels):
    """Check if labels are valid"""
    if not isinstance(labels, np.ndarray):
        raise CumlValueError("Labels must be a numpy array")
    if not labels.dtype.kind in ['i', 'u']:
        raise CumlValueError("Labels must be integers")
    if labels.size == 0:
        raise CumlValueError("Labels must not be empty")

def homogeneity_score(y_true, y_pred):
    """Compute homogeneity score between two clusterings.
    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won’t change the score value in any way.
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth class labels.
    y_pred : array-like, shape (n_samples,)
        Cluster labels to evaluate.
    Returns
    -------
    homogeneity : float
        Homogeneity of the clustering.
    """
    _check_labels(y_true)
    _check_labels(y_pred)
    unique_labels = np.unique(y_true)
    unique_pred_labels = np.unique(y_pred)
    if len(unique_labels) != len(unique_pred_labels):
        raise CumlValueError("Number of unique true labels and predicted labels must be equal")
    # Sort labels to ensure permutation invariance
    sorted_labels = np.sort(unique_labels)
    sorted_pred_labels = np.sort(unique_pred_labels)
    # Compute contingency matrix
    contingency = np.zeros((len(unique_labels), len(unique_pred_labels)), dtype=np.int32)
    for i, label in enumerate(y_true):
        contingency[np.searchsorted(sorted_labels, label), np.searchsorted(sorted_pred_labels, y_pred[i])] += 1
    # Compute homogeneity score
    homogeneity = np.sum(contingency * contingency) / np.sum(contingency)
    return homogeneity

def completeness_score(y_true, y_pred):
    """Compute completeness score between two clusterings.
    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won’t change the score value in any way.
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth class labels.
    y_pred : array-like, shape (n_samples,)
        Cluster labels to evaluate.
    Returns
    -------
    completeness : float
        Completeness of the clustering.
    """
    _check_labels(y_true)
    _check_labels(y_pred)
    unique_labels = np.unique(y_true)
    unique_pred_labels = np.unique(y_pred)
    if len(unique_labels) != len(unique_pred_labels):
        raise CumlValueError("Number of unique true labels and predicted labels must be equal")
    # Sort labels to ensure permutation invariance
    sorted_labels = np.sort(unique_labels)
    sorted_pred_labels = np.sort(unique_pred_labels)
    # Compute contingency matrix
    contingency = np.zeros((len(unique_labels), len(unique_pred_labels)), dtype=np.int32)
    for i, label in enumerate(y_true):
        contingency[np.searchsorted(sorted_labels, label), np.searchsorted(sorted_pred_labels, y_pred[i])] += 1
    # Compute completeness score
    completeness = np.sum(contingency * contingency) / np.sum(contingency)
    return completeness

def v_measure_score(y_true, y_pred):
    """Compute v-measure score between two clusterings.
    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won’t change the score value in any way.
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth class labels.
    y_pred : array-like, shape (n_samples,)
        Cluster labels to evaluate.
    Returns
    -------
    v_measure : float
        V-measure of the clustering.
    """
    _check_labels(y_true)
    _check_labels(y_pred)
    unique_labels = np.unique(y_true)
    unique_pred_labels = np.unique(y_pred)
    if len(unique_labels) != len(unique_pred_labels):
        raise CumlValueError("Number of unique true labels and predicted labels must be equal")
    # Sort labels to ensure permutation invariance
    sorted_labels = np.sort(unique_labels)
    sorted_pred_labels = np.sort(unique_pred_labels)
    # Compute contingency matrix
    contingency = np.zeros((len(unique_labels), len(unique_pred_labels)), dtype=np.int32)
    for i, label in enumerate(y_true):
        contingency[np.searchsorted(sorted_labels, label), np.searchsorted(sorted_pred_labels, y_pred[i])] += 1
    # Compute v-measure score
    homogeneity = np.sum(contingency * contingency) / np.sum(contingency)
    completeness = np.sum(contingency * contingency) / np.sum(contingency)
    v_measure = 2 * homogeneity * completeness / (homogeneity + completeness)
    return v_measure

def mutual_info_score(y_true, y_pred):
    """Compute mutual information between two clusterings.
    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won’t change the score value in any way.
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth class labels.
    y_pred : array-like, shape (n_samples,)
        Cluster labels to evaluate.
    Returns
    -------
    mutual_info : float
        Mutual information of the clustering.
    """
    _check_labels(y_true)
    _check_labels(y_pred)
    unique_labels = np.unique(y_true)
    unique_pred_labels = np.unique(y_pred)
    if len(unique_labels) != len(unique_pred_labels):
        raise CumlValueError("Number of unique true labels and predicted labels must be equal")
    # Sort labels to ensure permutation invariance
    sorted_labels = np.sort(unique_labels)
    sorted_pred_labels = np.sort(unique_pred_labels)
    # Compute contingency matrix
    contingency = np.zeros((len(unique_labels), len(unique_pred_labels)), dtype=np.int32)
    for i, label in enumerate(y_true):
        contingency[np.searchsorted(sorted_labels, label), np.searchsorted(sorted_pred_labels, y_pred[i])] += 1
    # Compute mutual information score
    mutual_info = 0
    for i in range(len(unique_labels)):
        for j in range(len(unique_pred_labels)):
            mutual_info += contingency[i, j] * np.log2(contingency[i, j] / np.sum(contingency[i, :]) / np.sum(contingency[:, j]))
    return mutual_info