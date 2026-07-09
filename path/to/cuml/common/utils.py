# Modified cuml.common.utils module to add sort_labels function
import numpy as np

def sort_labels(labels):
    """Sort labels to ensure permutation invariance"""
    unique_labels = np.unique(labels)
    sorted_labels = np.sort(unique_labels)
    return sorted_labels