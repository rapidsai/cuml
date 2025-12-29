import _expected_mutual_information
import numpy as np
import cupy as cp
from sklearn import metrics
import sys

def test_emi(true, pred):
    n_sample = len(true)
    contigency = metrics.cluster.contingency_matrix(true, pred)
    emi_sk =  metrics.cluster.expected_mutual_information(contigency, n_sample)
    contigency_cp = cp.array(contigency)
    emi = _expected_mutual_information.expected_mutual_information(contigency_cp, n_sample)
    return (emi_sk, emi)

print(test_emi([0, 1, 1, 0, 1, 0],[0, 1, 0, 0, 1, 1]))
print(test_emi([0, 0, 0, 1, 1, 1],[0, 0, 1, 1, 2, 2]))
print(test_emi([0, 1, 2, 0, 3, 4, 5, 1],[1, 1, 0, 0, 2, 2, 2, 2]))
print(test_emi([0, 0, 1, 1],[0, 0, 1, 1]))
# def main():
#     if len(sys.argv) != 3:
#         print(len(sys.argv))
#         print("Only use two arguments: {true and pred arrays}")
#         sys.exit(1)
#     print(test_emi(sys.argv[0], sys.argv[1]))

# if __name__ == "__main__":
#     main()
