import cudf
import cupy as cp
import pytest

from cuml.model_selection import StratifiedKFold


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("n_splits", [3, 5, 10])
def test_split_dataframe(n_splits, shuffle):
    N = 1200  # number of samples
    M = 4  # number of classes
    X = cudf.DataFrame({"x": range(N)})
    y = cp.arange(1200) % M
    cp.random.shuffle(y)
    y = cudf.Series(y)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
    for train_index, test_index in kf.split(X, y):
        assert len(train_index)+len(test_index) == N
        assert len(train_index) == len(test_index)*(n_splits-1)
        for i in range(M):
            ratio_tr = (y[train_index] == i).sum() / len(train_index)
            ratio_te = (y[test_index] == i).sum() / len(test_index)
            assert ratio_tr == ratio_te
