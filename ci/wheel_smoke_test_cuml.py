"""
A simple test for cuML based on scikit-learn.

Adapted from xgboost:
https://raw.githubusercontent.com/rapidsai/xgboost-conda/branch-23.02/recipes/xgboost/test-py-xgboost.py
"""
from cuml.ensemble import RandomForestClassifier as curfc
import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics

X, y = sklearn.datasets.load_iris(return_X_y=True)
Xtrn, Xtst, ytrn, ytst = sklearn.model_selection.train_test_split(
    X, y, train_size=0.8, random_state=4)

clf = curfc(
    max_depth=2,
    n_estimators=10,
    n_bins=32,
    accuracy_metric="multi:softmax")
clf.fit(Xtrn, ytrn)
ypred = clf.predict(Xtst)
acc = sklearn.metrics.accuracy_score(ytst, ypred)

print('cuml RandomForestClassifier accuracy on iris:', acc)
assert acc > 0.9
