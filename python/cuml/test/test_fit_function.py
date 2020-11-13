import pytest
import cuml
from cuml.test.utils import ClassEnumerator
import numpy as np
import cupy as cp

from sklearn.datasets import make_classification


def func_positional_arg(func):

    if hasattr(func, "__wrapped__"):
        return func_positional_arg(func.__wrapped__)

    elif hasattr(func, "__code__"):
        all_args = func.__code__.co_argcount
        if func.__defaults__ is not None:
            kwargs = len(func.__defaults__)
        else:
            kwargs = 0
        return all_args - kwargs
    return 2


@pytest.fixture(scope="session")
def dataset():
    X, y = make_classification(100, 5, random_state=42)
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    return X, y


models_config = ClassEnumerator(module=cuml)
models = models_config.get_models()


@pytest.mark.parametrize("model_name", list(models.keys()))
def test_fit_function(dataset, model_name):
    if model_name in [
        "SparseRandomProjection",
        "TSNE",
        "TruncatedSVD",
        "AutoARIMA",
        "MultinomialNB",
        "LabelEncoder",
    ]:
        pytest.xfail("These models are not tested yet")

    n_pos_args_constr = func_positional_arg(models[model_name].__init__)

    if model_name in ["SparseRandomProjection", "GaussianRandomProjection"]:
        model = models[model_name](n_components=2)
    elif model_name in ["ARIMA", "AutoARIMA", "ExponentialSmoothing"]:
        model = models[model_name](np.random.normal(0.0, 1.0, (10,)))
    else:
        if n_pos_args_constr == 1:
            model = models[model_name]()
        elif n_pos_args_constr == 2:
            model = models[model_name](5)
        else:
            model = models[model_name](5, 5)

    if hasattr(model, "fit"):
        # Unfortunately co_argcount doesn't work with decorated functions,
        # and the inspect module doesn't work with Cython. Therefore we need
        # to register the number of arguments manually if `fit` is decorated
        pos_args_spec = {
            "ARIMA": 1
        }
        n_pos_args_fit = (
            pos_args_spec[model_name]
            if model_name in pos_args_spec
            else func_positional_arg(models[model_name].fit)
        )

        X, y = dataset

        if model_name == "RandomForestClassifier":
            y = y.astype(np.int32)
            assert model.fit(X, y) is model
        else:
            if n_pos_args_fit == 1:
                assert model.fit() is model
            elif n_pos_args_fit == 2:
                assert model.fit(X) is model
            else:
                assert model.fit(X, y) is model

        # test classifiers correctly set self.classes_ during fit
        if hasattr(model, "_estimator_type"):
            if model._estimator_type == "classifier":
                cp.testing.assert_array_almost_equal(
                    model.classes_, np.unique(y)
                )
