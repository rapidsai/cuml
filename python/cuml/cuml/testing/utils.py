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
import pytest
from cuml.internals.mem_type import MemoryType
from cuml.internals.input_utils import input_to_cuml_array, is_array_like
from cuml.internals.base import Base
import cuml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, brier_score_loss
from sklearn.datasets import make_classification, make_regression
from sklearn import datasets
from pylibraft.common.cuda import Stream
from sklearn.datasets import make_regression as skl_make_reg
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numbers import Number
from cuml.internals.safe_imports import gpu_only_import_from
from itertools import dropwhile
from copy import deepcopy
from cuml.internals.safe_imports import cpu_only_import
import inspect
from textwrap import dedent, indent

from cuml.internals.safe_imports import gpu_only_import

cp = gpu_only_import("cupy")
np = cpu_only_import("numpy")
pd = cpu_only_import("pandas")

cuda = gpu_only_import_from("numba", "cuda")
cudf_pandas_active = gpu_only_import_from("cudf.pandas", "LOADED")


cudf = gpu_only_import("cudf")


def array_difference(a, b, with_sign=True):
    """
    Utility function to compute the difference between 2 arrays.
    """
    a = to_nparray(a)
    b = to_nparray(b)

    if len(a) == 0 and len(b) == 0:
        return 0

    if not with_sign:
        a, b = np.abs(a), np.abs(b)
    return np.sum(np.abs(a - b))


class array_equal:
    """
    Utility functor to compare 2 numpy arrays and optionally show a meaningful
    error message in case they are not. Two individual elements are assumed
    equal if they are within `unit_tol` of each other, and two arrays are
    considered equal if less than `total_tol` percentage of elements are
    different.

    """

    def __init__(self, a, b, unit_tol=1e-4, total_tol=1e-4, with_sign=True):
        self.a = to_nparray(a)
        self.b = to_nparray(b)
        self.unit_tol = unit_tol
        self.total_tol = total_tol
        self.with_sign = with_sign

    def compute_difference(self):
        return array_difference(self.a, self.b, with_sign=self.with_sign)

    def __bool__(self):
        if len(self.a) == len(self.b) == 0:
            return True

        if self.with_sign:
            a, b = self.a, self.b
        else:
            a, b = np.abs(self.a), np.abs(self.b)

        res = (np.sum(np.abs(a - b) > self.unit_tol)) / a.size < self.total_tol
        return bool(res)

    def __eq__(self, other):
        if isinstance(other, bool):
            return bool(self) == other
        return super().__eq__(other)

    def _repr(self, threshold=None):
        name = self.__class__.__name__

        return (
            [f"<{name}: "]
            + f"{np.array2string(self.a, threshold=threshold)} ".splitlines()
            + f"{np.array2string(self.b, threshold=threshold)} ".splitlines()
            + [
                f"unit_tol={self.unit_tol} ",
                f"total_tol={self.total_tol} ",
                f"with_sign={self.with_sign}",
                ">",
            ]
        )

    def __repr__(self):
        return "".join(self._repr(threshold=5))

    def __str__(self):
        tokens = self._repr(threshold=1000)
        return "\n".join(
            f"{'    ' if 0 < n < len(tokens) - 1 else ''}{token}"
            for n, token in enumerate(tokens)
        )


def assert_array_equal(a, b, unit_tol=1e-4, total_tol=1e-4, with_sign=True):
    """
    Raises an AssertionError if arrays are not considered equal.

    Uses the same arguments as array_equal(), but raises an AssertionError in
    case that the test considers the arrays to not be equal.

    This function will generate a nicer error message in the context of pytest
    compared to a plain `assert array_equal(...)`.
    """
    # Determine array equality.
    equal = array_equal(
        a, b, unit_tol=unit_tol, total_tol=total_tol, with_sign=with_sign
    )
    if not equal:
        # Generate indented array string representation ...
        str_a = indent(np.array2string(a), "   ").splitlines()
        str_b = indent(np.array2string(b), "   ").splitlines()
        # ... and add labels
        str_a[0] = f"a: {str_a[0][3:]}"
        str_b[0] = f"b: {str_b[0][3:]}"

        # Create assertion error message and raise exception.
        assertion_error_msg = (
            dedent(
                f"""
        Arrays are not equal

        unit_tol:  {unit_tol}
        total_tol: {total_tol}
        with_sign: {with_sign}

        """
            )
            + "\n".join(str_a + str_b)
        )
        raise AssertionError(assertion_error_msg)


def get_pattern(name, n_samples):
    np.random.seed(0)
    random_state = 170

    if name == "noisy_circles":
        data = datasets.make_circles(
            n_samples=n_samples, factor=0.5, noise=0.05
        )
        params = {
            "damping": 0.77,
            "preference": -240,
            "quantile": 0.2,
            "n_clusters": 2,
        }

    elif name == "noisy_moons":
        data = datasets.make_moons(n_samples=n_samples, noise=0.05)
        params = {"damping": 0.75, "preference": -220, "n_clusters": 2}

    elif name == "varied":
        data = datasets.make_blobs(
            n_samples=n_samples,
            cluster_std=[1.0, 2.5, 0.5],
            random_state=random_state,
        )
        params = {"eps": 0.18, "n_neighbors": 2}

    elif name == "blobs":
        data = datasets.make_blobs(n_samples=n_samples, random_state=8)
        params = {}

    elif name == "aniso":
        X, y = datasets.make_blobs(
            n_samples=n_samples, random_state=random_state
        )
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_aniso = np.dot(X, transformation)
        data = (X_aniso, y)
        params = {"eps": 0.15, "n_neighbors": 2}

    elif name == "no_structure":
        data = np.random.rand(n_samples, 2), None
        params = {}

    return [data, params]


def normalize_clusters(a0, b0, n_clusters):
    a = to_nparray(a0)
    b = to_nparray(b0)

    c = deepcopy(b)

    for i in range(n_clusters):
        (idx,) = np.where(a == i)
        a_to_b = c[idx[0]]
        b[c == a_to_b] = i

    return a, b


def as_type(type, *args):
    # Convert array args to type supported by
    # CumlArray.to_output ('numpy','cudf','cupy'...)
    # Ensure 2 dimensional inputs are not converted to 1 dimension
    # None remains as None
    # Scalar remains a scalar
    result = []
    for arg in args:
        if arg is None or np.isscalar(arg):
            result.append(arg)
        else:
            # make sure X with a single feature remains 2 dimensional
            if type in ("cudf", "pandas", "df_obj") and len(arg.shape) > 1:
                if type == "pandas":
                    mem_type = MemoryType.host
                else:
                    mem_type = None
                result.append(
                    input_to_cuml_array(arg).array.to_output(
                        output_type="dataframe", output_mem_type=mem_type
                    )
                )
            else:
                result.append(input_to_cuml_array(arg).array.to_output(type))
    if len(result) == 1:
        return result[0]
    return tuple(result)


def to_nparray(x):
    if isinstance(x, Number):
        return np.asarray([x])
    elif isinstance(x, pd.DataFrame):
        return x.values
    elif isinstance(x, cudf.DataFrame):
        return x.to_pandas().values
    elif isinstance(x, cudf.Series):
        return x.to_pandas().values
    elif isinstance(x, DeviceNDArray):
        return x.copy_to_host()
    elif isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def clusters_equal(a0, b0, n_clusters, tol=1e-4):
    a, b = normalize_clusters(a0, b0, n_clusters)
    return array_equal(a, b, total_tol=tol)


def assert_dbscan_equal(ref, actual, X, core_indices, eps):
    """
    Utility function to compare two numpy label arrays.
    The labels of core/noise points are expected to be equal, and the labels
    of border points are verified by finding a neighboring core point with the
    same label.
    """
    core_set = set(core_indices)
    N, _ = X.shape
    eps2 = eps**2

    def sqnorm(x):
        return np.inner(x, x)

    for i in range(N):
        la, lb = ref[i], actual[i]

        if i in core_set:  # core point
            assert (
                la == lb
            ), "Core point mismatch at #{}: " "{} (expected {})".format(
                i, lb, la
            )
        elif la == -1:  # noise point
            assert lb == -1, "Noise mislabelled at #{}: {}".format(i, lb)
        else:  # border point
            found = False
            for j in range(N):
                # Check if j is a core point with the same label
                if j in core_set and lb == actual[j]:
                    # Check if j is a neighbor of i
                    if sqnorm(X[i] - X[j]) <= eps2:
                        found = True
                        break
            assert found, (
                "Border point not connected to cluster at #{}: "
                "{} (reference: {})".format(i, lb, la)
            )

    # Note: we can also do it in a rand score fashion by checking that pairs
    # correspond in both label arrays for core points, if we need to drop the
    # requirement of minimality for core points


def get_handle(use_handle, n_streams=0):
    if not use_handle:
        return None, None
    s = Stream()
    h = cuml.Handle(stream=s, n_streams=n_streams)
    return h, s


def small_regression_dataset(datatype):
    X, y = make_regression(
        n_samples=1000, n_features=20, n_informative=10, random_state=10
    )
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    return X_train, X_test, y_train, y_test


def small_classification_dataset(datatype):
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_classes=2,
        random_state=10,
    )
    X = X.astype(datatype)
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    return X_train, X_test, y_train, y_test


def unit_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.unit)


def quality_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.quality)


def stress_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.stress)


class ClassEnumerator:
    """Helper class to automatically pick up every models classes in a module.
    Filters out classes not inheriting from cuml.Base.

    Parameters
    ----------
    module: python module (ex: cuml.linear_regression)
        The module for which to retrieve models.
    exclude_classes: list of classes (optional)
        Those classes will be filtered out from the retrieved models.
    custom_constructors: dictionary of {class_name: lambda}
        Custom constructors to use instead of the default one.
        ex: {'LogisticRegression': lambda: cuml.LogisticRegression(handle=1)}
    recursive: bool, default=False
        Instructs the class to recursively search submodules when True,
        otherwise only classes in the specified model will be enumerated
    """

    def __init__(
        self,
        module,
        exclude_classes=None,
        custom_constructors=None,
        recursive=False,
    ):
        self.module = module
        self.exclude_classes = exclude_classes or []
        self.custom_constructors = custom_constructors or []
        self.recursive = recursive

    def _get_classes(self):
        def recurse_module(module):
            classes = {}

            modules = []

            if self.recursive:
                modules = inspect.getmembers(module, inspect.ismodule)

            # Enumerate child modules only if they are a submodule of the
            # current one. i.e. `{parent_module}.{submodule}`
            for _, m in modules:
                if module.__name__ + "." in m.__name__:
                    classes.update(recurse_module(m))

            # Ensure we only get classes that are part of this module
            classes.update(
                {
                    (".".join((klass.__module__, klass.__qualname__))): klass
                    for name, klass in inspect.getmembers(
                        module, inspect.isclass
                    )
                    if module.__name__ + "."
                    in ".".join((klass.__module__, klass.__qualname__))
                }
            )

            return classes

        return [
            (val.__name__, val)
            for key, val in recurse_module(self.module).items()
        ]

    def get_models(self):
        """Picks up every models classes from self.module.
        Filters out classes not inheriting from cuml.Base.

        Returns
        -------
        models: dictionary of {class_name: class|class_constructor}
            Dictionary of models in the module, except when a
            custom_constructor is specified, in that case the value is the
            specified custom_constructor.
        """
        classes = self._get_classes()
        models = {
            name: cls
            for name, cls in classes
            if cls not in self.exclude_classes and issubclass(cls, Base)
        }
        models.update(self.custom_constructors)
        return models


def get_classes_from_package(package, import_sub_packages=False):
    """
    Gets all modules imported in the specified package and returns a dictionary
    of any classes that derive from `cuml.Base`

    Parameters
    ----------
    package : python module The python module to search import_sub_packages :
        bool, default=False When set to True, will try to import sub packages
        by searching the directory tree for __init__.py files and importing
        them accordingly. By default this is set to False

    Returns
    -------
    ClassEnumerator Class enumerator for the specified package
    """

    if import_sub_packages:
        import os
        import importlib

        # First, find all __init__.py files in subdirectories of this package
        root_dir = os.path.dirname(package.__file__)

        root_relative = os.path.dirname(root_dir)

        # Now loop
        for root, _, files in os.walk(root_dir):

            if "__init__.py" in files:

                module_name = os.path.relpath(root, root_relative).replace(
                    os.sep, "."
                )

                importlib.import_module(module_name)

    return ClassEnumerator(module=package, recursive=True).get_models()


def generate_random_labels(random_generation_lambda, seed=1234, as_cupy=False):
    """
    Generates random labels to act as ground_truth and predictions for tests.

    Parameters
    ----------
    random_generation_lambda : lambda function [numpy.random] -> ndarray
        A lambda function used to generate labels for either y_true or y_pred
        using a seeded numpy.random object.
    seed : int
        Seed for the numpy.random object.
    as_cupy : bool
        Choose return type of y_true and y_pred.
        True: returns Cupy ndarray
        False: returns Numba cuda DeviceNDArray

    Returns
    -------
    y_true, y_pred, np_y_true, np_y_pred : tuple
        y_true : Numba cuda DeviceNDArray or Cupy ndarray
            Random target values.
        y_pred : Numba cuda DeviceNDArray or Cupy ndarray
            Random predictions.
        np_y_true : Numpy ndarray
            Same as y_true but as a numpy ndarray.
        np_y_pred : Numpy ndarray
            Same as y_pred but as a numpy ndarray.
    """
    rng = np.random.RandomState(seed)  # makes it reproducible
    a = random_generation_lambda(rng)
    b = random_generation_lambda(rng)

    if as_cupy:
        return cp.array(a), cp.array(b), a, b
    else:
        return cuda.to_device(a), cuda.to_device(b), a, b


def score_labeling_with_handle(
    func, ground_truth, predictions, use_handle, dtype=np.int32
):
    """Test helper to standardize inputs between sklearn and our prims metrics.

    Using this function we can pass python lists as input of a test just like
    with sklearn as well as an option to use handle with our metrics.
    """
    a = cp.array(ground_truth, dtype=dtype)
    b = cp.array(predictions, dtype=dtype)

    handle, stream = get_handle(use_handle)

    return func(a, b, handle=handle)


def get_number_positional_args(func, default=2):
    # function to return number of positional arguments in func
    if hasattr(func, "__code__"):
        all_args = func.__code__.co_argcount
        if func.__defaults__ is not None:
            kwargs = len(func.__defaults__)
        else:
            kwargs = 0
        return all_args - kwargs
    return default


def get_shap_values(
    model,
    explainer,
    background_dataset,
    explained_dataset,
    api_type="shap_values",
):
    # function to get shap values from an explainer using SHAP style API.
    # This function allows isolating all calls in test suite for the case of
    # API changes.
    explainer = explainer(model=model, data=background_dataset)
    if api_type == "shap_values":
        shap_values = explainer.shap_values(explained_dataset)
    elif api_type == "__call__":
        shap_values = explainer(explained_dataset)

    return explainer, shap_values


def generate_inputs_from_categories(
    categories=None, n_samples=10, seed=5060, as_array=False
):
    if categories is None:
        if as_array:
            categories = {
                "strings": list(range(1000, 4000, 3)),
                "integers": list(range(1000)),
            }
        else:
            categories = {
                "strings": ["Foo", "Bar", "Baz"],
                "integers": list(range(1000)),
            }

    rd = np.random.RandomState(seed)
    pandas_df = pd.DataFrame(
        {name: rd.choice(cat, n_samples) for name, cat in categories.items()}
    )
    ary = from_df_to_numpy(pandas_df)
    if as_array:
        inp_ary = cp.array(ary)
        return inp_ary, ary
    else:
        if cudf_pandas_active:
            df = pandas_df
        else:
            df = cudf.DataFrame.from_pandas(pandas_df)
        return df, ary


def assert_inverse_equal(ours, ref):
    if isinstance(ours, cp.ndarray):
        cp.testing.assert_array_equal(ours, ref)
    else:
        if hasattr(ours, "to_pandas"):
            ours = ours.to_pandas()
        if hasattr(ref, "to_pandas"):
            ref = ref.to_pandas()
        pd.testing.assert_frame_equal(ours, ref)


def from_df_to_numpy(df):
    if isinstance(df, pd.DataFrame):
        return list(zip(*[df[feature] for feature in df.columns]))
    else:
        return list(zip(*[df[feature].values_host for feature in df.columns]))


def compare_svm(
    svm1,
    svm2,
    X,
    y,
    b_tol=None,
    coef_tol=None,
    report_summary=False,
    cmp_decision_func=False,
):
    """Compares two svm classifiers
    Parameters:
    -----------
    svm1 : svm classifier to be tested
    svm2 : svm classifier, the correct model
    b_tol : float
        tolerance while comparing the constant in the decision functions
    coef_tol: float
        tolerance used while comparing coef_ attribute for linear SVM

    Support vector machines have a decision function:

    F(x) = sum_{i=1}^{n_sv} d_i K(x_i, x) + b,

    where n_sv is the number of support vectors, K is the kernel function, x_i
    are the support vectors, d_i are the dual coefficients (more precisely
    d = alpha_i * y_i, where alpha_i is the dual coef), and b is the intercept.

    For linear svms K(x_i, x) = x_i * x, and we can simplify F by introducing
    w = sum_{i=1}^{n_sv} d_i x_i, the normal vector of the separating
    hyperplane:

    F(x) = w * x + b.

    Mathematically the solution of the optimization should be unique, which
    means w and b should be unique.

    There could be multiple set of vectors that lead to the same w, therefore
    comparing parameters d_k, n_sv or the support vector indices can lead to
    false positives.

    We can only evaluate w for linear models, for nonlinear models we can only
    test model accuracy and intercept.
    """

    n = X.shape[0]
    accuracy1 = svm1.score(X, y)
    accuracy2 = svm2.score(X, y)

    # We use at least 0.1% tolerance for accuracy comparison
    accuracy_tol_min = 0.001
    if accuracy2 < 1:
        # Set tolerance to include the 95% confidence interval of svm2's
        # accuracy. In practice this gives 0.9% tolerance for a 90% accurate
        # model (assuming n_test = 4000).
        accuracy_tol = 1.96 * np.sqrt(accuracy2 * (1 - accuracy2) / n)
        if accuracy_tol < accuracy_tol_min:
            accuracy_tol = accuracy_tol_min
    else:
        accuracy_tol = accuracy_tol_min

    assert accuracy1 >= accuracy2 - accuracy_tol

    if b_tol is None:
        b_tol = 100 * svm1.tol  # Using default tol=1e-3 leads to b_tol=0.1

    if accuracy2 < 0.5:
        # Increase error margin for classifiers that are not accurate.
        # Although analytically the classifier should always be the same,
        # we fit only until we reach a certain numerical tolerance, and
        # therefore the resulting SVM's can be different. We increase the
        # tolerance in these cases.
        #
        # A good example is the gaussian dataset with linear classifier:
        # the classes are concentric blobs, and we cannot separate that with a
        # straight line. When we have a large number of data points, then
        # any separating hyperplane that goes through the center would be good.
        b_tol *= 10
        if n >= 250:
            coef_tol = 2  # allow any direction
        else:
            coef_tol *= 10

    # Compare model parameter b (intercept). In practice some models can have
    # some differences in the model parameters while still being within
    # the accuracy tolerance.
    #
    # We skip this test for multiclass (when intercept_ is an array). Apart
    # from the larger discrepancies in multiclass case, sklearn also uses a
    # different sign convention for intercept in that case.
    if (not is_array_like(svm2.intercept_)) or svm2.intercept_.shape[0] == 1:
        if abs(svm2.intercept_) > 1e-6:
            assert (
                abs((svm1.intercept_ - svm2.intercept_) / svm2.intercept_)
                <= b_tol
            )
        else:
            assert abs((svm1.intercept_ - svm2.intercept_)) <= b_tol

    # For linear kernels we can compare the normal vector of the separating
    # hyperplane w, which is stored in the coef_ attribute.
    if svm1.kernel == "linear":
        if coef_tol is None:
            coef_tol = 1e-5
        cs = np.dot(svm1.coef_, svm2.coef_.T) / (
            np.linalg.norm(svm1.coef_) * np.linalg.norm(svm2.coef_)
        )
        assert cs > 1 - coef_tol

    if cmp_decision_func:
        if accuracy2 > 0.9 and svm1.kernel != "sigmoid":
            df1 = svm1.decision_function(X)
            df2 = svm2.decision_function(X)
            # For classification, the class is determined by
            # sign(decision function). We should not expect tight match for
            # the actual value of the function, therefore we set large tolerance
            assert svm_array_equal(
                df1, df2, tol=1e-1, relative_diff=True, report_summary=True
            )
        else:
            print(
                "Skipping decision function test due to low  accuracy",
                accuracy2,
            )

    # Compare support_ (dataset indices of points that form the support
    # vectors) and ensure that some overlap (~1/8) between two exists
    support1 = set(svm1.support_)
    support2 = set(svm2.support_)
    intersection_len = len(support1.intersection(support2))
    average_len = (len(support1) + len(support2)) / 2
    assert intersection_len > average_len / 8


def compare_probabilistic_svm(
    svc1, svc2, X_test, y_test, tol=1e-3, brier_tol=1e-3
):
    """Compare the probability output from two support vector classifiers."""

    prob1 = svc1.predict_proba(X_test)
    prob2 = svc2.predict_proba(X_test)
    assert mean_squared_error(prob1, prob2) <= tol

    if svc1.n_classes_ == 2:
        brier1 = brier_score_loss(y_test, prob1[:, 1])
        brier2 = brier_score_loss(y_test, prob2[:, 1])
        # Brier score - smaller is better
        assert brier1 - brier2 <= brier_tol


def create_synthetic_dataset(
    generator=skl_make_reg,
    n_samples=100,
    n_features=10,
    test_size=0.25,
    random_state_generator=None,
    random_state_train_test_split=None,
    dtype=np.float32,
    **kwargs,
):
    X, y = generator(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state_generator,
        **kwargs,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state_train_test_split
    )

    X_train = X_train.astype(dtype)
    X_test = X_test.astype(dtype)
    y_train = y_train.astype(dtype)
    y_test = y_test.astype(dtype)

    return X_train, X_test, y_train, y_test


def svm_array_equal(a, b, tol=1e-6, relative_diff=True, report_summary=False):
    diff = np.abs(a - b)
    if relative_diff:
        idx = np.nonzero(abs(b) > tol)
        diff[idx] = diff[idx] / abs(b[idx])
    equal = np.all(diff <= tol)
    if not equal and report_summary:
        idx = np.argsort(diff)
        print("Largest diffs")
        a = a.ravel()
        b = b.ravel()
        diff = diff.ravel()
        for i in idx[-5:]:
            if diff[i] > tol:
                print(diff[i], "at", i, "values", a[i], b[i])
        print(
            "Avgdiff:",
            np.mean(diff),
            "stddiyy:",
            np.std(diff),
            "avgval:",
            np.mean(b),
        )
    return equal


def normalized_shape(shape):
    """Normalize shape to tuple."""
    return (shape,) if isinstance(shape, int) else shape


def squeezed_shape(shape):
    """Remove all trailing axes of length 1 from shape.

    Similar to, but not exactly like np.squeeze().
    """
    return tuple(reversed(list(dropwhile(lambda d: d == 1, reversed(shape)))))


def series_squeezed_shape(shape):
    """Remove all but one axes of length 1 from shape."""
    if shape:
        return tuple([d for d in normalized_shape(shape) if d != 1]) or (1,)
    else:
        return ()
