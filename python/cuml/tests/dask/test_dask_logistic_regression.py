# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

from cuml.internals.safe_imports import gpu_only_import
import pytest
from cuml.dask.common import utils as dask_utils
from functools import partial
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as skLR
from cuml.internals.safe_imports import cpu_only_import
from cuml.testing.utils import array_equal
from scipy.sparse import csr_matrix

pd = cpu_only_import("pandas")
np = cpu_only_import("numpy")
cp = gpu_only_import("cupy")
dask_cudf = gpu_only_import("dask_cudf")
cudf = gpu_only_import("cudf")

pytestmark = pytest.mark.mg


def _prep_training_data(c, X_train, y_train, partitions_per_worker):
    workers = c.has_what().keys()
    n_partitions = partitions_per_worker * len(workers)
    X_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X_train))
    X_train_df = dask_cudf.from_cudf(X_cudf, npartitions=n_partitions)

    y_cudf = np.array(pd.DataFrame(y_train).values)
    y_cudf = y_cudf[:, 0]
    y_cudf = cudf.Series(y_cudf)
    y_train_df = dask_cudf.from_cudf(y_cudf, npartitions=n_partitions)

    X_train_df, y_train_df = dask_utils.persist_across_workers(
        c, [X_train_df, y_train_df], workers=workers
    )
    return X_train_df, y_train_df


def _prep_training_data_sparse(c, X_train, y_train, partitions_per_worker):
    "The implementation follows test_dask_tfidf.create_cp_sparse_dask_array"
    import dask.array as da

    workers = c.has_what().keys()
    target_n_partitions = partitions_per_worker * len(workers)

    def cal_chunks(dataset, n_partitions):

        n_samples = dataset.shape[0]
        n_samples_per_part = int(n_samples / n_partitions)
        chunk_sizes = [n_samples_per_part] * n_partitions
        samples_last_row = n_samples - (
            (n_partitions - 1) * n_samples_per_part
        )
        chunk_sizes[-1] = samples_last_row
        return tuple(chunk_sizes)

    assert (
        X_train.shape[0] == y_train.shape[0]
    ), "the number of data records is not equal to the number of labels"
    target_chunk_sizes = cal_chunks(X_train, target_n_partitions)

    X_da = da.from_array(X_train, chunks=(target_chunk_sizes, -1))
    y_da = da.from_array(y_train, chunks=target_chunk_sizes)

    X_da, y_da = dask_utils.persist_across_workers(
        c, [X_da, y_da], workers=workers
    )
    return X_da, y_da


def make_classification_dataset(datatype, nrows, ncols, n_info, n_classes=2):
    X, y = make_classification(
        n_samples=nrows,
        n_features=ncols,
        n_informative=n_info,
        n_classes=n_classes,
        random_state=0,
    )
    X = X.astype(datatype)
    y = y.astype(datatype)

    return X, y


def select_sk_solver(cuml_solver):
    if cuml_solver == "newton":
        return "newton-cg"
    elif cuml_solver in ["admm", "lbfgs"]:
        return "lbfgs"
    else:
        pytest.xfail("No matched sklearn solver")


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [1e5])
@pytest.mark.parametrize("ncols", [20])
@pytest.mark.parametrize("n_parts", [2, 6])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("gpu_array_input", [False, True])
@pytest.mark.parametrize(
    "solver", ["admm", "gradient_descent", "newton", "lbfgs", "proximal_grad"]
)
def test_lr_fit_predict_score(
    nrows,
    ncols,
    n_parts,
    fit_intercept,
    datatype,
    gpu_array_input,
    solver,
    client,
):
    sk_solver = select_sk_solver(cuml_solver=solver)

    def imp():
        import cuml.comm.serialize  # NOQA

    client.run(imp)

    from cuml.dask.extended.linear_model import (
        LogisticRegression as cumlLR_dask,
    )

    n_info = 5
    nrows = int(nrows)
    ncols = int(ncols)
    X, y = make_classification_dataset(datatype, nrows, ncols, n_info)

    gX, gy = _prep_training_data(client, X, y, n_parts)

    if gpu_array_input:
        gX = gX.values
        gX._meta = cp.asarray(gX._meta)
        gy = gy.values
        gy._meta = cp.asarray(gy._meta)

    cuml_model = cumlLR_dask(
        fit_intercept=fit_intercept, solver=solver, max_iter=10
    )

    # test fit and predict
    cuml_model.fit(gX, gy)
    cu_preds = cuml_model.predict(gX)
    accuracy_cuml = accuracy_score(y, cu_preds.compute().get())

    sk_model = skLR(fit_intercept=fit_intercept, solver=sk_solver, max_iter=10)
    sk_model.fit(X, y)
    sk_preds = sk_model.predict(X)
    accuracy_sk = accuracy_score(y, sk_preds)

    assert (accuracy_cuml >= accuracy_sk) | (
        np.abs(accuracy_cuml - accuracy_sk) < 1e-3
    )

    # score
    accuracy_cuml = cuml_model.score(gX, gy).compute().item()
    accuracy_sk = sk_model.score(X, y)

    assert (accuracy_cuml >= accuracy_sk) | (
        np.abs(accuracy_cuml - accuracy_sk) < 1e-3
    )

    # predicted probabilities should differ by <= 5%
    # even with different solvers (arbitrary)
    probs_cuml = cuml_model.predict_proba(gX).compute()
    probs_sk = sk_model.predict_proba(X)[:, 1]
    assert np.abs(probs_sk - probs_cuml.get()).max() <= 0.05


@pytest.mark.mg
@pytest.mark.parametrize("n_parts", [2])
@pytest.mark.parametrize("datatype", [np.float32])
def test_lbfgs_toy(n_parts, datatype, client):
    def imp():
        import cuml.comm.serialize  # NOQA

    client.run(imp)

    X = np.array([(1, 2), (1, 3), (2, 1), (3, 1)], datatype)
    y = np.array([1.0, 1.0, 0.0, 0.0], datatype)

    from cuml.dask.linear_model import LogisticRegression as cumlLBFGS_dask

    X_df, y_df = _prep_training_data(client, X, y, n_parts)

    lr = cumlLBFGS_dask()

    lr.fit(X_df, y_df)

    lr_coef = lr.coef_.to_numpy()
    lr_intercept = lr.intercept_.to_numpy()

    assert len(lr_coef) == 1
    assert lr_coef[0] == pytest.approx([-0.71483153, 0.7148315], abs=1e-6)
    assert lr_intercept == pytest.approx([-2.2614916e-08], abs=1e-6)

    # test predict
    preds = lr.predict(X_df, delayed=True).compute().to_numpy()
    from numpy.testing import assert_array_equal

    assert_array_equal(preds, y, strict=True)

    # assert error on float64
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    X_df, y_df = _prep_training_data(client, X, y, n_parts)
    with pytest.raises(
        RuntimeError,
        match="dtypes other than float32 are currently not supported yet. See issue: https://github.com/rapidsai/cuml/issues/5589",
    ):
        lr.fit(X_df, y_df)


def test_lbfgs_init(client):
    def imp():
        import cuml.comm.serialize  # NOQA

    client.run(imp)

    X = np.array([(1, 2), (1, 3), (2, 1), (3, 1)], dtype=np.float32)
    y = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)

    X_df, y_df = _prep_training_data(
        c=client, X_train=X, y_train=y, partitions_per_worker=2
    )

    from cuml.dask.linear_model.logistic_regression import (
        LogisticRegression as cumlLBFGS_dask,
    )

    def assert_params(
        tol,
        C,
        fit_intercept,
        max_iter,
        linesearch_max_iter,
        verbose,
        output_type,
    ):

        lr = cumlLBFGS_dask(
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            linesearch_max_iter=linesearch_max_iter,
            verbose=verbose,
            output_type=output_type,
        )

        lr.fit(X_df, y_df)
        qnpams = lr.qnparams.params
        assert qnpams["grad_tol"] == tol
        assert qnpams["loss"] == 0  # "sigmoid" loss
        assert qnpams["penalty_l1"] == 0.0
        assert qnpams["penalty_l2"] == 1.0 / C
        assert qnpams["fit_intercept"] == fit_intercept
        assert qnpams["max_iter"] == max_iter
        assert qnpams["linesearch_max_iter"] == linesearch_max_iter
        assert (
            qnpams["verbose"] == 5 if verbose is True else 4
        )  # cuml Verbosity Levels
        assert (
            lr.output_type == "input" if output_type is None else output_type
        )  # cuml.global_settings.output_type

    assert_params(
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        max_iter=1000,
        linesearch_max_iter=50,
        verbose=False,
        output_type=None,
    )

    assert_params(
        tol=1e-6,
        C=1.5,
        fit_intercept=False,
        max_iter=200,
        linesearch_max_iter=100,
        verbose=True,
        output_type="cudf",
    )


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [1e5])
@pytest.mark.parametrize("ncols", [20])
@pytest.mark.parametrize("n_parts", [2, 23])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("datatype", [np.float32])
@pytest.mark.parametrize("delayed", [True, False])
def test_lbfgs(
    nrows,
    ncols,
    n_parts,
    fit_intercept,
    datatype,
    delayed,
    client,
    penalty="l2",
    l1_ratio=None,
    C=1.0,
    n_classes=2,
    convert_to_sparse=False,
):
    tolerance = 0.005

    def imp():
        import cuml.comm.serialize  # NOQA

    client.run(imp)

    from cuml.dask.linear_model.logistic_regression import (
        LogisticRegression as cumlLBFGS_dask,
    )

    # set n_informative variable for calling sklearn.datasets.make_classification
    n_info = 5
    nrows = int(nrows)
    ncols = int(ncols)
    X, y = make_classification_dataset(
        datatype, nrows, ncols, n_info, n_classes=n_classes
    )

    if convert_to_sparse:
        # X_dask and y_dask are dask array
        X_dask, y_dask = _prep_training_data_sparse(client, X, y, n_parts)
    else:
        # X_dask and y_dask are dask cudf
        X_dask, y_dask = _prep_training_data(client, X, y, n_parts)

    lr = cumlLBFGS_dask(
        solver="qn",
        fit_intercept=fit_intercept,
        penalty=penalty,
        l1_ratio=l1_ratio,
        C=C,
        verbose=True,
    )
    lr.fit(X_dask, y_dask)

    def array_to_numpy(ary):
        if isinstance(ary, cp.ndarray):
            return cp.asarray(ary)
        elif isinstance(ary, cudf.DataFrame) or isinstance(ary, cudf.Series):
            return ary.to_numpy()
        else:
            assert isinstance(ary, np.ndarray)
            return ary

    lr_coef = array_to_numpy(lr.coef_)
    lr_intercept = array_to_numpy(lr.intercept_)

    if penalty == "l2" or penalty == "none":
        sk_solver = "lbfgs"
    elif penalty == "l1" or penalty == "elasticnet":
        sk_solver = "saga"
    else:
        raise ValueError(f"unexpected penalty {penalty}")

    sk_model = skLR(
        solver=sk_solver,
        fit_intercept=fit_intercept,
        penalty=penalty,
        l1_ratio=l1_ratio,
        C=C,
    )
    sk_model.fit(X, y)
    sk_coef = sk_model.coef_
    sk_intercept = sk_model.intercept_

    if sk_solver == "lbfgs":
        assert len(lr_coef) == len(sk_coef)
        assert array_equal(lr_coef, sk_coef, tolerance, with_sign=True)
        assert array_equal(
            lr_intercept, sk_intercept, tolerance, with_sign=True
        )

    # test predict
    cu_preds = lr.predict(X_dask, delayed=delayed).compute()
    if isinstance(cu_preds, cp.ndarray):
        cu_preds = cp.asnumpy(cu_preds)
    if not isinstance(cu_preds, np.ndarray):
        cu_preds = cu_preds.to_numpy()
    accuracy_cuml = accuracy_score(y, cu_preds)

    sk_preds = sk_model.predict(X)
    accuracy_sk = accuracy_score(y, sk_preds)

    assert len(cu_preds) == len(sk_preds)
    assert (accuracy_cuml >= accuracy_sk) | (
        np.abs(accuracy_cuml - accuracy_sk) < 1e-3
    )

    return lr


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_noreg(fit_intercept, client):
    lr = test_lbfgs(
        nrows=1e5,
        ncols=20,
        n_parts=23,
        fit_intercept=fit_intercept,
        datatype=np.float32,
        delayed=True,
        client=client,
        penalty="none",
    )

    qnpams = lr.qnparams.params
    assert qnpams["penalty_l1"] == 0.0
    assert qnpams["penalty_l2"] == 0.0

    l1_strength, l2_strength = lr._get_qn_params()
    assert l1_strength == 0.0
    assert l2_strength == 0.0


def test_n_classes_small(client):
    def assert_small(X, y, n_classes):
        X_df, y_df = _prep_training_data(client, X, y, partitions_per_worker=1)
        from cuml.dask.linear_model import LogisticRegression as cumlLBFGS_dask

        lr = cumlLBFGS_dask()
        lr.fit(X_df, y_df)
        assert lr._num_classes == n_classes
        return lr

    X = np.array([(1, 2), (1, 3)], np.float32)
    y = np.array([1.0, 0.0], np.float32)
    lr = assert_small(X=X, y=y, n_classes=2)
    assert np.array_equal(
        lr.classes_.to_numpy(), np.array([0.0, 1.0], np.float32)
    )

    X = np.array([(1, 2), (1, 3), (1, 2.5)], np.float32)
    y = np.array([1.0, 0.0, 1.0], np.float32)
    lr = assert_small(X=X, y=y, n_classes=2)
    assert np.array_equal(
        lr.classes_.to_numpy(), np.array([0.0, 1.0], np.float32)
    )

    X = np.array([(1, 2), (1, 2.5), (1, 3)], np.float32)
    y = np.array([1.0, 1.0, 0.0], np.float32)
    lr = assert_small(X=X, y=y, n_classes=2)
    assert np.array_equal(
        lr.classes_.to_numpy(), np.array([0.0, 1.0], np.float32)
    )

    X = np.array([(1, 2), (1, 3), (1, 2.5)], np.float32)
    y = np.array([10.0, 50.0, 20.0], np.float32)
    lr = assert_small(X=X, y=y, n_classes=3)
    assert np.array_equal(
        lr.classes_.to_numpy(), np.array([10.0, 20.0, 50.0], np.float32)
    )


@pytest.mark.parametrize("n_parts", [2, 23])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("n_classes", [8])
def test_n_classes(n_parts, fit_intercept, n_classes, client):
    lr = test_lbfgs(
        nrows=1e5,
        ncols=20,
        n_parts=n_parts,
        fit_intercept=fit_intercept,
        datatype=np.float32,
        delayed=True,
        client=client,
        penalty="l2",
        n_classes=n_classes,
    )

    assert lr._num_classes == n_classes


@pytest.mark.mg
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("datatype", [np.float32])
@pytest.mark.parametrize("delayed", [True])
@pytest.mark.parametrize("n_classes", [2, 8])
@pytest.mark.parametrize("C", [1.0, 10.0])
def test_l1(fit_intercept, datatype, delayed, n_classes, C, client):
    lr = test_lbfgs(
        nrows=1e5,
        ncols=20,
        n_parts=2,
        fit_intercept=fit_intercept,
        datatype=datatype,
        delayed=delayed,
        client=client,
        penalty="l1",
        n_classes=n_classes,
        C=C,
    )

    l1_strength, l2_strength = lr._get_qn_params()
    assert l1_strength == 1.0 / lr.C
    assert l2_strength == 0.0


@pytest.mark.mg
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("datatype", [np.float32])
@pytest.mark.parametrize("delayed", [True])
@pytest.mark.parametrize("n_classes", [2, 8])
@pytest.mark.parametrize("l1_ratio", [0.2, 0.8])
def test_elasticnet(
    fit_intercept, datatype, delayed, n_classes, l1_ratio, client
):
    lr = test_lbfgs(
        nrows=1e5,
        ncols=20,
        n_parts=2,
        fit_intercept=fit_intercept,
        datatype=datatype,
        delayed=delayed,
        client=client,
        penalty="elasticnet",
        n_classes=n_classes,
        l1_ratio=l1_ratio,
    )

    l1_strength, l2_strength = lr._get_qn_params()

    strength = 1.0 / lr.C
    assert l1_strength == lr.l1_ratio * strength
    assert l2_strength == (1.0 - lr.l1_ratio) * strength


@pytest.mark.mg
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize(
    "regularization",
    [
        ("none", 1.0, None),
        ("l2", 2.0, None),
        ("l1", 2.0, None),
        ("elasticnet", 2.0, 0.2),
    ],
)
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("n_classes", [2, 8])
def test_sparse_from_dense(
    fit_intercept, regularization, datatype, n_classes, client
):
    penalty, C, l1_ratio = regularization

    run_test = partial(
        test_lbfgs,
        nrows=1e5,
        ncols=20,
        n_parts=2,
        fit_intercept=fit_intercept,
        datatype=datatype,
        delayed=True,
        client=client,
        penalty=penalty,
        n_classes=n_classes,
        C=C,
        l1_ratio=l1_ratio,
        convert_to_sparse=True,
    )

    if datatype == np.float32:
        run_test()
    else:
        with pytest.raises(
            RuntimeError,
            match="dtypes other than float32 are currently not supported",
        ):
            run_test()


@pytest.mark.parametrize("dtype", [np.float32])
def test_sparse_nlp20news(dtype, nlp_20news, client):

    X, y = nlp_20news
    n_parts = 2  # partitions_per_worker

    from scipy.sparse import csr_matrix
    from sklearn.model_selection import train_test_split

    X = X.astype(dtype)

    X = csr_matrix(X)
    y = y.get().astype(dtype)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    from cuml.dask.linear_model import LogisticRegression as MG

    X_train_da, y_train_da = _prep_training_data_sparse(
        client, X_train, y_train, partitions_per_worker=n_parts
    )
    X_test_da, _ = _prep_training_data_sparse(
        client, X_test, y_test, partitions_per_worker=n_parts
    )

    cumg = MG(verbose=6, C=20.0)
    cumg.fit(X_train_da, y_train_da)

    preds = cumg.predict(X_test_da).compute()
    cuml_score = accuracy_score(y_test, preds.tolist())

    from sklearn.linear_model import LogisticRegression as CPULR

    cpu = CPULR(C=20.0)
    cpu.fit(X_train, y_train)
    cpu_preds = cpu.predict(X_test)
    cpu_score = accuracy_score(y_test, cpu_preds.tolist())
    assert cuml_score >= cpu_score or np.abs(cuml_score - cpu_score) < 1e-3


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_exception_one_label(fit_intercept, client):
    n_parts = 2
    datatype = "float32"

    X = np.array([(1, 2), (1, 3), (2, 1), (3, 1)], datatype)
    y = np.array([1.0, 1.0, 1.0, 1.0], datatype)
    X_df, y_df = _prep_training_data(client, X, y, n_parts)

    err_msg = "This solver needs samples of at least 2 classes in the data, but the data contains only one class: 1.0"

    from cuml.dask.linear_model import LogisticRegression as cumlLBFGS_dask

    mg = cumlLBFGS_dask(fit_intercept=fit_intercept, verbose=6)
    with pytest.raises(RuntimeError, match=err_msg):
        mg.fit(X_df, y_df)

    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(fit_intercept=fit_intercept)
    with pytest.raises(ValueError, match=err_msg):
        lr.fit(X, y)
