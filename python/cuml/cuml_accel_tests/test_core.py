# Copyright (c) 2025, NVIDIA CORPORATION.
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
import importlib
import multiprocessing
from inspect import Parameter, signature

import pytest

import cuml.accel
from cuml.accel.estimator_proxy import ProxyBase


def proxy_base_subclasses():
    """All defined subclasses of ProxyBase"""
    from cuml.accel.core import ACCELERATED_MODULES

    for mod in ACCELERATED_MODULES:
        importlib.import_module(mod)

    return sorted(ProxyBase.__subclasses__(), key=lambda cls: cls.__name__)


def test_multiple_import_styles_work():
    from sklearn import linear_model
    from sklearn.linear_model import LogisticRegression

    assert linear_model.LogisticRegression is LogisticRegression
    assert cuml.accel.is_proxy(LogisticRegression)


def test_enabled():
    assert cuml.accel.enabled()


@pytest.mark.parametrize("method", ["spawn", "fork"])
def test_enabled_in_subprocesses(method):
    ctx = multiprocessing.get_context(method)
    with ctx.Pool(processes=1) as pool:
        enabled = pool.apply(cuml.accel.enabled)
    assert enabled


def test_enabled_in_loky_executor():
    try:
        # Recent versions of joblib vendor loky
        from joblib.externals import loky
    except ImportError:
        loky = pytest.importorskip("loky")

    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    X, y = make_classification()

    with loky.get_reusable_executor(max_workers=1) as executor:
        enabled = executor.submit(cuml.accel.enabled).result()
        assert enabled
        remote = executor.submit(LogisticRegression().fit, X, y).result()
        assert cuml.accel.is_proxy(remote)


def iter_proxy_class_methods():
    """Generate test cases of (cls, method_name) for all ProxyBase proxied methods"""
    classes = proxy_base_subclasses()

    for cls in classes:
        for name in dir(cls._cpu_class):
            if not name.startswith("_") and callable(
                getattr(cls._cpu_class, name)
            ):
                # XXX: xfail umap.UMAP.get_feature_names_out for now
                if (
                    cls._cpu_class.__name__ == "UMAP"
                    and name == "get_feature_names_out"
                ):
                    yield pytest.param(
                        cls,
                        name,
                        marks=[
                            pytest.mark.xfail(
                                reason="umap-learn <= 0.5.7 doesn't implement `get_feature_names_out` properly",
                                strict=True,
                            )
                        ],
                    )
                else:
                    yield cls, name


@pytest.mark.parametrize("cls, name", iter_proxy_class_methods())
def test_proxied_methods_signature_compatibility(cls, name):
    """Check that the GPU proxy signatures are compatible with the CPU signatures.

    Any methods that fail this check should have an override `_gpu_{name}` method
    defined on the proxy class to fix the signature mismatch between the CPU method
    signature and cuml's version."""
    cpu_method = getattr(cls._cpu_class, name)

    # Get the GPU method definition the proxy will use
    gpu_method = getattr(cls, f"_gpu_{name}", None)
    if gpu_method is None:
        gpu_method = getattr(cls._gpu_class, name, None)
        if gpu_method is None:
            # CPU-only method, nothing to check
            return

    cpu_params = list(signature(cpu_method).parameters.values())
    gpu_params = list(signature(gpu_method).parameters.values())

    # Check that the GPU method signature is a superset of the CPU one
    assert len(gpu_params) >= len(cpu_params)
    for cpu_param, gpu_param in zip(cpu_params, gpu_params):
        assert cpu_param.name == gpu_param.name
        assert cpu_param.kind == gpu_param.kind

    # Check that any additional arguments are optional
    for gpu_param in gpu_params[len(cpu_params) :]:
        if gpu_param.kind == Parameter.POSITIONAL_OR_KEYWORD:
            assert gpu_param.default is not Parameter.empty
        else:
            assert gpu_param.kind in {
                Parameter.KEYWORD_ONLY,
                Parameter.VAR_KEYWORD,
            }
