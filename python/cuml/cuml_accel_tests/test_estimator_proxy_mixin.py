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

import pickle
import subprocess
import sys
from textwrap import dedent

import pytest
from sklearn import clone

from cuml.accel import is_proxy

hdbscan = pytest.importorskip("hdbscan")


def test_is_proxy():
    class Foo:
        pass

    assert is_proxy(hdbscan.HDBSCAN)
    assert is_proxy(hdbscan.HDBSCAN())
    assert not is_proxy(Foo)
    assert not is_proxy(Foo())


def test_meta_attributes():
    # Check that the proxy estimator pretends to look like the
    # class it is proxying

    # A random estimator, shouldn't matter which one as all are proxied
    # the same way.
    # We need an instance to get access to the `_cpu_model_class`
    # but we want to compare to the HDBSCAN class
    est = hdbscan.HDBSCAN()
    for attr in (
        "__module__",
        "__name__",
        "__qualname__",
        "__doc__",
        "__annotate__",
        "__type_params__",
    ):
        # if the original class has this attribute, the proxy should
        # have it as well and the values should match
        try:
            original_value = getattr(est._cpu_model_class, attr)
        except AttributeError:
            pass
        else:
            proxy_value = getattr(hdbscan.HDBSCAN, attr)

            assert original_value == proxy_value


def test_clone():
    # Test that cloning a proxy estimator preserves parameters, even those we
    # translate for the cuml class
    est = hdbscan.HDBSCAN(alpha=2.0, algorithm="auto", memory=None)
    est_clone = clone(est)

    assert est.get_params() == est_clone.get_params()


def test_pickle():
    est = hdbscan.HDBSCAN(alpha=2.0, algorithm="auto", memory=None)
    buf = pickle.dumps(est)
    est2 = pickle.loads(buf)
    assert type(est2) is hdbscan.HDBSCAN
    assert est2.get_params() == est.get_params()
    assert repr(est) == repr(est2)


def test_pickle_loads_doesnt_install_accelerator():
    est = hdbscan.HDBSCAN(alpha=2.0)
    buf = pickle.dumps(est)
    script = dedent(
        f"""
        import pickle

        model = pickle.loads({buf!r})

        params = model.get_params()
        assert params["alpha"] == 2.0
        assert type(model).__name__ == "HDBSCAN"

        from cuml.accel import enabled
        from cuml.accel.estimator_proxy_mixin import ProxyMixin
        from hdbscan import HDBSCAN

        # Unpickling hasn't installed the accelerator or patched sklearn
        assert not issubclass(HDBSCAN, ProxyMixin)
        assert not enabled()
        """
    )
    res = subprocess.run(
        [sys.executable, "-c", script],
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        text=True,
    )
    # Pull out attributes before assert for nicer error reporting on failure
    returncode = res.returncode
    stdout = res.stdout
    assert returncode == 0, stdout


def test_params():
    # Test that parameters match between constructor and get_params()
    # Mix of default and non-default values
    est = hdbscan.HDBSCAN(min_cluster_size=5, algorithm="brute", alpha=2.0)

    params = est.get_params()
    assert params["min_cluster_size"] == 5
    assert params["algorithm"] == "brute"
    assert params["alpha"] == 2.0
    # A parameter we never touched, should be the default
    assert params["max_cluster_size"] == 0
