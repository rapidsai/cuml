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

import numpy as np
from sklearn import clone
from sklearn.neighbors import NearestNeighbors

from cuml.accel import is_proxy


def test_is_proxy():
    class Foo:
        pass

    assert is_proxy(NearestNeighbors)
    assert is_proxy(NearestNeighbors())
    assert not is_proxy(Foo)
    assert not is_proxy(Foo())


def test_meta_attributes():
    # Check that the proxy estimator pretends to look like the
    # class it is proxying

    # A random estimator, shouldn't matter which one as all are proxied
    # the same way.
    # We need an instance to get access to the `_cpu_model_class`
    # but we want to compare to the NearestNeighbors class
    est = NearestNeighbors()
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
            proxy_value = getattr(NearestNeighbors, attr)

            assert original_value == proxy_value


def test_clone():
    # Test that cloning a proxy estimator preserves parameters, even those we
    # translate for the cuml class
    est = NearestNeighbors(n_neighbors=42, algorithm="brute")
    est_clone = clone(est)

    assert est.get_params() == est_clone.get_params()


def test_pickle():
    est = NearestNeighbors(n_neighbors=42, algorithm="brute")
    buf = pickle.dumps(est)
    est2 = pickle.loads(buf)
    assert type(est2) is NearestNeighbors
    assert est2.get_params() == est.get_params()
    assert repr(est) == repr(est2)


def test_pickle_loads_doesnt_install_accelerator():
    est = NearestNeighbors(n_neighbors=42)
    buf = pickle.dumps(est)
    script = dedent(
        f"""
        import pickle

        model = pickle.loads({buf!r})

        assert model.n_neighbors == 42
        assert type(model).__name__ == "NearestNeighbors"

        from cuml.accel import enabled
        from cuml.accel.estimator_proxy_mixin import ProxyMixin
        from sklearn.neighbors import NearestNeighbors

        # Unpickling hasn't installed the accelerator or patched sklearn
        assert not issubclass(NearestNeighbors, ProxyMixin)
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
    est = NearestNeighbors(n_neighbors=5, algorithm="brute", leaf_size=15)

    params = est.get_params()
    assert params["n_neighbors"] == 5
    assert params["algorithm"] == "brute"
    assert params["leaf_size"] == 15
    # A parameter we never touched, should be the default
    assert params["radius"] == 1.0


def test_defaults_args_only_methods():
    # Check that estimator methods that take no arguments work
    # These are slightly weird because basically everything else takes
    # a X as input.
    X = np.random.rand(1000, 3)
    y = X[:, 0] + np.sin(6 * np.pi * X[:, 1]) + 0.1 * np.random.randn(1000)

    nn = NearestNeighbors(metric="chebyshev", n_neighbors=3)
    nn.fit(X[:, 0].reshape((-1, 1)), y)
    nn.kneighbors()
