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

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import os
import sys
from textwrap import dedent

import pytest

from cuml.accel.accelerator import Accelerator, AccelModule


class MockLoader(importlib.abc.Loader):
    def __init__(self, source):
        self.source = source

    def create_module(self, spec):
        pass

    def exec_module(self, module):
        exec(self.source, module.__dict__)


class MockPackage(importlib.abc.MetaPathFinder):
    def __init__(self, modules: dict[str, str]) -> None:
        self.name = f"mockmodule_{os.urandom(4).hex()}"
        self.modules = {
            f"{self.name}.{key}" if key else self.name: dedent(source)
            for key, source in modules.items()
        }

    def find_spec(
        self, fullname: str, path, target=None
    ) -> importlib.machinery.ModuleSpec | None:
        if (source := self.modules.get(fullname)) is not None:
            spec = importlib.machinery.ModuleSpec(
                name=fullname, loader=MockLoader(source)
            )
            spec.submodule_search_locations = []
            return spec

    def __enter__(self):
        sys.meta_path.insert(0, self)
        return self.name

    def __exit__(self, *args):
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        for name in self.modules:
            sys.modules.pop(name, None)


@pytest.fixture
def clean_meta_path():
    """A fixture for resetting the `sys.meta_path` after a test"""
    meta_path = sys.meta_path.copy()
    yield
    sys.meta_path = meta_path


@pytest.fixture
def mockmod(clean_meta_path):
    modules = {
        "": """
        from .utils import fizz, buzz

        def fizzbuzz():
            return fizz() + buzz()

        def process(n):
            if n % 3:
                if n % 5:
                    return fizzbuzz()
                return fizz()
            return buzz()
        """,
        "utils": """
        def fizz():
            return "fizz"

        def buzz():
            return "buzz"
        """,
    }
    with MockPackage(modules) as name:
        yield name


def test_accelerator(mockmod):
    def fizz():
        return "FIZZ"

    accel = Accelerator()
    assert not accel.installed
    accel.register(f"{mockmod}.utils", {"fizz": fizz})
    accel.install()
    assert accel.installed

    # No harm in calling install more than once
    accel.install()

    mod = importlib.import_module(mockmod)
    assert mod.fizz is fizz
    assert mod.utils.fizz is fizz
    assert sys.modules[f"{mockmod}.utils"] is mod.utils

    assert mod.fizzbuzz() == "FIZZbuzz"


def test_accelerator_exclude_sequence(mockmod):
    def fizz():
        return "FIZZ"

    accel = Accelerator(exclude=[mockmod])
    accel.register(f"{mockmod}.utils", {"fizz": fizz})
    accel.install()

    mod = importlib.import_module(mockmod)
    assert mod.utils.fizz is fizz
    assert mod.fizz is not fizz
    assert mod.fizzbuzz() == "fizzbuzz"


def test_accelerator_exclude_callable(mockmod):
    exclude_called = False

    def exclude(module):
        nonlocal exclude_called
        exclude_called = True
        return module.split(".", 1)[0] == mockmod

    def fizz():
        return "FIZZ"

    accel = Accelerator(exclude=exclude)
    accel.register(f"{mockmod}.utils", {"fizz": fizz})
    accel.install()

    mod = importlib.import_module(mockmod)
    assert exclude_called
    assert mod.utils.fizz is fizz
    assert mod.fizz is not fizz
    assert mod.fizzbuzz() == "fizzbuzz"


def test_accelerator_external_exclude(mockmod):
    def fizz():
        return "FIZZ"

    accel = Accelerator(exclude=[mockmod, __name__])
    accel.register(f"{mockmod}.utils", {"fizz": fizz})
    accel.install()

    mod = importlib.import_module(mockmod)
    assert mod.utils.fizz is not fizz
    assert mod.fizz is not fizz
    assert mod.fizzbuzz() == "fizzbuzz"


def test_accelerator_import_in_patch(mockmod):
    """Check that imports of the original module work fine within a patch"""

    def patch(module):
        # Same as `from {mockmod}.utils import fizz`
        fizz = importlib.import_module(f"{mockmod}.utils").fizz
        assert fizz is module.fizz

        return {"fizz": lambda: fizz().upper()}

    accel = Accelerator()
    accel.register(f"{mockmod}.utils", patch)
    accel.install()

    mod = importlib.import_module(mockmod)
    assert mod.utils.fizz() == "FIZZ"
    assert mod.fizz() == "FIZZ"
    assert mod.fizzbuzz() == "FIZZbuzz"


def test_accelerator_install_after_import(mockmod):
    def fizz():
        return "FIZZ"

    accel = Accelerator()
    accel.register(f"{mockmod}.utils", {"fizz": fizz})

    mod = importlib.import_module(mockmod)
    assert mod.utils.fizz is not fizz

    accel.install()

    assert isinstance(mod.utils, AccelModule)
    assert sys.modules[f"{mockmod}.utils"] is mod.utils
    assert mod.utils.fizz is fizz


def test_accel_module(mockmod):
    orig_mod = importlib.import_module(mockmod)

    accel = Accelerator()
    accel.register(mockmod, {"fizz": lambda: orig_mod.fizz().upper()})
    accel.install()

    mod = importlib.import_module(mockmod)

    assert isinstance(mod, AccelModule)

    # dir uses original module
    assert dir(mod) == dir(orig_mod)

    # repr is modified
    assert repr(mod).startswith("<accelerated module")

    # getattr prefers overrides for accelerated versions
    assert mod.buzz is orig_mod.buzz
    assert mod.fizz() == "FIZZ"
    assert orig_mod.fizz() == "fizz"

    # getattr error
    with pytest.raises(AttributeError, match="oops"):
        mod.oops

    # setattr sets on the original module
    mod.foo = 1
    assert mod.foo == 1
    assert orig_mod.foo == 1

    # Meta attributes are set
    for attr in ["__package__", "__loader__", "__spec__"]:
        assert getattr(mod, attr) == getattr(orig_mod, attr)


def test_import_error_non_existant_file_in_accelerated_module(mockmod):
    accel = Accelerator()
    accel.install()

    with pytest.raises(ImportError, match=f"{mockmod}.oops"):
        importlib.import_module(f"{mockmod}.oops")
