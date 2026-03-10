#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types
from collections.abc import Iterable
from threading import RLock
from typing import Any, Callable, Union

__all__ = ("Accelerator",)


LazyNamespace = Union[
    Callable[[types.ModuleType], dict[str, Any]],
    dict[str, Any],
    str,
]


class AccelModule(types.ModuleType):
    """An accelerated module.

    Wraps an underlying module, replacing access to certain attributes
    with accelerated versions.
    """

    # Internal state. All have prefix `_accel_` to try and avoid accidental collisions
    # with names on the wrapped module.
    _accel_module: types.ModuleType
    _accel_exclude: Callable[[str], bool]
    _accel_overrides: dict[str, Any]

    def __repr__(self) -> str:
        return super().__repr__().replace("<module", "<accelerated module", 1)

    def __getattr__(self, name: str) -> Any:
        if (accelerated := self._accel_overrides.get(name)) is not None:
            # This has an override and could be accelerated. We first need
            # to check that the accessing module isn't excluded.
            #
            # To do this we walk up the stack, skipping frames in importlib
            frame = sys._getframe()
            while True:
                assert frame.f_back is not None
                # Get the module name of the caller (if available)
                modname = frame.f_back.f_globals.get("__name__")

                if modname is None:
                    return getattr(self._accel_module, name)

                if modname.split(".", 1)[0] == "importlib":
                    # Caller is in importlib, continue up the stack
                    frame = frame.f_back
                else:
                    break

            if not self._accel_exclude(modname):
                # Not excluded, use the accelerated version
                return accelerated

        return getattr(self._accel_module, name)

    def __setattr__(self, name: str, val: Any) -> None:
        if name in ("_accel_module", "_accel_exclude", "_accel_overrides"):
            super().__setattr__(name, val)
        else:
            setattr(self._accel_module, name, val)

    def __dir__(self) -> list[str]:
        return dir(self._accel_module)


def wrap_module(
    module: types.ModuleType, exclude: Callable[[str], bool]
) -> AccelModule:
    """Create a new AccelModule instance.

    Parameters
    ----------
    module : ModuleType
        The original, unaccelerated module.
    exclude : callable
        A callable for determining if a requesting module should be excluded from
        accessing an accelerated attribute.

    Returns
    -------
    AccelModule
        The accelerated module.
    """
    out = AccelModule(module.__name__, doc=getattr(module, "__doc__", None))
    out._accel_module = module
    out._accel_exclude = exclude
    out._accel_overrides = {}

    # ModuleType instances come with a few attributes pre-defined. Delete these
    # so they're forwarded to the wrapped version instead. These aren't critical
    # for most use cases, but some inspection tools may make use of them so we
    # should forward them appropriately.
    for field in ["__package__", "__loader__", "__spec__"]:
        out.__dict__.pop(field, None)
    return out


class ModuleTransform:
    def __init__(
        self,
        override: LazyNamespace | None = None,
        patch: LazyNamespace | None = None,
    ):
        self.override = override
        self.patch = patch

    @staticmethod
    def _load_namespace(
        module: AccelModule, namespace: LazyNamespace
    ) -> dict[str, Any]:
        assert isinstance(module, AccelModule)
        if callable(namespace):
            return namespace(module)
        elif isinstance(namespace, str):
            new_module = importlib.import_module(namespace)
            if (names := getattr(new_module, "__all__", None)) is None:
                raise ValueError(
                    f"Module `{namespace}` must define `__all__` to specify the names of "
                    "the attributes to override/patch in the accelerated module"
                )
            return {name: getattr(new_module, name) for name in names}
        else:
            return namespace

    def apply(self, module: AccelModule) -> None:
        """Load and apply patches/overrides to an accelerated module."""
        if self.patch is not None:
            ns = self._load_namespace(module, self.patch)
            for k, v in ns.items():
                setattr(module._accel_module, k, v)

        if self.override is not None:
            ns = self._load_namespace(module, self.override)
            module._accel_overrides.update(ns)


class AccelLoader(importlib.abc.Loader):
    """A loader for loading an accelerated module."""

    def __init__(
        self,
        spec: importlib.machinery.ModuleSpec,
        transform: ModuleTransform,
        exclude: Callable[[str], bool] | None = None,
    ) -> None:
        self._spec = spec
        self._transform = transform
        self._exclude = exclude

    def create_module(
        self, spec: importlib.machinery.ModuleSpec
    ) -> AccelModule:
        assert spec.name == self._spec.name
        module = importlib.util.module_from_spec(self._spec)
        return wrap_module(module, self._exclude)

    def exec_module(self, module: types.ModuleType) -> None:
        assert isinstance(module, AccelModule)
        assert self._spec.loader is not None
        self._spec.loader.exec_module(module._accel_module)
        self._transform.apply(module)


class AccelFinder(importlib.abc.MetaPathFinder):
    """A finder for loading accelerated modules.

    Intercepts registered imports and replaces them with accelerated versions.

    Parameters
    ----------
    accelerator : Accelerator
        The corresponding accelerator.
    """

    def __init__(self, accelerator: Accelerator) -> None:
        self.accelerator = accelerator
        self._importing: set[str] = set()

    def find_spec(
        self, fullname: str, path, target=None
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname in self._importing:
            return None

        if (transform := self.accelerator.transforms.get(fullname)) is None:
            return None

        try:
            self._importing.add(fullname)
            real_spec = importlib.util.find_spec(fullname)
        finally:
            self._importing.discard(fullname)

        if real_spec is None:
            return None

        spec = importlib.machinery.ModuleSpec(
            name=fullname,
            loader=AccelLoader(real_spec, transform, self.accelerator.exclude),
            origin=real_spec.origin,
            loader_state=real_spec.loader_state,
            is_package=real_spec.submodule_search_locations is not None,
        )
        spec.submodule_search_locations = real_spec.submodule_search_locations
        return spec


class Accelerator:
    """A module accelerator.

    Parameters
    ----------
    exclude : sequence or callable, optional
        A sequence of module names that shouldn't have access to the accelerated versions.
        Imports and attribute access from within these modules will always give the
        unaccelerated versions.
    """

    exclude: Callable[[str], bool]
    transforms: dict[str, ModuleTransform]
    _lock: RLock
    _installed: bool

    def __init__(
        self, exclude: Iterable[str] | Callable[[str], bool] | None = None
    ):
        if callable(exclude):
            self.exclude = exclude
        else:
            self.exclude = frozenset(exclude or ()).__contains__

        self.transforms = {}
        self._lock = RLock()
        self._installed = False

    @property
    def installed(self) -> bool:
        """Whether the accelerator is installed."""
        return self._installed

    @property
    def enabled(self) -> bool:
        """Whether the accelerator is enabled."""
        # For now the accelerator is always enabled if it is installed. Later on we
        # might want to add support for disabling in a thread-local context. Stubbing
        # out the name here for now so it can be referenced cleanly by the rest of cuml,
        # minimizing the changes needed if/when we want to add that feature.
        return self.installed

    def register(
        self,
        name: str,
        override: LazyNamespace | None = None,
        patch: LazyNamespace | None = None,
    ):
        """Register a new override or patch for a module.

        Overrides are only visible to non-excluded modules, and don't mutate
        the original module. Patches apply everywhere and do mutate the original
        module.

        For example, if `sklearn` is in the `exclude` list for the accelerator,
        an override for `sklearn.linear_models.LinearRegression` won't be
        visible to consumers within `sklearn` itself (they'll still get the
        original `LinearRegression`). In contrast, a `patch` will be visible
        everywhere, and including for consumers within `sklearn` itself.

        Parameters
        ----------
        name : str
            The name of the unaccelerated module to patch.
        override, patch : mapping, callable, or str
            May be one of the following:

            - A mapping of attributes to override/patch in the accelerated
              module.
            - A callable taking the original module and returning a mapping of
              attributes to override/patch in the accelerated module.
            - A string name of a module to import containing overrides or
              patches. All names specified in `__all__` in the module will be
              used to override/patch attributes in the accelerated module.

        Examples
        --------
        >>> accel = Accelerator()

        Register a mapping of attribute overrides to use for module ``fizzbuzz``.

        >>> accel.register("fizzbuzz", {"fizz": lambda: "buzz"})

        Register a module path defining overrides to use for module ``foobar``.

        >>> accel.register("foobar", "fast.foobar")
        """
        assert not self._installed
        assert name not in self.transforms
        self.transforms[name] = ModuleTransform(override=override, patch=patch)

    def _maybe_transform(self, name: str) -> None:
        if (module := sys.modules.get(name)) is None:
            # Not imported yet, import system will load patch lazily later
            return

        if isinstance(module, AccelModule):
            # Already accelerated, nothing to do
            return

        # The unaccelerated module is already imported.
        # 1. Wrap the unaccelerated module
        accelerated = wrap_module(module, self.exclude)

        # 2. Replace it in sys.modules
        sys.modules[name] = accelerated

        # 3. Replace the reference on the parent module (if any)
        parent_name, _, child_name = name.rpartition(".")
        parent = sys.modules.get(parent_name) if parent_name else None
        if getattr(parent, child_name, None) is module:
            setattr(parent, child_name, accelerated)

        # 4. Apply module transforms
        self.transforms[name].apply(accelerated)

    def install(self) -> None:
        """Install the accelerator.

        This installs the import hooks to intercept future imports of
        accelerated modules. It also wraps/overrides previous imports of these
        modules in a best-effort approach.
        """
        with self._lock:
            if self._installed:
                return

            # Install the import hook. This handles patching any modules imported later.
            sys.meta_path.insert(0, AccelFinder(self))

            # Wrap any modules that are already imported.
            for name in self.transforms:
                self._maybe_transform(name)

            self._installed = True
