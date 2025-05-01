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

from importlib import import_module
from typing import Any

from cuml.internals.mem_type import MemoryType
from cuml.internals.memory_utils import using_output_type

__all__ = (
    "UnsupportedOnGPU",
    "UnsupportedOnCPU",
    "InteropMixin",
    "to_gpu",
)


def to_gpu(x, order="K"):
    """Coerce `x` to the equivalent GPU type."""
    # Ideally we'd get rid of this helper eventually, using it for now to reduce typing
    from cuml.internals.input_utils import input_to_cuml_array

    return input_to_cuml_array(
        x, order=order, convert_to_mem_type=MemoryType.device
    )[0]


class UnsupportedOnGPU(ValueError):
    """An exception raised when a conversion of a CPU to a GPU estimator isn't supported"""


class UnsupportedOnCPU(ValueError):
    """An exception raised when a conversion of a GPU to a CPU estimator isn't supported"""


class classproperty:
    """Like ``property``, but on a class rather than an instance."""

    def __init__(self, fget):
        self.fget = fget

    def __get__(self, obj, owner):
        return self.fget(owner)


class InteropMixin:
    """A mixin for enabling conversion of a cuml estimator to/from its
    CPU-based counterpart.

    Subclasses should define:

    - `_cpu_class_path`
    - `_params_from_cpu`
    - `_params_to_cpu`
    - `_attrs_from_cpu`
    - `_attrs_to_cpu`

    In return they get ``as_sklearn``/``from_sklearn`` methods, and can also
    be used as the basis for a proxy estimator in ``cuml.accel``.
    """

    # The import path to use to import the CPU model class
    _cpu_class_path: str

    @classproperty
    def _cpu_class(cls):
        """The CPU class that corresponds to this GPU model"""
        module, _, name = cls._cpu_class_path.rpartition(".")
        return getattr(import_module(module), name)

    @classmethod
    def _params_from_cpu(cls, model) -> dict[str, Any]:
        """Get parameters to use to instantiate a GPU model from a CPU model.

        Parameters
        ----------
        model
            The CPU model, of the same type as ``self._cpu_class``.

        Returns
        -------
        dict
            A mapping of keyword arguments that may be used to instantiate ``cls``
            to create an equivalent GPU model.

        Raises
        ------
        UnsupportedOnGPU
            If one or more hyperparameters are unsupported by the GPU model.
        """
        raise NotImplementedError

    def _params_to_cpu(self) -> dict[str, Any]:
        """Get parameters to use to instantiate a CPU model from a GPU model.

        Returns
        -------
        dict
            A mapping of keyword arguments that may be used to instantiate
            ``self._cpu_class`` to create an equivalent CPU model.

        Raises
        ------
        UnsupportedOnCPU
            If one or more hyperparameters are unsupported by the CPU model.
        """
        raise NotImplementedError

    def _attrs_from_cpu(self, model) -> dict[str, Any]:
        """Get attributes to set on ``self`` from a fit CPU model.

        The callers of this method check that ``model`` is of the correct
        type and has been fit, no need to check that in an implementation.

        The base class handles common metadata attributes, implementations should
        be sure to call ``super()._attrs_from_cpu()`` to include those in the output.

        Parameters
        ----------
        model
            The CPU model, of the same type as ``self._cpu_class``.

        Returns
        -------
        dict
            A mapping of attributes to set on ``self`` to create a fit GPU model
            from a fit CPU model.

        Raises
        ------
        UnsupportedOnGPU
            If one or more attributes are unsupported by the GPU model.
        """
        out = {}
        for name in ["n_features_in_", "feature_names_in_"]:
            try:
                out[name] = getattr(model, name)
            except AttributeError:
                pass
        return out

    def _attrs_to_cpu(self, model) -> dict[str, Any]:
        """Get attributes to set on CPU model ``model`` from ``self``.

        The callers of this method check that ``model`` is of the correct
        type and that ``self`` has been fit, no need to check that in an implementation.

        The base class handles common metadata attributes, implementations should
        be sure to call ``super()._attrs_to_cpu()`` to include those in the output.

        Parameters
        ----------
        model
            The CPU model, of the same type as ``self._cpu_class``.

        Returns
        -------
        dict
            A mapping of attributes to set on ``model`` to create a fit CPU model
            from a fit GPU model.

        Raises
        ------
        UnsupportedOnCPU
            If one or more attributes are unsupported by the CPU model.
        """
        out = {}
        for name in ["n_features_in_", "feature_names_in_"]:
            try:
                out[name] = getattr(self, name)
            except AttributeError:
                pass
        return out

    def _sync_attrs_to_cpu(self, model) -> None:
        """Sync any fitted attributes from ``self`` to ``model``.

        Parameters
        ----------
        model
            An instance of ``self._cpu_class``.
        """
        if getattr(self, "n_features_in_", None) is None:
            # GPU model not fitted, nothing to do
            return

        # XXX: we use this for now to ensure _attrs_to_cpu can rely on
        # a consistent type for all fitted attributes, rather than
        # having things potentially vary based on `self.output_type`.
        with using_output_type("cuml"):
            attrs = self._attrs_to_cpu(model)

        for name, value in attrs.items():
            setattr(model, name, value)

    def _sync_attrs_from_cpu(self, model) -> None:
        """Sync any fitted attributes from ``model`` to ``self``.

        Parameters
        ----------
        model
            An instance of ``self._cpu_class``.
        """
        if getattr(model, "n_features_in_", None) is None:
            # CPU model not fitted, nothing to do
            return

        attrs = self._attrs_from_cpu(model)
        for name, value in attrs.items():
            setattr(self, name, value)

    def as_sklearn(self):
        """
        Convert this estimator into an equivalent scikit-learn (or scikit-learn
        extension) estimator.

        Returns
        -------
        sklearn.base.BaseEstimator
            A scikit-learn compatible estimator instance that mirrors the trained
            state of the current estimator.
        """
        params = self._params_to_cpu()
        model = self._cpu_class(**params)
        self._sync_attrs_to_cpu(model)
        return model

    @classmethod
    def from_sklearn(cls, model):
        """
        Create a cuml estimator from a scikit-learn estimator.

        Parameters
        ----------
        model : sklearn.base.BaseEstimator
            A compatible scikit-learn (or scikit-learn extension) estimator.

        Returns
        -------
        cls
            A new instance of this cuml estimator class that mirrors the
            state of the input estimator.

        Notes
        -----
        `output_type` of the estimator is set to "numpy" by default, as these
        cannot be inferred from training arguments. If something different is
        required, then please use cuml's output_type configuration utilities.
        """
        if not isinstance(model, cls._cpu_class):
            raise TypeError(
                f"Expected instance of {cls._cpu_class_path!r}, got "
                f"{type(model).__name__!r}"
            )
        params = cls._params_from_cpu(model)
        out = cls(**params)

        out._sync_attrs_from_cpu(model)

        # Set output type to numpy, since we can't infer it from the inputs.
        out.output_type = "numpy"
        out.output_mem_type = MemoryType.host

        return out
