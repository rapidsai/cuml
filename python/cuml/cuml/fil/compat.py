#
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""
Compatibility shim for cuml.fil -> nvForest

TODO(26.10): This module will be removed in 26.10.
"""

import warnings

import cupy as cp
import nvforest
import treelite

from cuml.internals.base import Base, get_handle
from cuml.internals.device_type import DeviceType
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.mixins import CMajorInputTagMixin
from cuml.internals.outputs import mlfunc
from cuml.internals.validation import check_array


def _is_nvforest_model_on_device(nvforest_model) -> bool:
    return isinstance(
        nvforest_model,
        (
            nvforest.GPUForestInferenceClassifier,
            nvforest.GPUForestInferenceRegressor,
        ),
    )


class set_fil_device_type:
    """Set the device type used by FIL.

    May optionally be used as a context-manager to set the device type only
    within a context.

    This is deprecated and will be removed in 26.10.

    Parameters
    ----------
    device_type : {'cpu', 'gpu'}
        The device type to use.

    Examples
    --------
    >>> from cuml.fil import set_fil_device_type  # doctest: +SKIP

    Set the device type globally to use CPU.

    >>> set_fil_device_type("cpu")  # doctest: +SKIP

    Set the device type globally to use GPU.

    >>> set_fil_device_type("gpu")  # doctest: +SKIP

    Set the device type to use CPU within a context.

    >>> with set_fil_device_type("cpu"):  # doctest: +SKIP
    ...     ...
    """

    def __init__(self, device_type):
        warnings.warn(
            "cuml.fil.set_fil_device_type is deprecated and will be removed in 26.10. ",
            FutureWarning,
        )
        device_type = DeviceType.from_str(device_type)
        self._previous = GlobalSettings().fil_device_type
        GlobalSettings().fil_device_type = device_type

    def __enter__(self):
        return self

    def __exit__(self, *_):
        GlobalSettings().fil_device_type = self._previous


def get_fil_device_type() -> DeviceType:
    """Get the device type used by FIL."""
    return GlobalSettings().fil_device_type


class ForestInference(CMajorInputTagMixin, Base):
    def __init__(
        self,
        *,
        treelite_model=None,
        output_type=None,
        verbose=False,
        is_classifier=False,
        layout="depth_first",
        default_chunk_size=None,
        align_bytes=None,
        precision="single",
        device_id=None,
        ensure_all_finite=False,
        _suppress_deprecation_warning=False,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        if not _suppress_deprecation_warning:
            warnings.warn(
                "cuml.fil.ForestInference is deprecated and will be removed in 26.10. "
                "Use nvforest.load_model() or nvforest.load_from_sklearn() instead.",
                FutureWarning,
                stacklevel=2,
            )
        if treelite_model is None:
            self.model = None
        elif isinstance(treelite_model, (treelite.Model, bytes)):
            if isinstance(treelite_model, bytes):
                treelite_model = treelite.Model.deserialize_bytes(
                    treelite_model
                )
            self.model = nvforest.load_from_treelite_model(
                tl_model=treelite_model,
                device="gpu"
                if get_fil_device_type() == DeviceType.device
                else "cpu",
                layout=layout,
                default_chunk_size=default_chunk_size,
                align_bytes=align_bytes,
                precision=precision,
                device_id=device_id,
                handle=get_handle(),
            )
        else:
            raise ValueError(
                f"Unrecognized type for treelite_model: {type(treelite_model)}"
            )
        self.ensure_all_finite = ensure_all_finite
        self._suppress_deprecation_warning = _suppress_deprecation_warning

    @property
    def align_bytes(self):
        return self.model.align_bytes if self.model else None

    @align_bytes.setter
    def align_bytes(self, value):
        raise NotImplementedError(
            "Setter for align_bytes is no longer supported"
        )

    @property
    def precision(self):
        return self.model.precision if self.model else None

    @precision.setter
    def precision(self, value):
        raise NotImplementedError(
            "Setter for precision is no longer supported"
        )

    @property
    def is_classifier(self):
        return self.model.is_classifier if self.model else None

    @is_classifier.setter
    def is_classifier(self, value):
        raise NotImplementedError(
            "Setter for is_classifier is no longer supported"
        )

    @property
    def device_id(self):
        return self.model.device_id if self.model else None

    @device_id.setter
    def device_id(self, value):
        raise NotImplementedError(
            "Setter for device_id is no longer supported"
        )

    @property
    def treelite_model(self):
        warnings.warn(
            "Attribute treelite_model is no longer supported",
            FutureWarning,
            stacklevel=2,
        )
        return None

    @treelite_model.setter
    def treelite_model(self, value):
        raise NotImplementedError(
            "Setter for treelite_model is no longer supported"
        )

    @property
    def layout(self):
        return self.model.layout if self.model else None

    @layout.setter
    def layout(self, value):
        raise NotImplementedError("Setter for layout is no longer supported")

    def num_outputs(self):
        return self.model.num_outputs if self.model else None

    def num_trees(self):
        return self.model.num_trees if self.model else None

    @property
    def default_chunk_size(self):
        return self.model.default_chunk_size if self.model else None

    @classmethod
    def load(
        cls,
        path,
        *,
        is_classifier=False,
        precision="single",
        model_type=None,
        output_type=None,
        verbose=False,
        default_chunk_size=None,
        align_bytes=None,
        layout="depth_first",
        device_id=0,
    ):
        warnings.warn(
            "cuml.fil.ForestInference.load() is deprecated and will be removed in 26.10. "
            "Use nvforest.load_model() instead.",
            FutureWarning,
            stacklevel=2,
        )
        obj = cls(
            output_type=output_type,
            verbose=verbose,
            _suppress_deprecation_warning=True,
        )
        obj.model = nvforest.load_model(
            model_file=path,
            model_type=model_type,
            device="gpu"
            if get_fil_device_type() == DeviceType.device
            else "cpu",
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
            precision=precision,
            device_id=device_id,
            handle=get_handle(),
        )
        return obj

    @classmethod
    def load_from_sklearn(
        cls,
        skl_model,
        *,
        is_classifier=False,
        precision="single",
        model_type=None,
        output_type=None,
        verbose=False,
        default_chunk_size=None,
        align_bytes=None,
        layout="depth_first",
        device_id=0,
    ):
        warnings.warn(
            "cuml.fil.ForestInference.load_from_sklearn() is deprecated "
            "and will be removed in 26.10. "
            "Use nvforest.load_from_sklearn() instead.",
            FutureWarning,
            stacklevel=2,
        )
        obj = cls(
            output_type=output_type,
            verbose=verbose,
            _suppress_deprecation_warning=True,
        )
        obj.model = nvforest.load_from_sklearn(
            skl_model=skl_model,
            device="gpu"
            if get_fil_device_type() == DeviceType.device
            else "cpu",
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
            precision=precision,
            device_id=device_id,
            handle=get_handle(),
        )
        return obj

    @classmethod
    def load_from_treelite_model(
        cls,
        tl_model,
        *,
        is_classifier=False,
        precision="single",
        model_type=None,
        output_type=None,
        verbose=False,
        default_chunk_size=None,
        align_bytes=None,
        layout="depth_first",
        device_id=0,
    ):
        warnings.warn(
            "cuml.fil.ForestInference.load_from_treelite_model() is deprecated "
            "and will be removed in 26.10. "
            "Use nvforest.load_from_treelite_model() instead.",
            FutureWarning,
            stacklevel=2,
        )
        obj = cls(
            output_type=output_type,
            verbose=verbose,
            _suppress_deprecation_warning=True,
        )
        obj.model = nvforest.load_from_treelite_model(
            tl_model=tl_model,
            device="gpu"
            if get_fil_device_type() == DeviceType.device
            else "cpu",
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
            precision=precision,
            device_id=device_id,
            handle=get_handle(),
        )
        return obj

    def get_dtype(self):
        if self.model is None:
            raise RuntimeError("ForestInference not yet loaded")
        return self.model.forest.get_dtype()

    @mlfunc(preserve_index=True)
    def predict_proba(
        self,
        X,
        *,
        preds=None,
        chunk_size=None,
    ):
        if self.model is None:
            raise RuntimeError("ForestInference not yet loaded")
        if preds is not None:
            raise NotImplementedError(
                "Setting preds argument is no longer supported"
            )
        if isinstance(
            self.model,
            (
                nvforest.GPUForestInferenceClassifier,
                nvforest.CPUForestInferenceClassifier,
            ),
        ):
            X = check_array(
                X,
                dtype=self.get_dtype(),
                order="C",
                mem_type="device"
                if _is_nvforest_model_on_device(self.model)
                else "host",
                ensure_all_finite=self.ensure_all_finite,
                input_name="X",
            )
            out = self.model.predict_proba(X, chunk_size=chunk_size)
            mem_type = GlobalSettings().fil_memory_type.name
            return cp.asarray(out) if mem_type == "device" else cp.asnumpy(out)
        raise RuntimeError("Must be a classifier to run predict_proba()")

    @mlfunc(preserve_index=True)
    def predict(
        self,
        X,
        *,
        preds=None,
        chunk_size=None,
        threshold=None,
    ):
        if self.model is None:
            raise RuntimeError("ForestInference not yet loaded")
        if preds is not None:
            raise NotImplementedError(
                "Setting preds argument is no longer supported"
            )
        X = check_array(
            X,
            dtype=self.get_dtype(),
            order="C",
            mem_type="device"
            if _is_nvforest_model_on_device(self.model)
            else "host",
            ensure_all_finite=self.ensure_all_finite,
            input_name="X",
        )
        if isinstance(
            self.model,
            (
                nvforest.GPUForestInferenceClassifier,
                nvforest.CPUForestInferenceClassifier,
            ),
        ):
            out = self.model.predict(
                X, chunk_size=chunk_size, threshold=threshold
            )
        elif isinstance(
            self.model,
            (
                nvforest.GPUForestInferenceRegressor,
                nvforest.CPUForestInferenceRegressor,
            ),
        ):
            out = self.model.predict(X, chunk_size=chunk_size)
        else:
            raise NotImplementedError(
                f"Unrecognized type for self.model: {type(self.model)}"
            )
        mem_type = GlobalSettings().fil_memory_type.name
        return cp.asarray(out) if mem_type == "device" else cp.asnumpy(out)

    @mlfunc(preserve_index=True)
    def predict_per_tree(self, X, *, preds=None, chunk_size=None):
        if self.model is None:
            raise RuntimeError("ForestInference not yet loaded")
        if preds is not None:
            raise NotImplementedError(
                "Setting preds argument is no longer supported"
            )
        X = check_array(
            X,
            dtype=self.get_dtype(),
            order="C",
            mem_type="device"
            if _is_nvforest_model_on_device(self.model)
            else "host",
            ensure_all_finite=self.ensure_all_finite,
            input_name="X",
        )
        out = self.model.predict_per_tree(X, chunk_size=chunk_size)
        mem_type = GlobalSettings().fil_memory_type.name
        return cp.asarray(out) if mem_type == "device" else cp.asnumpy(out)

    @mlfunc(preserve_index=True)
    def apply(self, X, *, preds=None, chunk_size=None):
        if self.model is None:
            raise RuntimeError("ForestInference not yet loaded")
        if preds is not None:
            raise NotImplementedError(
                "Setting preds argument is no longer supported"
            )
        X = check_array(
            X,
            dtype=self.get_dtype(),
            order="C",
            mem_type="device"
            if _is_nvforest_model_on_device(self.model)
            else "host",
            ensure_all_finite=self.ensure_all_finite,
            input_name="X",
        )
        out = self.model.apply(X, chunk_size=chunk_size)
        mem_type = GlobalSettings().fil_memory_type.name
        return cp.asarray(out) if mem_type == "device" else cp.asnumpy(out)

    def optimize(
        self,
        *,
        data=None,
        batch_size=1024,
        unique_batches=10,
        timeout=0.2,
        predict_method="predict",
        max_chunk_size=None,
        seed=0,
    ):
        if self.model is None:
            raise RuntimeError("ForestInference not yet loaded")
        return self.model.optimize(
            data=data,
            batch_size=batch_size,
            unique_batches=unique_batches,
            timeout=timeout,
            predict_method=predict_method,
            max_chunk_size=max_chunk_size,
            seed=seed,
        )

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "treelite_model",
            "is_classifier",
            "layout",
            "default_chunk_size",
            "align_bytes",
            "precision",
            "device_id",
            "ensure_all_finite",
            "_suppress_deprecation_warning",
        ]
