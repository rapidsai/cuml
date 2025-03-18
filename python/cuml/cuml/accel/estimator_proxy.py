#
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

import inspect
import sys
import types
from typing import Optional, Tuple, Dict, Any, Type, List

from cuml.internals import logger
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.safe_imports import gpu_only_import, cpu_only_import


class ProxyModule:
    """
    A proxy module that dynamically replaces specified classes with proxy estimators
    based on GlobalSettings.

    Parameters
    ----------
    original_module : module
        The module to be proxied.
    Attributes
    ----------
    _original_module : module
        The original module being proxied.
    _proxy_estimators : dict of str to type
        A dictionary mapping accelerated class names to their proxy estimators.
    """

    def __init__(self, original_module: types.ModuleType) -> None:
        """Initialize the ProxyModule with the original module."""
        self._original_module = original_module
        self._proxy_estimators: Dict[str, Type[Any]] = {}

    def add_estimator(
        self, class_name: str, proxy_estimator: Type[Any]
    ) -> None:
        """
        Add a proxy estimator for a specified class name.
        Parameters
        ----------
        class_name : str
            The name of the class in the original module to be replaced.
        proxy_estimator : type
            The proxy estimator class to use as a replacement.
        """
        self._proxy_estimators[class_name] = proxy_estimator

    def __getattr__(self, name: str) -> Any:
        """
        Intercept attribute access on the proxy module.
        If the attribute name is in the proxy estimators and the accelerator is active,
        return the proxy estimator; otherwise, return the attribute from the original module.
        Parameters
        ----------
        name : str
            The name of the attribute being accessed.
        Returns
        -------
        Any
            The attribute from the proxy estimator or the original module.
        """
        if name in self._proxy_estimators:
            use_proxy = getattr(GlobalSettings(), "accelerator_active", False)
            if use_proxy:
                return self._proxy_estimators[name]
            else:
                return getattr(self._original_module, name)
        else:
            return getattr(self._original_module, name)

    def __dir__(self) -> List[str]:
        """
        Provide a list of attributes available in the proxy module.
        Returns
        -------
        list of str
            A list of attribute names from the original module.
        """
        return dir(self._original_module)


def intercept(
    original_module: str,
    accelerated_module: str,
    original_class_name: str,
    accelerated_class_name: Optional[str] = None,
):
    """
    Factory function that creates class definitions of ProxyEstimators that
    accelerate estimators of the original class.

    This function dynamically creates a new class called `ProxyEstimator` that
    inherits from the GPU-accelerated class in the `accelerated_module`
    (e.g., cuML) and acts as a drop-in replacement for the original class in
    `original_module` (e.g., scikit-learn). Then, this class can be used to
    create instances of ProxyEstimators that dispatch to either library.

    **Design of the ProxyEstimator Class Inside**

    **`ProxyEstimator` Class:**
        - The `ProxyEstimator` class inherits from the GPU-accelerated
        class (`class_b`) obtained from the `accelerated_module`.
        - It serves as a wrapper that adds additional functionality
        to maintain compatibility with the original CPU-based estimator.
        Key methods and attributes:
            - `__init__`: Initializes the proxy estimator, stores a
            reference to the original class before ModuleAccelerator
            replaces the original module, translates hyperparameters,
            and initializes the parent (cuML) class.
            - `__repr__` and `__str__`: Provide string representations
            that reference the original CPU-based class.
            - Attribute `_cpu_model_class`: Stores a reference to the
            original CPU-based estimator class.
            - Attribute `_gpuaccel`: Indicates whether GPU acceleration
            is enabled.
            - By designing the `ProxyEstimator` in this way, we can
            seamlessly replace the original CPU-based estimator with a
            GPU-accelerated version without altering the existing codebase.
            The metaclass ensures that the class behaves and appears
            like the original estimator, while the proxy class manages
            the underlying acceleration and compatibility.

    **Serialization/Pickling of ProxyEstimators**

    Since pickle has strict rules about serializing classes, we cannot
    (reasonably) create a method that just pickles and unpickles a
    ProxyEstimator as if it was just an instance of the original module.

    Therefore, doing a pickling of ProxyEstimator will make it serialize to
    a file that can be opened in systems with cuML installed (CPU or GPU).
    To serialize for non cuML systems, the to_sklearn and from_sklearn APIs
    are being introduced in

    https://github.com/rapidsai/cuml/pull/6102

    Parameters
    ----------
    original_module : str
        Original module that is being accelerated
    accelerated_module : str
        Acceleration module
    class_name: str
        Name of class beign accelerated
    accelerated_class_name : str, optional
        Name of accelerator class. If None, then it is assumed it is the same
        name as class_name (i.e. the original class in the original module).

    Returns
    -------
    A class definition of ProxyEstimator that inherits from
    the accelerated library class (cuML).

    Examples
    --------
    >>> from module_accelerator import intercept
    >>> ProxyEstimator = intercept('sklearn.linear_model',
    ...                            'cuml.linear_model', 'LinearRegression')
    >>> model = ProxyEstimator()

    """

    if accelerated_class_name is None:
        accelerated_class_name = original_class_name

    # Import the original host module and cuML
    module_a = cpu_only_import(original_module)
    module_b = gpu_only_import(accelerated_module)

    # Store a reference to the original (CPU) class
    original_class_a = getattr(module_a, original_class_name)

    original_class_sig = inspect.signature(original_class_a)

    # Get the class from cuML so ProxyEstimator inherits from it
    class_b = getattr(module_b, accelerated_class_name)

    class ProxyEstimator(class_b):
        """
        A proxy estimator class that wraps the accelerated estimator and provides
        compatibility with the original estimator interface.

        The ProxyEstimator inherits from the accelerated estimator class and
        wraps additional functionality to maintain compatibility with the original
        CPU-based estimator.

        It handles the translation of hyperparameters and the transfer of models
        between CPU and GPU.

        """

        _cpu_model_class = original_class_a
        _cpu_hyperparams = list(original_class_sig.parameters.keys())

        def __init__(self, *args, **kwargs):
            # The cuml signature may not align with the sklearn signature.
            # Additionally, some sklearn models support positional arguments.
            # To work around this, we
            # - Bind arguments to the sklearn signature
            # - Convert the arguments to named parameters
            # - Translate them to cuml equivalents
            # - Then forward them on to the cuml class
            bound = original_class_sig.bind_partial(*args, **kwargs)
            translated_kwargs, self._gpuaccel = self._hyperparam_translator(
                **bound.arguments
            )
            super().__init__(**translated_kwargs)
            self.build_cpu_model(**kwargs)

        def __repr__(self):
            """
            Return a formal string representation of the object.

            Returns
            -------
            str
                A string representation indicating that this is a wrapped
                 version of the original CPU-based estimator.
            """
            return self._cpu_model.__repr__()

        def __str__(self):
            """
            Return an informal string representation of the object.

            Returns
            -------
            str
                A string representation indicating that this is a wrapped
                 version of the original CPU-based estimator.
            """
            return self._cpu_model.__str__()

        def __getstate__(self):
            """
            Prepare the object state for pickling. We need it since
            we have a custom function in __reduce__.

            Returns
            -------
            dict
                The state of the Estimator.
            """
            return self.__dict__.copy()

        def __reduce__(self):
            """
            Helper for pickle.

            Returns
            -------
            tuple
                A tuple containing the callable to reconstruct the object
                and the arguments for reconstruction.

            Notes
            -----
            Disables the module accelerator during pickling to ensure correct serialization.
            """
            return (
                reconstruct_proxy,
                (
                    original_module,
                    accelerated_module,
                    original_class_name,
                    (),
                    self.__getstate__(),
                ),
            )

    # Help make the proxy class look more like the original class
    for attr in (
        "__module__",
        "__name__",
        "__qualname__",
        "__doc__",
        "__annotate__",
        "__type_params__",
    ):
        try:
            value = getattr(original_class_a, attr)
        except AttributeError:
            pass
        else:
            setattr(ProxyEstimator, attr, value)

    ProxyEstimator.__init__.__signature__ = inspect.signature(
        original_class_a.__init__
    )

    logger.debug(
        f"Created proxy estimator: ({module_b}, {original_class_name}, {ProxyEstimator})"
    )
    setattr(module_b, original_class_name, ProxyEstimator)
    accelerated_modules = GlobalSettings().accelerated_modules

    if original_module in accelerated_modules:
        proxy_module = accelerated_modules[original_module]
    else:
        proxy_module = ProxyModule(original_module=module_a)
        GlobalSettings().accelerated_modules[original_module] = proxy_module

    proxy_module.add_estimator(
        class_name=original_class_name, proxy_estimator=ProxyEstimator
    )

    sys.modules[original_module] = proxy_module

    return ProxyEstimator


def reconstruct_proxy(
    original_module: str,
    accelerated_module: str,
    class_name: str,
    args: Tuple,
    kwargs: Dict,
):
    """
    Function to enable pickling of ProxyEstimators since they are defined inside
    a function, which Pickle doesn't like without a function or something
    that has an absolute import path like this function.

    Parameters
    ----------
    original_module : str
        Original module that is being accelerated
    accelerated_module : str
        Acceleration module
    class_name: str
        Name of class beign accelerated
    args : Tuple
        Args of class to be deserialized (typically empty for ProxyEstimators)
    kwargs : Dict
        Keyword arguments to reconstruct the ProxyEstimator instance, typically
        state from __setstate__ method.

    Returns
    -------
    Instance of ProxyEstimator constructed with the kwargs passed to the function.

    """
    # We probably don't need to intercept again here, since we already stored
    # the variables in _wrappers
    cls = intercept(
        original_module=original_module,
        accelerated_module=accelerated_module,
        original_class_name=class_name,
    )

    estimator = cls()
    estimator.__dict__.update(kwargs)
    return estimator
