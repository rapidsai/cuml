#
# Copyright (c) 2024, NVIDIA CORPORATION.
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


import cuml
import inspect

from cuml.internals.mem_type import MemoryType
from cuml.internals import logger
from cuml.internals.safe_imports import gpu_only_import, cpu_only_import
from typing import Optional, Tuple, Dict


# currently we just use this dictionary for debugging purposes
patched_classes = {}


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
    (reasonably) create a method that just pickles and unpickles a ProxyEstimat
    as if it was just an instance of the original module.

    To overcome this limitation and offer compatibility between environments
    with acceleration and environments without, a ProxyEstimator serializes
    *both* the underlying _cpu_model as well as the ProxyEstimator itself.
    See the example below to see how it works in practice.

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
    >>> with open("ProxyEstimator.pkl", "wb") as f:
    >>>     # This saves two pickled files, a pickle corresponding to
    >>>     # the ProxyEstimator and a "ProxyEstimator_pickle.pkl" that is
    >>>     # the CPU model pickled.
    >>>     loaded = load(f)

    """

    if accelerated_class_name is None:
        accelerated_class_name = original_class_name

    # Import the original host module and cuML
    module_a = cpu_only_import(original_module)
    module_b = gpu_only_import(accelerated_module)

    # Store a reference to the original (CPU) class
    original_class_a = getattr(module_a, original_class_name)

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

        def __init__(self, *args, **kwargs):
            self._cpu_model_class = (
                original_class_a  # Store a reference to the original class
            )
            kwargs, self._gpuaccel = self._hyperparam_translator(**kwargs)
            super().__init__(*args, **kwargs)

            self._cpu_hyperparams = list(
                inspect.signature(
                    self._cpu_model_class.__init__
                ).parameters.keys()
            )

        def __repr__(self):
            """
            Return a formal string representation of the object.

            Returns
            -------
            str
                A string representation indicating that this is a wrapped
                 version of the original CPU-based estimator.
            """
            return f"wrapped {self._cpu_model_class}"

        def __str__(self):
            """
            Return an informal string representation of the object.

            Returns
            -------
            str
                A string representation indicating that this is a wrapped
                 version of the original CPU-based estimator.
            """
            return f"ProxyEstimator of {self._cpu_model_class}"

        def _check_cpu_model(self):
            """
            Checks if an estimator already has created a _cpu_model,
            and creates one if necessary.
            """
            if not hasattr(self, "_cpu_model"):
                self.import_cpu_model()
                self.build_cpu_model()

                self.gpu_to_cpu()

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
            import pickle
            from .module_accelerator import disable_module_accelerator

            with disable_module_accelerator():
                filename = self.__class__.__name__ + "_sklearn"
                with open(filename, "wb") as f:
                    self._check_cpu_model()
                    pickle.dump(self._cpu_model, f)

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

    logger.debug(
        f"Created proxy estimator: ({module_b}, {original_class_name}, {ProxyEstimator})"
    )
    setattr(module_b, original_class_name, ProxyEstimator)
    setattr(module_a, original_class_name, ProxyEstimator)

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
