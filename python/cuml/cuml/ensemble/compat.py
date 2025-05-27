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

import ctypes
import functools
import inspect
import warnings

from treelite import Model as TreeliteModel
from treelite.core import _LIB as treelite_LIB
from treelite.core import _check_call


class TreeliteModelCompat(TreeliteModel):
    """A compatibility wrapper for treelite.Model that adds deprecated properties.

    This class is used to maintain backward compatibility with code that expects
    certain properties to be available on treelite.Model instances.
    """

    @classmethod
    def deserialize_bytes(cls, model_bytes: bytes) -> "TreeliteModelCompat":
        # Adapted from treelite/model.py
        handle = ctypes.c_void_p()
        bytes_len = len(model_bytes)
        bytes_buffer = ctypes.create_string_buffer(model_bytes, bytes_len)

        _check_call(
            treelite_LIB.TreeliteDeserializeModelFromBytes(
                ctypes.pointer(bytes_buffer),
                ctypes.c_size_t(bytes_len),
                ctypes.byref(handle),
            )
        )

        return TreeliteModelCompat(handle=handle)

    @property
    def num_trees(self):
        """Deprecated property that returns the number of trees in the model.

        This property is deprecated and will be removed in a future version.
        Please use num_tree instead.
        """
        warnings.warn(
            "Property 'num_trees' was deprecated in version 25.06 and will be removed in 25.08. "
            "Please use 'num_tree' instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.num_tree

    @property
    def num_features(self):
        """
        Deprecated property. Use num_feature instead.
        """
        warnings.warn(
            "Property 'num_features' was deprecated in version 25.06 and will be "
            "removed in 25.08. Please use 'num_feature' instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.num_feature

    def to_treelite_checkpoint(self, filename):
        """Deprecated method that serializes the model to a file.

        This method is deprecated and will be removed in a future version.
        Please use serialize() instead.

        Parameters
        ----------
        filename : str
            Path to the output file
        """
        warnings.warn(
            "Method 'to_treelite_checkpoint()' was deprecated in version 25.06 and will be "
            "removed in 25.08. Please use 'serialize()' instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.serialize(filename)


# TODO: Remove in 25.08 - This decorator handles deprecations for 25.06 and
# should be removed in 25.08.
def _handle_deprecated_rf_args(*deprecated_params):
    """Decorator to handle deprecated arguments in a function.

    Parameters
    ----------
    *deprecated_params : tuple
        Variable length tuple of parameter names that are deprecated

    Returns
    -------
    callable
        Decorated function that handles deprecated arguments

    Raises
    ------
    ValueError
        If any unexpected keyword arguments remain after processing
        If fil_sparse_format is "not_supported" or if fil_sparse_format is False
        and algo is "tree_reorg" or "batch_tree_reorg"
    """

    def decorator(func):
        # Get the function's signature to check for valid parameters
        sig = inspect.signature(func)
        valid_params = {
            name
            for name, param in sig.parameters.items()
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate deprecated parameters if they were provided
            if "fil_sparse_format" in kwargs or "algo" in kwargs:
                fil_sparse_format = kwargs.get("fil_sparse_format", "auto")
                algo = kwargs.get("algo", "auto")

                if fil_sparse_format == "not_supported":
                    raise ValueError(
                        "fil_sparse_format='not_supported' is not supported"
                    )
                if not fil_sparse_format or algo in [
                    "tree_reorg",
                    "batch_tree_reorg",
                ]:
                    raise ValueError(
                        f"fil_sparse_format=False is not supported with algo={algo}"
                    )

            if "output_class" in kwargs:
                warnings.warn(
                    "Parameter `output_class` was deprecated in version 25.06 and will be "
                    "removed in 25.08. Use `is_classifier` instead.",
                    FutureWarning,
                )
                kwargs["is_classifier"] = kwargs.pop("output_class")

            for param in deprecated_params:
                if param in kwargs:
                    warnings.warn(
                        f"Parameter `{param}` was deprecated in version 25.06 and will be "
                        "removed in 25.08. Use `layout`, `default_chunk_size`, and "
                        "`align_bytes` instead.",
                        FutureWarning,
                    )
                    kwargs.pop(param)

            # Check for unexpected keyword arguments that aren't in the function signature
            unexpected_kwargs = set(kwargs.keys()) - valid_params
            if unexpected_kwargs:
                raise ValueError(
                    f"Unexpected keyword arguments: {list(unexpected_kwargs)}"
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator
