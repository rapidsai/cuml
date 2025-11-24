#
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import contextlib
from collections import deque

import cuml.internals.input_utils as iu
from cuml.internals.array_sparse import SparseCumlArray
from cuml.internals.global_settings import GlobalSettings


@contextlib.contextmanager
def _using_mirror_output_type():
    """
    Sets global_settings.output_type to "mirror" for internal API
    handling. We need a separate function since `cuml.using_output_type()`
    doesn't accept "mirror"

    Yields
    -------
    string
        Returns the previous value in global_settings.output_type
    """
    prev_output_type = GlobalSettings().output_type
    try:
        GlobalSettings().output_type = "mirror"
        yield prev_output_type
    finally:
        GlobalSettings().output_type = prev_output_type


def set_api_output_type(output_type: str):
    assert GlobalSettings().root_cm is not None

    # Quick exit
    if isinstance(output_type, str):
        GlobalSettings().root_cm.output_type = output_type
        return

    # Try to convert any array objects to their type
    array_type = iu.determine_array_type(output_type)

    # Ensure that this is an array-like object
    assert output_type is None or array_type is not None

    GlobalSettings().root_cm.output_type = array_type


class InternalAPIContext(contextlib.ExitStack):
    def __init__(self):
        super().__init__()

        def cleanup():
            GlobalSettings().root_cm = None

        self.callback(cleanup)

        self.prev_output_type = self.enter_context(_using_mirror_output_type())

        # Set the output type to the prev_output_type. If "input", set to None
        # to allow inner functions to specify the input
        self.output_type = (
            None if self.prev_output_type == "input" else self.prev_output_type
        )

        self._count = 0

        self.call_stack = {}

        GlobalSettings().root_cm = self

    def pop_all(self):
        """Preserve the context stack by transferring it to a new instance."""
        new_stack = contextlib.ExitStack()
        new_stack._exit_callbacks = self._exit_callbacks
        self._exit_callbacks = deque()
        return new_stack

    def __enter__(self) -> int:
        self._count += 1

        return self._count

    def __exit__(self, *exc_details):
        self._count -= 1

        return

    @contextlib.contextmanager
    def push_output_types(self):
        try:
            old_output_type = self.output_type
            self.output_type = None
            yield
        finally:
            self.output_type = (
                old_output_type
                if old_output_type is not None
                else self.output_type
            )


def get_internal_context() -> InternalAPIContext:
    """Return the current "root" context manager used to control output type
    for external API calls and minimize unnecessary internal output
    conversions"""

    if GlobalSettings().root_cm is None:
        GlobalSettings().root_cm = InternalAPIContext()

    return GlobalSettings().root_cm


class InternalAPIContextBase(contextlib.ExitStack):
    def __init__(self, base=None, process_return=None):
        super().__init__()

        self._base = base
        self._process_return = process_return

        self.root_cm = get_internal_context()

        self.is_root = False

        self._should_set_output_type_from_base = (
            self._base is not None
            and self.root_cm.prev_output_type in (None, "input")
        )

    def set_output_type_from_base(self, root_cm):
        output_type = root_cm.output_type

        # Check if output_type is None, can happen if no output type has
        # been set by estimator
        if output_type is None:
            output_type = self._base.output_type

        if output_type == "input":
            output_type = self._base._input_type

        if output_type != root_cm.output_type:
            set_api_output_type(output_type)

        assert output_type != "mirror"

    def __enter__(self):
        # Enter the root context to know if we are the root cm
        self.is_root = self.enter_context(self.root_cm) == 1

        # If we are the first, push any callbacks from the root into this CM
        # If we are not the first, this will have no effect
        self.push(self.root_cm.pop_all())

        if self._process_return:
            self.enter_context(self.root_cm.push_output_types())
        if self._should_set_output_type_from_base:
            self.callback(self.set_output_type_from_base, self.root_cm)

        # Only convert output:
        # - when returning results from a root api call
        # - when the output type is explicitly set
        self._convert_output = (
            self.is_root or GlobalSettings().output_type != "mirror"
        )

        return super().__enter__()

    def process_return(self, res):
        """Traverse a result, converting it to the proper output type"""
        if isinstance(res, tuple):
            return tuple(self.process_return(i) for i in res)
        elif isinstance(res, list):
            return [self.process_return(i) for i in res]
        elif isinstance(res, dict):
            return {k: self.process_return(v) for k, v in res.items()}

        # Get the output type
        arr_type, is_sparse = iu.determine_array_type_full(res)

        if arr_type is None:
            # Not an array, just return
            return res

        # If we are a supported array and not already cuml, convert to cuml
        if arr_type != "cuml":
            if is_sparse:
                res = SparseCumlArray(res, convert_index=False)
            else:
                res = iu.input_to_cuml_array(res, order="K").array

        if not self._convert_output:
            # Return CumlArray/SparseCumlArray directly
            return res

        output_type = GlobalSettings().output_type

        if output_type in (None, "mirror", "input"):
            output_type = self.root_cm.output_type

        assert (
            output_type is not None
            and output_type != "mirror"
            and output_type != "input"
        ), ("Invalid root_cm.output_type: '{}'.").format(output_type)

        return res.to_output(output_type=output_type)
