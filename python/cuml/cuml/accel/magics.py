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
import ast
import os
import re
import sys

from cuml.accel import profilers
from cuml.accel.core import install


def run_cell_with_profiler(source, namespace, profiler):
    """Run a cell source with an active profiler.

    Returns the cell output (if any).
    """
    # Generate a unique filename for this cell
    filename = f"<cuml-accel-input-{os.urandom(6).hex()}>"

    # Any cell ending with an expression and not a statement should render the
    # output. The way IPython handles this is a bit nuanced, and relies on some
    # obscure details of the Python interpreter
    #
    # - Expressions to generate output cells (and populate magic output
    # variables like `_`) should be compiled as interactive nodes. The easiest
    # way to do this is to work at the ast layer and wrap the ``ast.Expr`` in a
    # ``ast.Interactive`` node before compilation.
    #
    # - Upon execution of an interactive node, the python interpreter will call
    # `sys.displayhook`. IPython overrides this hook to handle rendering
    # output, etc...
    #
    # - We temporarily override the hook ourselves to allow extracting the cell
    # output to return directly. This allows our magics to compose nicely with
    # other IPython features. Failing to do this (and instead letting IPython's
    # displayhook be called upon execution) can result in double rendering of
    # results, and/or results rendered before the magic cells complete. IPython
    # will re-call the displayhook on our returned result.
    #
    # - IPython itself handles silencing lines that end with `;`. We always
    # generate an Interactive node for these and let IPython do the filtering
    # later.
    #
    # While this uses some obscure features, these features are how both the
    # builtin and IPython interpreters work and are documented in the standard library.
    # I don't expect this code to be brittle to upstream changes, as the standard library
    # is pretty stable.

    output = None

    def displayhook(obj):
        """A displayhook for capturing the cell output (if any)"""
        nonlocal output
        output = obj

    tree = ast.parse(source, filename=filename, mode="exec")
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        # The last element of the body is an expression. Split the body
        # into non-interactive/interactive bits.
        blocks = []
        if head := tree.body[:-1]:
            blocks.append(compile(ast.Module(head), filename, "exec"))
        blocks.append(
            compile(ast.Interactive([tree.body[-1]]), filename, "single")
        )
    else:
        # No interactive components needed, just compile as normal
        blocks = [compile(tree, filename, "exec")]

    # Temporarily patch out the displayhook so we can grab the output
    orig_displayhook = sys.displayhook
    sys.displayhook = displayhook
    try:
        with profiler:
            # Execute the blocks in order
            for block in blocks:
                exec(block, namespace)
    finally:
        sys.displayhook = orig_displayhook

    return output


def load_ipython_extension(ip):
    from IPython.core.magic import (
        Magics,
        cell_magic,
        magics_class,
        output_can_be_silenced,
    )

    @magics_class
    class CumlAccelMagics(Magics):
        @cell_magic("cuml.accel.profile")
        @output_can_be_silenced
        def profile(self, _, cell):
            """Run a cell with the cuml.accel profiler."""
            cell = self.shell.transform_cell(cell)

            return run_cell_with_profiler(
                cell, self.shell.user_ns, profilers.profile()
            )

        @cell_magic("cuml.accel.line_profile")
        @output_can_be_silenced
        def line_profile(self, _, cell):
            """Run a cell with the cuml.accel line profiler."""
            # When cell magics are composed the `exec` call that executes the
            # code in this cell may come from a different nested magic. As
            # such, we cannot rely on the filename generated here matching the
            # one that's used for execution. Instead, we inject a call to
            # `LineProfiler.start` into the user code, and rely on that to both
            # start the profiler and detect the frame/filename to use.

            # Split off any leading cell magics from the rest of the user code
            index = 0
            for match in re.finditer("(?m)^%%.*$", cell):
                index = match.end()

            source = cell[index:].strip("\n\r")

            # Inject a private call to `LineProfiler.start` before the rest of
            # the python source
            cell = "\n".join(
                [cell[:index], "__start_cuml_accel_line_profiler__()", source]
            )

            # Next apply IPython's standard cell transformations. After this ``cell``
            # is just standard python code.
            cell = self.shell.transform_cell(cell)

            profiler = profilers.LineProfiler(source=source, filename=None)

            # Temporarily inject ``profiler.start`` into the user's namespace
            # so it may be called during execution.
            self.shell.user_ns[
                "__start_cuml_accel_line_profiler__"
            ] = profiler.start
            try:
                return run_cell_with_profiler(
                    cell, self.shell.user_ns, profiler
                )
            finally:
                self.shell.user_ns.pop(
                    "__start_cuml_accel_line_profiler__", None
                )

    install()
    ip.register_magics(CumlAccelMagics)
