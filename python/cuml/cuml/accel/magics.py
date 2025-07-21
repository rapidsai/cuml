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

from cuml.accel.core import install
from cuml.accel.runners import exec_source


def load_ipython_extension(ip):
    from IPython.core.magic import Magics, cell_magic, magics_class

    @magics_class
    class CumlAccelMagics(Magics):
        @cell_magic("cuml.accel.profile")
        def profile(self, _, cell):
            """Run a cell with the cuml.accel profiler."""
            cell = self.shell.transform_cell(cell)
            exec_source(cell, self.shell.user_ns, profile=True)

        @cell_magic("cuml.accel.line_profile")
        def line_profile(self, _, cell):
            """Run a cell with the cuml.accel line profiler."""
            cell = self.shell.transform_cell(cell)
            exec_source(cell, self.shell.user_ns, line_profile=True)

    install()
    ip.register_magics(CumlAccelMagics)
