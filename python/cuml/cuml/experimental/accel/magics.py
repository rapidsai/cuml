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


try:
    from IPython.core.magic import Magics, cell_magic, magics_class

    # from .profiler import Profiler, lines_with_profiling

    # @magics_class
    # class CumlAccelMagic(Magics):
    #     @cell_magic("cuml.accelerator.profile")
    #     def profile(self, _, cell):
    #         with Profiler() as profiler:
    #             get_ipython().run_cell(cell)  # noqa: F821
    #         profiler.print_per_function_stats()

    #     @cell_magic("cuml.accelerator.line_profile")
    #     def line_profile(self, _, cell):
    #         new_cell = lines_with_profiling(cell.split("\n"))
    #         get_ipython().run_cell(new_cell)  # noqa: F821

    def load_ipython_extension(ip):
        from . import install

        install()
        # ip.register_magics(CumlAccelMagic)

except ImportError:

    def load_ipython_extension(ip):
        pass
