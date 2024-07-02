#
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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


from packaging.version import Version

MIN_SKLEARN_VERSION = Version('1.5')


try:
    import sklearn  # noqa: F401  # no-cython-lint

    CPU_ENABLED = True

    if(Version(sklearn.__version__) >= MIN_SKLEARN_VERSION):
        MIN_SKLEARN_PRESENT = (True, None, None)
    else:
        MIN_SKLEARN_PRESENT = (False, sklearn.__version__, MIN_SKLEARN_VERSION)

except ImportError:
    CPU_ENABLED = False
    MIN_SKLEARN_PRESENT = (False, None, None)

IF GPUBUILD == 1:
    GPU_ENABLED = True

ELSE:
    GPU_ENABLED = False
