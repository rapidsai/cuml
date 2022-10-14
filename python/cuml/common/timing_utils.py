#
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
import time
from contextlib import contextmanager


# Helper function for timing blocks of code.
@contextmanager
def timed(name):
    """
    For timing blocks of code.

    Examples
    --------

    >>> with timed("Print Call"):
    ...     print("Hello, World") # doctest: +SKIP
    Hello, World
    ..Print Call              :    0.0005

    """
    t0 = time.time()
    yield
    t1 = time.time()
    print("..%-24s:  %8.4f" % (name, t1 - t0))
