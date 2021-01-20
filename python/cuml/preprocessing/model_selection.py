
# Copyright (c) 2019, NVIDIA CORPORATION.
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
import warnings

from cuml.model_selection._split import _stratify_split  # noqa: F401
from cuml.model_selection._split import _approximate_mode  # noqa: F401
from cuml.model_selection._split import train_test_split  # noqa: F401


warnings.warn("cuml.preprocessing.model_selection is deprecated and will "
              "be removed in v0.18. Use cuml.model_selection instead.",
              DeprecationWarning)
