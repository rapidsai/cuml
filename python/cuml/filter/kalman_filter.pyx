#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

class KalmanFilter():
    """
    NOTE: This KalmanFilter model has been deprecated and removed.
    The previous version had numerical and performance issues.
    See cuML issue #1758.
    """
    def __init__(self, dim_x, dim_z, solver='long', precision='single',
                 seed=False):
        raise NotImplementedError(
            """The Kalman filter is no longer provided due to issues with the
            previous implementation. See cuML issue #1758 for more details.""")
