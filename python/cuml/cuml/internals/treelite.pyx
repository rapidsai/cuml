#
# Copyright (c) 2025, NVIDIA CORPORATION.
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
cdef str _get_treelite_error():
    cdef str err = TreeliteGetLastError().decode("UTF-8")
    return err


def safe_treelite_call(res: int, err_msg: str) -> None:
    if res < 0:
        raise RuntimeError(f"{err_msg}\n{_get_treelite_error()}")
