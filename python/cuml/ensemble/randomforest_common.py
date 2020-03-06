#
# Copyright (c) 2020, NVIDIA CORPORATION.
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


def _check_fil_parameter_validity(depth, storage_format, algo):
    if (depth > 16 and (storage_format == 'DENSE' or
                        algo == 'tree_reorg' or
                        algo == 'batch_tree_reorg')):
        raise ValueError("While creating a forest with max_depth greater "
                         "than 16, `fil_sparse_format` should be True. "
                         "If `fil_sparse_format=False` then the memory"
                         "consumed while creating the FIL forest is very "
                         "large and the process will be aborted. In "
                         "addition, `algo` must be either set to `naive' "
                         "or `auto` to set 'fil_sparse_format=True`.")


def _check_fil_value(fil_sparse_format):
    if fil_sparse_format == 'auto':
        storage_type = fil_sparse_format
    elif not fil_sparse_format:
        storage_type = 'DENSE'
    elif fil_sparse_format:
        storage_type = 'SPARSE'
    else:
        raise ValueError("The value entered for spares_forest is not "
                         "supported. Please refer to the documentation "
                         "to see the accepted values.")
    return storage_type
