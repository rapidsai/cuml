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

from cuml import ForestInference
from cuml.fil.fil import TreeliteModel as tl


def _check_fil_parameter_validity(depth, algo, fil_sparse_format):
    storage_format = _check_fil_sparse_format_value(fil_sparse_format)
    if (depth > 16 and (storage_format == 'dense' or
                        algo == 'tree_reorg' or
                        algo == 'batch_tree_reorg')):
        raise ValueError("While creating a forest with max_depth greater "
                         "than 16, `fil_sparse_format` should be True. "
                         "If `fil_sparse_format=False` then the memory"
                         "consumed while creating the FIL forest is very "
                         "large and the process will be aborted. In "
                         "addition, `algo` must be either set to `naive' "
                         "or `auto` to set 'fil_sparse_format=True`.")
    return storage_format


def _check_fil_sparse_format_value(fil_sparse_format):
    accepted_vals = [True, False, 'auto']
    if fil_sparse_format == 'auto':
        storage_format = fil_sparse_format
    elif not fil_sparse_format:
        storage_format = 'dense'
    elif fil_sparse_format not in accepted_vals:
        raise ValueError("The value entered for spares_forest is not "
                         "supported. Please refer to the documentation "
                         "to see the accepted values.")
    else:
        storage_format = 'sparse'

    return storage_format


def _obtain_treelite_model(treelite_handle):
    """
    Creates a Treelite model using the treelite handle
    obtained from the cuML Random Forest model.

    Returns
    ----------
    tl_to_fil_model : Treelite version of this model
    """
    treelite_model = \
        tl.from_treelite_model_handle(treelite_handle)
    return treelite_model


def _obtain_fil_model(treelite_handle, depth,
                      output_class=True,
                      threshold=0.5, algo='auto',
                      fil_sparse_format='auto'):
    """
    Creates a Forest Inference (FIL) model using the treelite
    handle obtained from the cuML Random Forest model.

    Returns
    ----------
    fil_model :
        A Forest Inference model which can be used to perform
        inferencing on the random forest model.
    """

    storage_format = \
        _check_fil_parameter_validity(depth=depth,
                                      fil_sparse_format=fil_sparse_format,
                                      algo=algo)

    fil_model = ForestInference()
    tl_to_fil_model = \
        fil_model.load_from_randomforest(treelite_handle,
                                         output_class=output_class,
                                         threshold=threshold,
                                         algo=algo,
                                         storage_type=storage_format)

    return tl_to_fil_model
