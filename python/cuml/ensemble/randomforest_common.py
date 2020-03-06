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


def _obtain_treelite_model(treelite_handle):
    """
    Converts the cuML RF model to a Treelite model

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
        Create a Forest Inference (FIL) model from the trained cuML
        Random Forest model.

        Parameters
        ----------
        output_class: boolean (default = True)
            This is optional and required only while performing the
            predict operation on the GPU.
            If true, return a 1 or 0 depending on whether the raw
            prediction exceeds the threshold. If False, just return
            the raw prediction.
        algo : string (default = 'auto')
            This is optional and required only while performing the
            predict operation on the GPU.
            'naive' - simple inference using shared memory
            'tree_reorg' - similar to naive but trees rearranged to be more
                           coalescing-friendly
            'batch_tree_reorg' - similar to tree_reorg but predicting
                                 multiple rows per thread block
            `auto` - choose the algorithm automatically. Currently
                     'batch_tree_reorg' is used for dense storage
                     and 'naive' for sparse storage
        threshold : float (default = 0.5)
            Threshold used for classification. Optional and required only
            while performing the predict operation on the GPU.
            It is applied if output_class == True, else it is ignored
        fil_sparse_format : boolean or string (default = auto)
            This variable is used to choose the type of forest that will be
            created in the Forest Inference Library. It is not required
            while using predict_model='CPU'.
            'auto' - choose the storage type automatically
                     (currently True is chosen by auto)
             False - create a dense forest
             True - create a sparse forest, requires algo='naive'
                    or algo='auto'
        Returns
        ----------
        fil_model :
           A Forest Inference model which can be used to perform
           inferencing on the random forest model.
    """

    storage_type = _check_fil_value(fil_sparse_format)

    _check_fil_parameter_validity(depth=depth,
                                  storage_format=storage_type,
                                  algo=algo)

    fil_model = ForestInference()
    tl_to_fil_model = \
        fil_model.load_from_randomforest(treelite_handle,
                                         output_class=output_class,
                                         threshold=threshold,
                                         algo=algo,
                                         storage_type=storage_type)

    return tl_to_fil_model
