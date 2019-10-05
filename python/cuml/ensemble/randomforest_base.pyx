#
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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3
from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_dev_array, zeros
from cuml import ForestInference

from cuml.common.base import Base
from cuml.ensemble.randomforest_shared cimport *


class RandomForestBase(Base):
    def save_treelite_protobuf(self, file_name):
        file_name_bytes = bytes(file_name, "utf8")

        tl = self._get_treelite(self.n_cols, task_category=self.n_classes)
        cdef ModelHandle tl_handle = <ModelHandle><size_t>tl.value
        res = TreeliteExportProtobufModel(file_name_bytes, tl_handle)
        if res < 0:
            last_err = TreeliteGetLastError()
            raise RuntimeError("Failed to export model to %s: %s" % (
                file_name, last_err))

    def _predict_model_on_gpu(self,
                              X,
                              algo='BATCH_TREE_REORG',
                              output_class=False,
                              threshold=0.5,
                              n_classes=1):
        _, _, n_rows, n_cols, _ = \
            input_to_dev_array(X, order='C')
        if n_cols != self.n_cols:
            raise ValueError("The number of columns/features in the training"
                             " and test data should be the same ")

        treelite_model = self._get_treelite(num_features=n_cols,
                                            task_category=self.n_classes)

        fil_model = ForestInference()
        tl_to_fil_model = \
            fil_model.load_from_randomforest(treelite_model.value,
                                             output_class=output_class,
                                             threshold=threshold,
                                             algo=algo)
        preds = tl_to_fil_model.predict(X)
        return preds
