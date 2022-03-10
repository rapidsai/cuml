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
from cuml.model_selection import train_test_split
from cuml.preprocessing.LabelEncoder import LabelEncoder
from cuml.preprocessing.label import LabelBinarizer, label_binarize
from cuml.preprocessing.encoders import OneHotEncoder
from cuml.preprocessing.TargetEncoder import TargetEncoder
from cuml.preprocessing import text

from cuml._thirdparty.sklearn.preprocessing import StandardScaler, \
    MinMaxScaler, MaxAbsScaler, Normalizer, Binarizer, PolynomialFeatures, \
    SimpleImputer, RobustScaler, KBinsDiscretizer, MissingIndicator
from cuml._thirdparty.sklearn.preprocessing import scale, minmax_scale, \
    maxabs_scale, normalize, add_dummy_feature, binarize, robust_scale

__all__ = [
    # Classes
    'Binarizer',
    'KBinsDiscretizer',
    'MaxAbsScaler',
    'MinMaxScaler',
    'Normalizer',
    'PolynomialFeatures',
    'RobustScaler',
    'SimpleImputer',
    'MissingIndicator',
    'StandardScaler',
    'LabelEncoder',
    'LabelBinarizer',
    'OneHotEncoder',
    'TargetEncoder',
    # Functions
    'add_dummy_feature',
    'binarize',
    'minmax_scale',
    'maxabs_scale',
    'normalize',
    'robust_scale',
    'scale',
    'label_binarize',
    'train_test_split',
    # Modules
    'text',
]
