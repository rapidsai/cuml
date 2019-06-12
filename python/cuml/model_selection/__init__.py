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

from cuml.model_selection._split import BaseCrossValidator
from cuml.model_selection._split import KFold
from cuml.model_selection._split import GroupKFold
from cuml.model_selection._split import StratifiedKFold
from cuml.model_selection._split import TimeSeriesSplit
from cuml.model_selection._split import LeaveOneGroupOut
from cuml.model_selection._split import LeaveOneOut
from cuml.model_selection._split import LeavePGroupsOut
from cuml.model_selection._split import LeavePOut
from cuml.model_selection._split import RepeatedKFold
from cuml.model_selection._split import RepeatedStratifiedKFold
from cuml.model_selection._split import ShuffleSplit
from cuml.model_selection._split import GroupShuffleSplit
from cuml.model_selection._split import StratifiedShuffleSplit
from cuml.model_selection._split import PredefinedSplit
from cuml.model_selection._split import train_test_split
from cuml.model_selection._split import check_cv

# from cuml.model_selection._validation import cross_val_score
# from cuml.model_selection._validation import cross_val_predict
# from cuml.model_selection._validation import cross_validate
# from cuml.model_selection._validation import learning_curve
# from cuml.model_selection._validation import permutation_test_score
# from cuml.model_selection._validation import validation_curve

# from cuml.model_selection._search import GridSearchCV
# from cuml.model_selection._search import RandomizedSearchCV
# from cuml.model_selection._search import ParameterGrid
# from cuml.model_selection._search import ParameterSampler
# from cuml.model_selection._search import fit_grid_point

__all__ = ('BaseCrossValidator',
           # 'GridSearchCV',
           'TimeSeriesSplit',
           'KFold',
           'GroupKFold',
           'GroupShuffleSplit',
           'LeaveOneGroupOut',
           'LeaveOneOut',
           'LeavePGroupsOut',
           'LeavePOut',
           'RepeatedKFold',
           'RepeatedStratifiedKFold',
           # 'ParameterGrid',
           # 'ParameterSampler',
           'PredefinedSplit',
           # 'RandomizedSearchCV',
           'ShuffleSplit',
           'StratifiedKFold',
           'StratifiedShuffleSplit',
           'check_cv',
           # 'cross_val_predict',
           # 'cross_val_score',
           # 'cross_validate',
           # 'fit_grid_point',
           # 'learning_curve',
           # 'permutation_test_score',
           'train_test_split')
           # 'validation_curve')