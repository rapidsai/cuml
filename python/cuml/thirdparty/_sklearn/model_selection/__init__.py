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

from cuml.thirdparty._sklearn.model_selection._split import (
    BaseCrossValidator, KFold, GroupKFold, StratifiedKFold, TimeSeriesSplit,
    LeaveOneGroupOut, LeaveOneOut, LeavePGroupsOut, LeavePOut, RepeatedKFold,
    RepeatedStratifiedKFold, ShuffleSplit, GroupShuffleSplit,
    StratifiedShuffleSplit, PredefinedSplit, train_test_split, check_cv)

# from cuml.thirdparty._sklearn.model_selection._validation import (
#     cross_val_score, cross_val_predict, cross_validate, learning_curve,
#     permutation_test_score, validation_curve)

# from cuml.thirdparty._sklearn.model_selection._search import (
#     GridSearchCV, RandomizedSearchCV, ParameterGrid, ParameterSampler,
#     fit_grid_point)

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