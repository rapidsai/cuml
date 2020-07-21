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

import numpy as np
import cudf as gd
import cupy as cp

class TargetEncoder:
    """
    A cudf based implementation of target encoding

    Parameters
    ----------
    n_folds : int (default=5)
        Default number of folds
    smooth : int (default=0)
        Number of samples to smooth the encoding
    seed : int (default=42)
        Random seed
    split : {'random', 'continuous', 'interleaved'}, default='interleaved'
    Examples
    --------
    Converting a categorical implementation to a numerical one

    .. code-block:: python

        from cudf import DataFrame, Series

        train = DataFrame({'category': ['a', 'b', 'b', 'a'],
                           'label': [1, 0, 1, 1]})
        test = DataFrame({'category': ['a', 'c', 'b', 'a']})

        # There is only one correct way to do this
        le = TargetEncoder()
        train_encoded = le.fit_transform(train.category, train.label)
        test_encoded = le.transform(test.category)
        print(train_encoded)
        print(test_encoded)

    Output:

    .. code-block:: python

        [1. 1. 0. 1.]
        [1.   0.75 0.5  1.  ]

    """    
    def __init__(self, n_folds=4, smooth=0, seed=42, split='interleaved'):
        self.n_folds = n_folds
        self.seed = seed
        self.smooth = smooth
        self.split = split 
        self.y_col = '__TARGET__'
        self.x_col = '__FEA__' 
        self.out_col = '__TARGET_ENCODE__'
        self.fold_col = '__FOLD__'
        self.id_col = '__INDEX__'
 
    def fit_transform(self, x, y):
        cp.random.seed(self.seed)
        train = self._to_frame(x)
        x_cols = [i for i in train.columns.tolist() if i!=self.id_col]
        
        if isinstance(y, gd.Series):
            train[self.y_col] = y.values
        elif isinstance(y, cp.ndarray):
            if len(y.shape) == 1:
                train[self.y_col] = y
            elif y.shape[1] == 1:
                train[self.y_col] = y[:,0]
            else:
                raise ValueError(f'Input of shape {y.shape} is not a 1-D array.')
        else:
            raise TypeError(
                "Input of type {type(y)} is not cudf.Series, "
                "or cupy.ndarray") 

        if self.split == 'random':
            train[self.fold_col] = cp.random.randint(0,self.n_folds,len(train))
        elif self.split == 'continuous':
            train[self.fold_col] = cp.arange(len(train))/(len(train)/self.n_folds)
            train[self.fold_col] = train[self.fold_col]%self.n_folds
        elif self.split == 'interleaved':
            train[self.fold_col] = cp.arange(len(train))
            train[self.fold_col] = train[self.fold_col]%self.n_folds
        else:
            msg = ("split should be either 'random' or 'continuous' or 'interleaved', "
                   "got {0}.".format(self.split))
            raise ValueError(msg)
        
        train[self.fold_col] = train[self.fold_col].astype('int32')       
        self.mean = train[self.y_col].mean()#.compute().astype('float32')
        
        cols = [self.fold_col]+x_cols
        
        agg_each_fold = train.groupby(cols).agg({self.y_col:['count','sum']}).reset_index()
        agg_each_fold.columns = cols + ['count_y','sum_y']
        
        agg_all = agg_each_fold.groupby(x_cols).agg({'count_y':'sum','sum_y':'sum'}).reset_index()
        cols = x_cols
        agg_all.columns = cols + ['count_y_all','sum_y_all']
        
        agg_each_fold = agg_each_fold.merge(agg_all,on=x_cols,how='left')
        agg_each_fold['count_y_all'] = agg_each_fold['count_y_all'] - agg_each_fold['count_y']
        agg_each_fold['sum_y_all'] = agg_each_fold['sum_y_all'] - agg_each_fold['sum_y']
        agg_each_fold[self.out_col] = (agg_each_fold['sum_y_all']+self.smooth*self.mean)/(agg_each_fold['count_y_all']+self.smooth)
        agg_each_fold = agg_each_fold.drop(['count_y_all','count_y','sum_y_all','sum_y'],axis=1)
        
        agg_all[self.out_col] = (agg_all['sum_y_all']+self.smooth*self.mean)/(agg_all['count_y_all']+self.smooth)
        agg_all = agg_all.drop(['count_y_all','sum_y_all'],axis=1)
        self.agg_all = agg_all
        
        cols = [self.fold_col]+x_cols
        train = train.merge(agg_each_fold,on=cols,how='left')
        del agg_each_fold
        return self._get_return_value(train)
    
    def transform(self, x):
        test = self._to_frame(x)
        x_cols = [i for i in test.columns.tolist() if i!=self.id_col]
        test = test.merge(self.agg_all,on=x_cols,how='left')
        return self._get_return_value(test) 

    def _get_return_value(self, df):
        df[self.out_col] = df[self.out_col].nans_to_nulls()
        df[self.out_col] = df[self.out_col].fillna(self.mean)
        df = df.sort_values(self.id_col) 
        res = df[self.out_col].values.copy()
        del df
        return res 

    def _to_frame(self, x):
        if isinstance(x, gd.DataFrame):
            df = x.copy()
        elif isinstance(x, gd.Series):
            df = x.to_frame()
        elif isinstance(x, cp.ndarray):
            df = gd.DataFrame()
            if len(x.shape) == 1:
                df[self.x_col] = x
            else:
                df = gd.DataFrame(x, columns=[f'{self.x_col}_{i}' for i in range(x.shape[1])])
        else:
            raise TypeError(
                f"Input of type {x.shape} is not cudf.Series, cudf.DataFrame "
                "or cupy.ndarray")
        df[self.id_col] = cp.arange(len(x))
        return df
