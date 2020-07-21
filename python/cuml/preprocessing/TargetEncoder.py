import numpy as np
import cudf as gd
import cupy as cp

class TargetEncoder:
    
    def __init__(self, folds, smooth, seed=42):
        self.folds = folds
        self.seed = seed
        self.smooth = smooth
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
                assert 0
        else:
            assert 0
        train[self.fold_col] = cp.random.randint(0,self.folds,len(train))
        
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
            assert 0
        df[self.id_col] = cp.arange(len(x))
        return df
