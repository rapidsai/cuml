# Copyright (c) 2018, NVIDIA CORPORATION.
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
import pandas as pd
from sklearn.decomposition import TruncatedSVD as skTSVD
from test_utils import timer,load_mortgage,pd2pygdf,array_equal,parse_args,write_log
import pytest

@pytest.mark.xfail
def test_tsvd_mortgage(nrows=1000,ncols=100,n_components=10,
        algorithm='randomized',random_state=42,
        threshold=1e-3,data_source = 'random',use_assert=True,
        quarters=8,rows_per_quarter=100000,test_model='cuml'):
    print()
    if data_source=='random':
        X = np.random.rand(nrows,ncols)
    elif data_source=='mortgage':
        X = load_mortgage(quarters=quarters,rows_per_quarter=rows_per_quarter)
        X = X[np.random.randint(0,X.shape[0]-1,nrows),:ncols]
    else:
        raise NotImplementedError
    X = pd.DataFrame({'fea%d'%i:X[:,i] for i in range(X.shape[1])})
    print('%s data'%data_source,X.shape)
    test_tsvd_helper(X,n_components,algorithm,random_state,threshold,use_assert,test_model)

@pytest.mark.skip(reason="helper function, don't test")
def test_tsvd_helper(X,n_components,algorithm,random_state,threshold,use_assert,test_model):
    tsvd_imp1 = run_tsvd(X,
        n_components,algorithm,random_state,model='sklearn')
    print()
    if test_model == 'cuml':
        X = pd2pygdf(X)
    elif test_model == 'h2o4gpu':
        X = np.array(X).astype(np.float32)
    tsvd_imp2 = run_tsvd(X,
        n_components,algorithm,random_state,model=test_model)
    print()
    for attr in ['singular_values_','components_','explained_variance_','explained_variance_ratio_','transformed_result']:
        with_sign = False if attr in ['components_','transformed_result'] else True
        passed = array_equal(getattr(tsvd_imp1,attr),getattr(tsvd_imp2,attr),
            threshold,with_sign=with_sign)
        message = 'compare tsvd: %s vs sklearn %s %s'%(test_model,attr,'equal' if passed else 'NOT equal')
        print(message)
        write_log(message)
        if use_assert:
            assert passed,message
    print()
    del tsvd_imp1,tsvd_imp2,X

@timer
def run_tsvd(X,n_components,algorithm,random_state,model):
    if model == 'sklearn':
        tsvd = skTSVD(n_components=n_components,
            algorithm=algorithm,  random_state=random_state)
    elif model == 'h2o4gpu':
        from h2o4gpu.solvers import TruncatedSVDH2O as h2oTSVD
        if algorithm == 'arpack':
            algorithm = 'cusolver'
        tsvd = h2oTSVD(n_components=n_components,
            algorithm=algorithm,  random_state=random_state)
    elif model == 'cuml':
        from cuSKL import TruncatedSVD as cumlTSVD
        tsvd = cumlTSVD(n_components=n_components,
              random_state=random_state)
    else:
        raise NotImplementedError

    @timer
    def fit_(tsvd,X,model):
        tsvd.fit(X)
        return tsvd
    @timer
    def transform_(tsvd,X,model):
        return tsvd.transform(X)

    tsvd = fit_(tsvd,X,model=model)
    Xtsvd = transform_(tsvd,X,model=model)
    tsvd.transformed_result = lambda: None
    setattr(tsvd,'transformed_result',Xtsvd)
    return tsvd


if __name__ == '__main__':
    args = parse_args()
    write_log(args)
    test_tsvd_mortgage(data_source=args.data,use_assert=args.use_assert,nrows=args.nrows,
        ncols=args.ncols,quarters=args.quarters,random_state=args.random_state,
        test_model=args.test_model,threshold=args.threshold
        )
