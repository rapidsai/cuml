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
from sklearn.decomposition import PCA as skPCA
from test_utils import timer,load_mortgage,pd2pygdf,array_equal,parse_args,write_log
import pytest

def test_pca(nrows=1000,ncols=512,n_components=10,
        svd_solver='full',whiten=False,random_state=42,
        threshold=1e-3,data_source = 'mortgage',use_assert=True,
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
    test_pca_helper(X,n_components,svd_solver,whiten,random_state,threshold,use_assert,test_model)

@pytest.mark.skip(reason="helper function, don't test")    
def test_pca_helper(X,n_components,svd_solver,whiten,random_state,threshold,use_assert,test_model):
    pca_imp1 = run_pca(X,
        n_components,svd_solver,whiten,random_state,model='sklearn')
    print()
    if test_model == 'cuml':
        X = pd2pygdf(X)
    elif test_model == 'h2o4gpu':
        X = np.array(X).astype(np.float32)

    pca_imp2 = run_pca(X,
        n_components,svd_solver,whiten,random_state,model=test_model)  
    print()
    for attr in ['singular_values_','components_','explained_variance_','explained_variance_ratio_','noise_variance_','transformed_result']:
        with_sign = False if attr in ['components_','transformed_result'] else True
        passed = array_equal(getattr(pca_imp1,attr),getattr(pca_imp2,attr),
            threshold,with_sign=with_sign)
        message = 'compare pca: %s vs sklearn %s %s'%(test_model,attr,'equal' if passed else 'NOT equal')
        print(message)
        write_log(message)
        if use_assert:
            assert passed,message
    print()
    del pca_imp1,pca_imp2,X

@timer
def run_pca(X,n_components,svd_solver,whiten,random_state,model):
    if model == 'sklearn':
        pca = skPCA(n_components=n_components, 
            svd_solver=svd_solver, whiten=whiten, random_state=random_state)
    elif model == 'h2o4gpu':
        from h2o4gpu.solvers.pca import PCAH2O as h2oPCA
        pca = h2oPCA(n_components=n_components,
            whiten=whiten)#, random_state=random_state)
    elif model == 'cuml':
        from cuML import PCA as cumlPCA
        pca = cumlPCA(n_components=n_components,
            svd_solver=svd_solver, whiten=whiten, random_state=random_state)
    else:
        raise NotImplementedError

    @timer
    def fit_(pca,X,model):
        pca.fit(X)
        return pca
    @timer
    def transform_(pca,X,model):
        return pca.transform(X)

    pca = fit_(pca,X,model=model)
    Xpca = transform_(pca,X,model=model)
    pca.transformed_result = lambda: None
    setattr(pca,'transformed_result',Xpca)
    return pca


if __name__ == '__main__':
    args = parse_args()
    write_log(args)
    test_pca(data_source=args.data,use_assert=args.use_assert,nrows=args.nrows,
        ncols=args.ncols,quarters=args.quarters,random_state=args.random_state,
        test_model=args.test_model,threshold=args.threshold
        )
