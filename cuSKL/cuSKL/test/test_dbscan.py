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
from sklearn.cluster import DBSCAN as skDBSCAN
from test_utils import timer,load_mortgage,pd2pygdf,array_equal,parse_args,write_log
import pytest

def test_dbscan(nrows=1000,ncols=512, eps = 3, min_samples = 2,
        threshold=1e-3,data_source = 'mortgage',use_assert=True,
        quarters=8,rows_per_quarter=100000,test_model='cuml'):
    print()
    #X = np.random.rand(nrows,ncols)
    X = np.array([[1, 2], [2, 2], [2, 3],[8, 7], [8, 8], [25, 80]],dtype='float64')
    #if data_source=='random':
        #X = np.random.rand(nrows,ncols)
    #elif data_source=='mortgage':
        #X = load_mortgage(quarters=quarters,rows_per_quarter=rows_per_quarter)
        #X = X[np.random.randint(0,X.shape[0]-1,nrows),:ncols]
    #else:
        #raise NotImplementedError        
    X = pd.DataFrame({'fea%d'%i:X[:,i] for i in range(X.shape[1])})
    print('%s data'%data_source,X.shape)
    test_dbscan_helper(X,eps,min_samples,threshold,use_assert,test_model)

@pytest.mark.skip(reason="helper function, don't test")    
def test_dbscan_helper(X, eps, min_samples, threshold, use_assert, test_model):
    dbscan_imp1 = run_dbscan(X,
        eps, min_samples, model='sklearn')
    print()
    if test_model == 'cuml':
        X = pd2pygdf(X)

    dbscan_imp2 = run_dbscan(X,
        eps, min_samples, model=test_model)  
    print()
    for attr in ['labels_']:
        passed = array_equal(getattr(dbscan_imp1,attr),getattr(dbscan_imp2,attr),
            threshold,with_sign = True)
        message = 'compare pca: %s vs sklearn %s %s'%(test_model,attr,'equal' if passed else 'NOT equal')
        print(message)
        write_log(message)
        if use_assert:
            assert passed,message
    print()
    del dbscan_imp1,dbscan_imp2,X

@timer
def run_dbscan(X,eps,min_samples,model):
    if model == 'sklearn':
        clustering = skDBSCAN(eps = eps, min_samples = min_samples)
    elif model == 'cuml':
        from cuML import DBSCAN as cumlDBSCAN
        clustering = cumlDBSCAN(eps = eps, min_samples = min_samples)
    else:
        raise NotImplementedError

    @timer
    def fit_(clustering,X,model):
        clustering.fit(X)
        return clustering
     
    #@timer
    #def transform_(pca,X,model):
        #return pca.transform(X)

    clustering = fit_(clustering,X,model=model)
    print(clustering.labels_)
    #Xpca = transform_(pca,X,model=model)
    #pca.transformed_result = lambda: None
    #setattr(pca,'transformed_result',Xpca)
    return clustering


if __name__ == '__main__':
    args = parse_args()
    write_log(args)
    test_dbscan(data_source=args.data,use_assert=False,nrows=args.nrows,
        ncols=args.ncols,quarters=args.quarters,
        test_model=args.test_model,threshold=args.threshold
        )

