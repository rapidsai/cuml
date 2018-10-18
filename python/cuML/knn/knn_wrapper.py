import faiss
import numpy as np
import pandas as pd
import pygdf

class KNNparams:
    def __init__(self,n_gpus):
        self.n_gpus = n_gpus

class KNN:

    def __init__(self, n_gpus=-1): # -1 means using all gpus
        self.params = KNNparams(n_gpus)

    def fit(self,X):
        X = self.to_nparray(X)
        assert len(X.shape)==2, 'data should be two dimensional'
        n_dims = X.shape[1]
        cpu_index = faiss.IndexFlatL2(n_dims) # build a flat (CPU) index
        if self.params.n_gpus==1:
            res = faiss.StandardGpuResources()  # use a single GPU
            # make it a flat GPU index
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            gpu_index = faiss.index_cpu_to_all_gpus(cpu_index,ngpu=self.params.n_gpus)
        gpu_index.add(X)
        self.gpu_index = gpu_index

    def query(self,X,k):
        X = self.to_nparray(X)
        D,I = self.gpu_index.search(X, k)
        D = self.to_pygdf(D,col='distance')
        I = self.to_pygdf(I,col='index')
        return D,I

    def to_nparray(self,x):
        if isinstance(x,pd.DataFrame):
            x = x.values
        elif isinstance(x,pygdf.DataFrame):
            x = x.to_pandas().values
        return np.ascontiguousarray(x)

    def to_pygdf(self,df,col=''):
        # convert pandas dataframe to pygdf dataframe
        if isinstance(df,np.ndarray):
            df = pd.DataFrame({'%s_neighbor_%d'%(col,i):df[:,i] for i in range(df.shape[1])})
        pdf = pygdf.DataFrame.from_pandas(df)
        return pdf
