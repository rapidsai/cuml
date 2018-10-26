import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree as skKNN
from cuML import KNN
import pygdf
import os

def test_knn():
    np_float = np.array([
      [1.,2.,3.], # 1st point 
      [1.,2.,4.], # 2nd point
      [2.,2.,4.]  # 3rd point
    ]).astype('float32')
    gdf_float = pygdf.DataFrame()
    gdf_float['dim_0'] = np.ascontiguousarray(np_float[:,0])
    gdf_float['dim_1'] = np.ascontiguousarray(np_float[:,1])
    gdf_float['dim_2'] = np.ascontiguousarray(np_float[:,2])

    knn_float = KNN(n_gpus=1)
    knn_float.fit(gdf_float)
    Distance,Index = knn_float.query(k=3) #get 3 nearest neighbors
