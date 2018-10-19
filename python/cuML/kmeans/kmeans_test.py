from cuML import KMeans
import pygdf
import numpy as np
import pandas as pd
print("\n***********TESTING FOR FLOAT DATATYPE***********")

#gdf_float = pygdf.DataFrame()
#gdf_float['x']=np.asarray([1.0,1.0,3.0,4.0],dtype=np.float32)
#gdf_float['y']=np.asarray([1.0,2.0,2.0,3.0],dtype=np.float32)

def np2pygdf(df):
    # convert numpy array to pygdf dataframe 
    df = pd.DataFrame({'fea%d'%i:df[:,i] for i in range(df.shape[1])})
    pdf = pygdf.DataFrame()
    for c,column in enumerate(df):
        pdf[str(c)] = df[column]
    return pdf

y=np.asarray([[1.0,2.0],[1.0,4.0],[1.0,0.0],[4.0,2.0],[4.0,4.0],[4.0,0.0]],dtype=np.float32)
x=np2pygdf(y)
q=np.asarray([[0, 0], [4, 4]],dtype=np.float32)
p=np2pygdf(q)
a=np.asarray([[1.0, 1.0], [1.0, 2.0], [3.0, 2.0], [4.0, 3.0]],dtype=np.float32)
b=np2pygdf(a)
print("input:")
print(b)

print("\nCalling fit")
kmeans_float = KMeans(n_clusters=2, n_gpu=1)
kmeans_float.fit(b)
print("labels:")
print(kmeans_float.labels_)
print("cluster_centers:")
print(kmeans_float.cluster_centers_)

'''
print("\nCalling Predict")
print("labels:")
print(kmeans_float.predict(p))
print("cluster_centers:")
print(kmeans_float.cluster_centers_)
'''


print("\nCalling fit_predict")
kmeans_float2 = KMeans(n_clusters=2, n_gpu=1)
print("labels:")
print(kmeans_float2.fit_predict(b))
print("cluster_centers:")
print(kmeans_float2.cluster_centers_)


print("\nCalling transform")
print("\ntransform result:")
print(kmeans_float2.transform(b))
