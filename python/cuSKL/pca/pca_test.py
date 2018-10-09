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

from cuML import PCA
import pygdf
import numpy as np

print("\n***********TESTING FOR FLOAT DATATYPE***********")

gdf_float = pygdf.DataFrame()
gdf_float['0']=np.asarray([1.0,2.0,5.0],dtype=np.float32)
gdf_float['1']=np.asarray([4.0,2.0,1.0],dtype=np.float32)
gdf_float['2']=np.asarray([4.0,2.0,1.0],dtype=np.float32)
print("\ninput:")
print(gdf_float)

print("Calling fit")
pca_float = PCA(n_components = 2)
pca_float.fit(gdf_float)

print("\ncomponents:")
print(pca_float.components_)
print("\nexplained variance:")
print(pca_float.explained_variance_)
print("\nexplained variance ratio:")
print(pca_float.explained_variance_ratio_)
print("\nsingular values:")
print(pca_float.singular_values_)
print("\nmean:")
print(pca_float.mean_)
print("\nnoise variance:")
print(pca_float.noise_variance_)

print("Calling transform")
trans_gdf_float = pca_float.transform(gdf_float)

print("\nTransformed matrix")
print(trans_gdf_float)

print("\nCalling inverse_transform")
print("\nInput Matrix:")
input_gdf_float = pca_float.inverse_transform(trans_gdf_float)
print(input_gdf_float)
 
print("\n***********TESTING FOR DOUBLE DATATYPE***********")

gdf_double = pygdf.DataFrame()
gdf_double['0']=np.asarray([1.0,2.0,5.0],dtype=np.float64)
gdf_double['1']=np.asarray([4.0,2.0,1.0],dtype=np.float64)
gdf_double['2']=np.asarray([4.0,2.0,1.0],dtype=np.float64)
print("\ninput:")
print(gdf_double)

print("Calling fit_transform")
pca_double = PCA(n_components = 2)
trans_gdf_double = pca_double.fit_transform(gdf_double)
#print(trans_gdf_double)

print("\ncomponents:")
print(pca_double.components_)
print("\nexplained variance:")
print(pca_double.explained_variance_)
print("\nexplained variance ratio:")
print(pca_double.explained_variance_ratio_)
print("\nsingular values:")
print(pca_double.singular_values_)
print("\nmean:")
print(pca_double.mean_)
print("\nnoise variance:")
print(pca_double.noise_variance_)

print("\nTransformed matrix")
print(trans_gdf_double)
print("\nCalling inverse_transform")
print("\nInput Matrix:")
input_gdf_double = pca_double.inverse_transform(trans_gdf_double)

print(input_gdf_double)

