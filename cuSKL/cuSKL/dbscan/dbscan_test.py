#
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

from cuML import DBSCAN
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
dbscan_float = DBSCAN(eps=1.0, min_samples=1)
dbscan_float.fit(gdf_float)
print(dbscan_float.labels_)

 
print("\n***********TESTING FOR DOUBLE DATATYPE***********")

gdf_double = pygdf.DataFrame()
gdf_double['0']=np.asarray([1.0,2.0,5.0],dtype=np.float64)
gdf_double['1']=np.asarray([4.0,2.0,1.0],dtype=np.float64)
gdf_double['2']=np.asarray([4.0,2.0,1.0],dtype=np.float64)

print("\ninput:")
print(gdf_double)

print("Calling fit_transform")
dbscan_double = DBSCAN(eps=1.0, min_samples=1)
dbscan_double.fit(gdf_double)
print(dbscan_double.labels_)

