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
from sklearn.decomposition import PCA as PCA_SKL
import pygdf
import numpy as np
import pandas as pd
import time

m = 3
n = 5
pc = m

rng = np.random.RandomState(1)
data = np.dot(rng.rand(n, n).astype(np.float32), rng.randn(n, m).astype(np.float32))
index = [i for i in range(1, len(data)+1)]

df = pd.DataFrame(data)#, index = index)
df = df.astype('float32')

gdf = pygdf.DataFrame.from_pandas(df)

print("\ninput:")

print("Calling fit_transform")
ts_gpu = time.time()
pca_float = PCA(n_components = pc, whiten=True, svd_solver="jacobi", iterated_power=15, tol=1e-7)
pca_float.fit(gdf)
trans_gdf_float = pca_float.transform(gdf)
gdf_back = pca_float.inverse_transform(trans_gdf_float)
trans_gdf_float=pca_float.transform(gdf_back)
te_gpu = time.time()

pca_float_2 = PCA(n_components = pc, whiten=True, svd_solver="jacobi", iterated_power=15, tol=1e-7)
trans_gdf_float_2 = pca_float_2.fit_transform(gdf)

pca_float_skl = PCA_SKL(n_components = pc, whiten=True)
pca_float_skl.fit(data)
trans_data_skl = pca_float_skl.transform(data)

print("Elapsed time for GPU run = ",te_gpu - ts_gpu, "seconds")


print("\nexplained variance:")
print(pca_float.explained_variance_)
print("\nexplained variance ratio:")
print(pca_float.explained_variance_ratio_)
print("\nsingular values:")
print(pca_float.singular_values_)
print("\nmean:")
print(pca_float.mean_)
print(pca_float_skl.mean_)
print("\nnoise variance:")
print(pca_float.noise_variance_)

print("\nTransformed matrix")
print(trans_gdf_float)
print("\nCalling inverse_transform")
print("\nInput Matrix:")
input_gdf_float = pca_float.inverse_transform(trans_gdf_float)
print(input_gdf_float)


print("\nTransformed matrix")
print(trans_gdf_float)
print("")
print(trans_gdf_float_2)
print("")
print(trans_data_skl)

print("\nComponents")
print(pca_float.components_)
print("")
print(pca_float_2.components_)
print("")
print(pca_float_skl.components_)

