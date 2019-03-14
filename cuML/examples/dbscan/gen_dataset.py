#!/usr/bin/python
#
# Copyright (c) 2019, NVIDIA CORPORATION.
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
from sklearn.datasets.samples_generator import make_blobs

nRows = 10000
nCols = 25
nClusters = 15
datasetFile = '../datasets/synthetic-%dx%d-clusters-%d.txt' 
              % (nRows, nCols, nClusters)

X, _ = make_blobs(n_samples=nRows, n_features=nCols, centers=nClusters,
                    cluster_std=0.1, random_state=123456)


fp = open(datasetFile, 'w')
for row in range(nRows):
    for col in range(nCols):
        fp.write('%f\n' %X[row, col])
fp.close()

print 'Dataset file: %s' % datasetFile
print 'Total %d rows (data-points) with %d columns (features)' % (nRows, nCols)
print 'Number of clusters = %d' % nClusters