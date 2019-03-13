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