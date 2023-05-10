#!/usr/bin/python3
#
# Copyright (c) 2023, NVIDIA CORPORATION.
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
import argparse
from sklearn.datasets import make_blobs
import numpy as np

parser = argparse.ArgumentParser('gen_dataset.py ')

parser.add_argument('-ng', '--num_groups', type=int, default=32,
                    help='Number of groups (default 32)')
parser.add_argument('-ns', '--num_samples', type=int, default=1024,
                    help='Maximum of the numbers of samples (default 1024)')
parser.add_argument('-nf', '--num_features', type=int, default=64,
                    help='Number of features (default 64)')
parser.add_argument('-nc', '--num_clusters', type=int, default=15,
                    help='Number of clusters (default 15)')
parser.add_argument('--filename_prefix', type=str, default='synthetic',
                    help='Prefix used for output  file (default synthetic)')
parser.add_argument('-sd', '--standard_dev', type=str, default=0.1,
                    help='Standard deviation of samples generated')
parser.add_argument('-st', '--random_state', type=str, default=123456,
                    help='Random state of samples generated')

args = parser.parse_args()

datasetFile = '%s-%dx%dx%d-clusters-%d.txt' \
              % (args.filename_prefix, args.num_groups, args.num_samples, 
                 args.num_features, args.num_clusters)

data_list = []
for g in range(args.num_groups):
    random_state = np.random.randint(0, args.random_state * 4) % args.random_state
    X, _ = make_blobs(n_samples=args.num_samples, n_features=args.num_features,
                    centers=args.num_clusters, cluster_std=args.standard_dev,
                    random_state=random_state)
    data_list.append(X)

fp = open(datasetFile, 'w')
for g in range(args.num_groups):
    X = data_list[g]
    for row in range(args.num_samples):
        for col in range(args.num_features):
            fp.write('%f\n' % X[row, col])
fp.close()

print('Dataset file: %s' % datasetFile)
print('Generated total %d groups, and each one contains %d samples with %d features each' \
       % (args.num_groups, args.num_samples, args.num_features))
print('Number of clusters = %d' % args.num_clusters)