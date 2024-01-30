#!/usr/bin/python3
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
import argparse
from sklearn.datasets import make_blobs

parser = argparse.ArgumentParser('gen_dataset.py ')

parser.add_argument('-ns', '--num_samples', type=int, default=10000,
                    help='Number of samples (default 10000)')
parser.add_argument('-nf', '--num_features', type=int, default=25,
                    help='Number of features (default 25)')
parser.add_argument('-nc', '--num_clusters', type=int, default=15,
                    help='Number of clusters (default 15)')
parser.add_argument('--filename_prefix', type=str, default='synthetic',
                    help='Prefix used for output  file (default synthetic)')
parser.add_argument('-sd', '--standard_dev', type=str, default=0.1,
                    help='Standard deviation of samples generated')
parser.add_argument('-st', '--random_state', type=str, default=123456,
                    help='Standard deviation of samples generated')

args = parser.parse_args()

datasetFile = '%s-%dx%d-clusters-%d.txt' \
              % (args.filename_prefix, args.num_samples, args.num_features,
                 args.num_clusters)

X, _ = make_blobs(n_samples=args.num_samples, n_features=args.num_features,
                  centers=args.num_clusters, cluster_std=args.standard_dev,
                  random_state=args.random_state)


fp = open(datasetFile, 'w')
for row in range(args.num_samples):
    for col in range(args.num_features):
        fp.write('%f\n' % X[row, col])
fp.close()

print('Dataset file: %s' % datasetFile)
print('Generated total %d samples with %d features each' % (args.num_samples,
                                                            args.num_features))
print('Number of clusters = %d' % args.num_clusters)
