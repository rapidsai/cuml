#!/usr/bin/env python3
#
# Copyright 2017-2018 H2O.ai, Inc.
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
# Based on
# https://github.com/h2oai/h2o4gpu/blob/master/examples/py/demos/H2O4GPU_KMeans_Homesite.ipynb
# as received on December 6th 2018
import pandas as pd
import numpy as np
import sys
import os

# read the data
train_file = "train.csv"
if len(sys.argv) > 1:
    train_file = sys.argv[1]
test_file = "test.csv"
if len(sys.argv) > 2:
    test_file = sys.argv[2]
output_file = "output.txt"
if len(sys.argv) > 3:
    output_file = sys.argv[3]
print("Reading Input from train_file = %s and test_file = %s" % (train_file,
                                                                 test_file))

if not os.path.exists(train_file) or not os.path.exists(test_file):
    raise Exception("Download the dataset from here:"
                    " https://www.kaggle.com/c/homesite-quote-conversion/data")

train = pd.read_csv(train_file)
print("Training dataset dimension: ", train.shape)
test = pd.read_csv(test_file)
print("Test dataset dimension:     ", test.shape)
# Data munging step - KMeans takes only numerical values
train.drop(['QuoteConversion_Flag'], axis=1, inplace=True)
dataset = pd.concat([train, test], ignore_index=True)
tmp = dataset.dtypes.reset_index().rename(columns={0: "type"})
indx = tmp["type"] == "object"
categoricals = tmp[indx]["index"].tolist()
# Replace nans as new category
for col in dataset.columns:
    dataset[col] = dataset[col].fillna("0")
# Encode unfreq categories
for col in categoricals:
    val_dict = dataset[col].value_counts()
    val_dict = dataset[col].value_counts().reset_index()
    indx = val_dict[col] < 100
    res = val_dict[indx]["index"].tolist()
    indx = dataset[col].isin(res)
    vals = dataset[col].values
    vals[indx] = "___UNFREQ___"
    dataset[col] = vals
# Encode all as freqs
for col in categoricals:
    val_dict = dataset[col].value_counts()
    val_dict = val_dict / float(dataset.shape[0])
    val_dict = val_dict.to_dict()
    dataset[col] = dataset[col].apply(lambda x: val_dict[x])
trainenc = dataset.iloc[:train.shape[0], :].reset_index(drop=True)
trainencflt = trainenc.values.astype(np.float32)
print("Output dataset dimension:   ", trainencflt.shape)
output = open(output_file, "w+")
num_items = 0
for row in trainencflt:
    for val in row:
        output.write("%f\n" % val)
        num_items += 1
output.close()
print("Wrote %d values in row major order to output %s" % (num_items,
                                                           output_file))
