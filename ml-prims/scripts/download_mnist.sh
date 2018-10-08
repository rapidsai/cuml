#!/bin/bash
# Helper script to download the mnist dataset

function download() {
    local file=$1
    if [ ! -e "$file" ]; then
        wget "http://yann.lecun.com/exdb/mnist/$file"
    fi
}

for file in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
    download $file
done
