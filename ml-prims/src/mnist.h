/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <stdio.h>
#include <zlib.h>
#include <stdint.h>
#include "cuda_utils.h"


namespace MLCommon {
namespace mnist {

/** Class to parse and load the mnist dataset into working memory */
struct Dataset {
    /** number of images in the dataset */
    int nImages;
    /** image resolution */
    int nRows, nCols;
    /** total chars in the dataset */
    size_t len;
    /** the dataset */
    uint8_t* data;
    /** labels */
    uint8_t* label;

    /** Read the dataset and labels from the "images" and "labels" files */
    Dataset(const char* images, const char* labels):
        nImages(0), nRows(0), nCols(0), len(0), data(nullptr), label(nullptr) {
        gzFile ifp = gzopen(images, "rb");
        ASSERT(ifp != nullptr, "Mnist: Failed to read images file '%s'", images);
        readAndReverse(ifp); // magic number
        nImages = readAndReverse(ifp);
        nRows = readAndReverse(ifp);
        nCols = readAndReverse(ifp);
        len = nImages * nRows * nCols;
        data = new uint8_t[len];
        int count = gzread(ifp, data, sizeof(uint8_t)*len);
        ASSERT(count == (int)len, "Mnist: images read=%d expected=%d",
               count, (int)len);
        label = new uint8_t[nImages];
        gzclose(ifp);
        gzFile lfp = gzopen(labels, "rb");
        ASSERT(lfp != nullptr, "Mnist: Failed to read labels file '%s'", labels);
        readAndReverse(lfp); // magic number
        readAndReverse(lfp); // num items
        count = gzread(lfp, label, sizeof(uint8_t)*nImages);
        ASSERT(count == nImages, "Mnist: labels read=%d expected=%d",
               count, nImages);
        gzclose(lfp);
    }

    /** Dtor */
    ~Dataset() {
        delete [] data;
        delete [] label;
    }

    /**
     * convert char images into float numbers in range [0.0, 1.0]
     * @note: It's the responsibility of caller to delete [] the out array
     */
    float* toFloatImage() const {
        float* out = new float[len];
        for(size_t i=0;i<len;++i) {
            out[i] = (float)data[i] / 255.f;
        }
        return out;
    }

    /**
     * convert char images into device float array in range [0.0, 1.0]
     * @note: It's the responsibility of caller to cudaFree the output array!
     */
    float* toFloatImageDevice() const {
        float* out = toFloatImage();
        float* dOut;
        CUDA_CHECK(cudaMalloc((void**)&dOut, sizeof(float)*len));
        updateDevice(dOut, out, len);
        delete [] out;
        return dOut;
    }

    /** convert labels to float as well */
    float* toFloatLabel() const {
        float* out = new float[nImages];
        for(size_t i=0;i<nImages;++i) {
            out[i] = (float)label[i];
        }
        return out;
    }

private:
    int readAndReverse(gzFile& fp) {
        int val;
        int count = gzread(fp, &val, sizeof(int));
        ASSERT(count == (int)sizeof(int),
               "Mnist: readAndReverse: couldn't read int count=%d", count);
        return reverse(val);
    }

    unsigned reverse(unsigned val) {
        unsigned out = 0;
        for(int i=0;i<4;++i,val>>=8) {
            uint8_t byte = val & 0xff;
            out <<= 8;
            out |= byte;
        }
        return out;
    }
};

} // end namespace mnist
} // end namespace MLCommon
