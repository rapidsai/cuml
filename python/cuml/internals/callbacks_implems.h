/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <Python.h>
#include "internals/internals.h"

#include <iostream>

namespace ML {
    namespace  Internals {

        class DefaultGraphBasedDimRedCallback : public GraphBasedDimRedCallback {
            public:
                void on_preprocess_end(void* embeddings) override
                {
                    PyObject* pycl = (PyObject*)this->pyCallbackClass;
                    PyObject* numba_matrix = PyObject_CallMethod(pycl,
                        "get_numba_matrix", "(l(ll))", (long)embeddings,
                        (long)n, (long)n_components);
                    PyObject* res = PyObject_CallMethod(pycl, "on_preprocess_end",
                                        "(O)", numba_matrix);
                    Py_DECREF(numba_matrix);
                    Py_DECREF(res);
                }

                void on_epoch_end(void* embeddings) override
                {
                    PyObject* pycl = (PyObject*)this->pyCallbackClass;
                    PyObject* numba_matrix = PyObject_CallMethod(pycl,
                        "get_numba_matrix", "(l(ll))", (long)embeddings,
                        (long)n, (long)n_components);
                    PyObject* res = PyObject_CallMethod(pycl, "on_epoch_end",
                                        "(O)", numba_matrix);
                    Py_DECREF(numba_matrix);
                    Py_DECREF(res);
                }

                void on_train_end(void* embeddings) override
                {
                    PyObject* pycl = (PyObject*)this->pyCallbackClass;
                    PyObject* numba_matrix = PyObject_CallMethod(pycl,
                        "get_numba_matrix", "(l(ll))", (long)embeddings,
                        (long)n, (long)n_components);
                    PyObject* res = PyObject_CallMethod(pycl, "on_train_end",
                                        "(O)", numba_matrix);
                    Py_DECREF(numba_matrix);
                    Py_DECREF(res);
                }

            public:
                void* pyCallbackClass;
        };

    }
}