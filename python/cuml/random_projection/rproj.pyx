#
# Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import cudf
import numpy as np

from numba import cuda

from libc.stdint cimport uintptr_t
from libcpp cimport bool

from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle

cdef extern from "random_projection/rproj_c.h" namespace "ML":

    # Structure holding random projection hyperparameters
    cdef struct paramsRPROJ:
        int n_samples           # number of samples
        int n_features          # number of features (original dimension)
        int n_components        # number of components (target dimension)
        double eps              # error tolerance according to Johnson-Lindenstrauss lemma
        bool gaussian_method    # toggle Gaussian or Sparse random projection methods
        double density		    # ratio of non-zero component in the random projection matrix (used for sparse random projection)
        bool dense_output       # toggle random projection's transformation as a dense or sparse matrix
        int random_state        # seed used by random generator

    # Structure describing random matrix
    cdef cppclass rand_mat[T]:
        rand_mat() except +     # random matrix structure constructor (set all to nullptr)
        T *dense_data           # dense random matrix data
        int *indices            # sparse CSC random matrix indices
        int *indptr             # sparse CSC random matrix indptr
        T *sparse_data          # sparse CSC random matrix data
        size_t sparse_data_size # sparse CSC random matrix number of non-zero elements

    # Function used to fit the model
    cdef void RPROJfit[T](const cumlHandle& handle, rand_mat[T] *random_matrix,
                            paramsRPROJ* params)
    
    # Function used to apply data transformation
    cdef void RPROJtransform[T](const cumlHandle& handle, T *input,
                                rand_mat[T] *random_matrix, T *output,
                                paramsRPROJ* params)

    # Function used to compute the Johnson Lindenstrauss minimal distance
    cdef size_t c_johnson_lindenstrauss_min_dim "ML::johnson_lindenstrauss_min_dim" (size_t n_samples, double eps)


def johnson_lindenstrauss_min_dim(n_samples, eps=0.1):
    """
    In mathematics, the Johnson–Lindenstrauss lemma states that high-dimensional data
    can be embedded into lower dimension while preserving the distances.

    With p the random projection :
    (1 - eps) ||u - v||^2 < ||p(u) - p(v)||^2 < (1 + eps) ||u - v||^2

    This function finds the minimum number of components to guarantee that
    the embedding is inside the eps error tolerance.

    Parameters
    ----------

    n_samples : int
        Number of samples.
    eps : float in (0,1) (default = 0.1)
        Maximum distortion rate as defined by the Johnson-Lindenstrauss lemma.
    
    Returns
    -------

    n_components : int
        The minimal number of components to guarantee with good probability
        an eps-embedding with n_samples.

    """
    return c_johnson_lindenstrauss_min_dim(<size_t>n_samples, <double>eps)

cdef class BaseRandomProjection():
    """
    Base class for random projections.
    This class is not intended to be used directly.
    
    Random projection is a dimensionality reduction technique. Random projection methods
    are powerful methods known for their simplicity, computational efficiency and restricted model size.
    This algorithm also has the advantage to preserve distances well between any two samples
    and is thus suitable for methods having this requirement.

    Parameters
    ----------

    n_components : int (default = 'auto')
        Dimensionality of the target projection space. If set to 'auto',
        the parameter is deducted thanks to Johnson–Lindenstrauss lemma.
        The automatic deduction make use of the number of samples and
        the eps parameter.

        The Johnson–Lindenstrauss lemma can produce very conservative
        n_components parameter as it makes no assumption on dataset structure.

    eps : float (default = 0.1)
        Error tolerance during projection. Used by Johnson–Lindenstrauss
        automatic deduction when n_components is set to 'auto'.

    dense_output : boolean (default = True)
        If set to True transformed matrix will be dense otherwise sparse.

    random_state : int (default = None)
        Seed used to initilize random generator

    Attributes
    ----------
        params : Cython structure
            Structure holding model's hyperparameters

        rand_matS/rand_matD : Cython pointers to structures
            Structures holding pointers to data describing random matrix.
            S for simple/float and D for double.

    Notes
    ------
        Inspired from sklearn's implementation : https://scikit-learn.org/stable/modules/random_projection.html

    """

    cdef paramsRPROJ params
    cdef rand_mat[float]* rand_matS
    cdef rand_mat[double]* rand_matD

    def __cinit__(self):
        self.rand_matS = new rand_mat[float]()
        self.rand_matD = new rand_mat[double]()

    def __dealloc__(self):
        del self.rand_matS
        del self.rand_matD

    def __init__(self, n_components='auto', eps=0.1,
                dense_output=True, random_state=None):
        self.params.n_components = n_components if n_components != 'auto' else -1
        self.params.eps = eps
        self.params.dense_output = dense_output
        if random_state is not None:
            self.params.random_state = random_state

        self.params.gaussian_method = self.gaussian_method
        self.params.density = self.density

    # Gets device pointer from Numba's Cuda array
    def _get_ctype_ptr(self, obj):
        # The manner to access the pointers in the gdf's might change, so
        # encapsulating access in the following 3 methods. They might also be
        # part of future gdf versions.
        return obj.device_ctypes_pointer.value

    # Gets device pointer from cuDF dataframe's column
    def _get_column_ptr(self, obj):
        return self._get_ctype_ptr(obj._column._data.to_gpu_array())

    def fit(self, X, y=None):
        """
        Fit the model. This function generates the random matrix on GPU.

        Parameters
        ----------
            X : cuDF DataFrame or Numpy array
                Dense matrix (floats or doubles) of shape (n_samples, n_features)
                Used to provide shape information

        Returns
        -------
            The transformer itself with deducted 'auto' parameters and
            generated random matrix as attributes

        """
        if isinstance(X, cudf.DataFrame):
            self.gdf_datatype = np.dtype(X[X.columns[0]]._column.dtype)
            n_samples = len(X)
            n_features = len(X._cols)

        elif isinstance(X, np.ndarray):
            self.gdf_datatype = X.dtype
            n_samples, n_features = X.shape

        else:
            msg = "X matrix format not supported"
            raise TypeError(msg)

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()
        self.params.n_samples = n_samples
        self.params.n_features = n_features

        if self.gdf_datatype.type == np.float32:
            RPROJfit[float](handle_[0], self.rand_matS, &self.params)
        else:
            RPROJfit[double](handle_[0], self.rand_matD, &self.params)

        self.handle.sync()

        return self

    def transform(self, X):
        """
        Apply transformation on provided data. This function outputs
        a multiplication between the input matrix and the generated random matrix

        Parameters
        ----------
            X : cuDF DataFrame or Numpy array
                Dense matrix (floats or doubles) of shape (n_samples, n_features)
                Used as input matrix

        Returns
        -------
            The output projected matrix of shape (n_samples, n_components)
            Result of multiplication between input matrix and random matrix

        """

        if isinstance(X, cudf.DataFrame):
            self.gdf_datatype = np.dtype(X[X.columns[0]]._column.dtype)
            X_m = X.as_gpu_matrix()
            n_samples = len(X)
            n_features = len(X._cols)

        elif isinstance(X, np.ndarray):
            self.gdf_datatype = X.dtype
            X_m = cuda.to_device(np.array(X, order='F'))
            n_samples, n_features = X.shape

        else:
            msg = "X matrix format not supported"
            raise TypeError(msg)

        X_new = cuda.device_array((n_samples, self.params.n_components),
                                        dtype=self.gdf_datatype,
                                        order='F')

        cdef uintptr_t input_ptr = self._get_ctype_ptr(X_m)
        cdef uintptr_t output_ptr = self._get_ctype_ptr(X_new)

        if self.params.n_features != n_features:
            raise ValueError("n_features must be same as on fitting: %d" %
                         self.params.n_features)

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        if self.gdf_datatype.type == np.float32:
            RPROJtransform[float](handle_[0],
                        <float*> input_ptr,
                        self.rand_matS,
                        <float*> output_ptr,
                        &self.params)
        else:
            RPROJtransform[double](handle_[0],
                        <double*> input_ptr,
                        self.rand_matD,
                        <double*> output_ptr,
                        &self.params)

        self.handle.sync()

        if (isinstance(X, cudf.DataFrame)):
            del(X_m)
            h_X_new = X_new.copy_to_host()
            del(X_new)
            gdf_X_new = cudf.DataFrame()
            for i in range(0, h_X_new.shape[1]):
                gdf_X_new[str(i)] = h_X_new[:,i]
            return gdf_X_new

        else:
            return X_new.copy_to_host()


class GaussianRandomProjection(Base, BaseRandomProjection):
    """
    Gaussian Random Projection method derivated from BaseRandomProjection class.

    Random projection is a dimensionality reduction technique. Random projection methods
    are powerful methods known for their simplicity, computational efficiency and restricted model size.
    This algorithm also has the advantage to preserve distances well between any two samples
    and is thus suitable for methods having this requirement.

    The components of the random matrix are drawn from N(0, 1 / n_components).

    Example
    ---------

    .. code-block:: python
        from cuml.random_projection import GaussianRandomProjection
        from sklearn.datasets.samples_generator import make_blobs
        from sklearn.svm import SVC

        # dataset generation
        data, target = make_blobs(n_samples=800, centers=400, n_features=3000, random_state=42)

        # model fitting
        model = GaussianRandomProjection(n_components=5, random_state=42).fit(data)

        # dataset transformation
        transformed_data = model.transform(data)

        # classifier training
        classifier = SVC(gamma=0.001).fit(transformed_data, target)

        # classifier scoring
        score = classifier.score(transformed_data, target)

        # measure information preservation
        print("Score: {}".format(score))

    Output:

    .. code-block:: python
        Score: 1.0

    Parameters
    ----------

    handle : cuml.Handle
        If it is None, a new one is created just for this class

    n_components : int (default = 'auto')
        Dimensionality of the target projection space. If set to 'auto',
        the parameter is deducted thanks to Johnson–Lindenstrauss lemma.
        The automatic deduction make use of the number of samples and
        the eps parameter.

        The Johnson–Lindenstrauss lemma can produce very conservative
        n_components parameter as it makes no assumption on dataset structure.

    eps : float (default = 0.1)
        Error tolerance during projection. Used by Johnson–Lindenstrauss
        automatic deduction when n_components is set to 'auto'.

    random_state : int (default = None)
        Seed used to initilize random generator

    Attributes
    ----------
        gaussian_method : boolean
            To be passed to base class in order to determine
            random matrix generation method

    Notes
    ------
        Inspired from sklearn's implementation : https://scikit-learn.org/stable/modules/random_projection.html

    """

    def __init__(self, handle=None, n_components='auto', eps=0.1,
                    random_state=None, verbose=False):
        Base.__init__(self, handle, verbose)
        self.gaussian_method = True
        self.density = -1.0 # not used
        
        BaseRandomProjection.__init__(
            self,
            n_components=n_components,
            eps=eps,
            dense_output=True,
            random_state=random_state)


class SparseRandomProjection(Base, BaseRandomProjection):
    """
    Sparse Random Projection method derivated from BaseRandomProjection class.

    Random projection is a dimensionality reduction technique. Random projection methods
    are powerful methods known for their simplicity, computational efficiency and restricted model size.
    This algorithm also has the advantage to preserve distances well between any two samples
    and is thus suitable for methods having this requirement.

    Sparse random matrix is an alternative to dense random projection matrix (e.g. Gaussian)
    that guarantees similar embedding quality while being much more memory efficient
    and allowing faster computation of the projected data (with sparse enough matrices).
    If we note 's = 1 / density' the components of the random matrix are
    drawn from:
      - -sqrt(s) / sqrt(n_components)   with probability 1 / 2s
      -  0                              with probability 1 - 1 / s
      - +sqrt(s) / sqrt(n_components)   with probability 1 / 2s

    Example
    ---------

    .. code-block:: python
        from cuml.random_projection import SparseRandomProjection
        from sklearn.datasets.samples_generator import make_blobs
        from sklearn.svm import SVC

        # dataset generation
        data, target = make_blobs(n_samples=800, centers=400, n_features=3000, random_state=42)

        # model fitting
        model = SparseRandomProjection(n_components=5, random_state=42).fit(data)

        # dataset transformation
        transformed_data = model.transform(data)

        # classifier training
        classifier = SVC(gamma=0.001).fit(transformed_data, target)

        # classifier scoring
        score = classifier.score(transformed_data, target)

        # measure information preservation
        print("Score: {}".format(score))

    Output:

    .. code-block:: python
        Score: 1.0

    Parameters
    ----------

    handle : cuml.Handle
        If it is None, a new one is created just for this class

    n_components : int (default = 'auto')
        Dimensionality of the target projection space. If set to 'auto',
        the parameter is deducted thanks to Johnson–Lindenstrauss lemma.
        The automatic deduction make use of the number of samples and
        the eps parameter.

        The Johnson–Lindenstrauss lemma can produce very conservative
        n_components parameter as it makes no assumption on dataset structure.

    density : float in range (0, 1] (default = 'auto')
        Ratio of non-zero component in the random projection matrix.

        If density = 'auto', the value is set to the minimum density
        as recommended by Ping Li et al.: 1 / sqrt(n_features).        

    eps : float (default = 0.1)
        Error tolerance during projection. Used by Johnson–Lindenstrauss
        automatic deduction when n_components is set to 'auto'.

    dense_output : boolean (default = True)
        If set to True transformed matrix will be dense otherwise sparse.

    random_state : int (default = None)
        Seed used to initilize random generator

    Attributes
    ----------
        gaussian_method : boolean
            To be passed to base class in order to determine
            random matrix generation method

    Notes
    ------
        Inspired from sklearn's implementation : https://scikit-learn.org/stable/modules/random_projection.html

    """

    def __init__(self, handle=None, n_components='auto', density='auto',
                    eps=0.1, dense_output=True, random_state=None, verbose=False):
        Base.__init__(self, handle, verbose)
        self.gaussian_method = False
        self.density = density if density != 'auto' else -1.0

        BaseRandomProjection.__init__(
            self,
            n_components=n_components,
            eps=eps,
            dense_output=dense_output,
            random_state=random_state)