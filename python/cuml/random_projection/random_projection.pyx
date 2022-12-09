#
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

# distutils: language = c++

import numpy as np

from libc.stdint cimport uintptr_t
from libcpp cimport bool

import cuml.internals
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from pylibraft.common.handle cimport *
from cuml.common import input_to_cuml_array
from cuml.internals.mixins import FMajorInputTagMixin

from rmm._lib.cuda_stream_view cimport cuda_stream_view

cdef extern from "cuml/random_projection/rproj_c.h" namespace "ML":

    # Structure holding random projection hyperparameters
    cdef struct paramsRPROJ:
        int n_samples           # number of samples
        int n_features          # number of features (original dimension)
        int n_components        # number of components (target dimension)
        double eps              # error tolerance according to Johnson-Lindenstrauss lemma # noqa E501
        bool gaussian_method    # toggle Gaussian or Sparse random projection methods # noqa E501
        double density		    # ratio of non-zero component in the random projection matrix (used for sparse random projection) # noqa E501
        bool dense_output       # toggle random projection's transformation as a dense or sparse matrix # noqa E501
        int random_state        # seed used by random generator

    # Structure describing random matrix
    cdef cppclass rand_mat[T]:
        rand_mat(cuda_stream_view stream) except +     # random matrix structure constructor (set all to nullptr) # noqa E501
        T *dense_data           # dense random matrix data
        int *indices            # sparse CSC random matrix indices
        int *indptr             # sparse CSC random matrix indptr
        T *sparse_data          # sparse CSC random matrix data
        size_t sparse_data_size # sparse CSC random matrix number of non-zero elements # noqa E501

    # Function used to fit the model
    cdef void RPROJfit[T](const handle_t& handle, rand_mat[T] *random_matrix,
                          paramsRPROJ* params) except +

    # Function used to apply data transformation
    cdef void RPROJtransform[T](const handle_t& handle, T *input,
                                rand_mat[T] *random_matrix, T *output,
                                paramsRPROJ* params) except +

    # Function used to compute the Johnson Lindenstrauss minimal distance
    cdef size_t c_johnson_lindenstrauss_min_dim \
        "ML::johnson_lindenstrauss_min_dim" (size_t n_samples,
                                             double eps) except +


def johnson_lindenstrauss_min_dim(n_samples, eps=0.1):
    """
    In mathematics, the Johnson–Lindenstrauss lemma states that
    high-dimensional data can be embedded into lower dimension while preserving
    the distances.

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

    Random projection is a dimensionality reduction technique. Random
    projection methods are powerful methods known for their simplicity,
    computational efficiency and restricted model size.
    This algorithm also has the advantage to preserve distances well between
    any two samples and is thus suitable for methods having this requirement.

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
        Seed used to initialize random generator

    Attributes
    ----------
        params : Cython structure
            Structure holding model's hyperparameters

        rand_matS/rand_matD : Cython pointers to structures
            Structures holding pointers to data describing random matrix.
            S for single/float and D for double.

    Notes
    ------
        Inspired from sklearn's implementation :
        https://scikit-learn.org/stable/modules/random_projection.html

    """

    cdef paramsRPROJ params
    cdef rand_mat[float]* rand_matS
    cdef rand_mat[double]* rand_matD

    def __dealloc__(self):
        del self.rand_matS
        del self.rand_matD

    def __init__(self, *, bool gaussian_method, double density,
                 n_components='auto', eps=0.1, dense_output=True,
                 random_state=None):

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        cdef cuda_stream_view stream = handle_.get_stream()
        self.rand_matS = new rand_mat[float](stream)
        self.rand_matD = new rand_mat[double](stream)

        self.params.n_components = n_components if n_components != 'auto'\
            else -1
        self.params.eps = eps
        self.params.dense_output = dense_output
        if random_state is not None:
            self.params.random_state = random_state

        self.params.gaussian_method = gaussian_method
        self.params.density = density

    @property
    def n_components(self):
        return self.params.n_components

    @n_components.setter
    def n_components(self, value):
        self.params.n_components = value

    @property
    def eps(self):
        return self.params.eps

    @eps.setter
    def eps(self, value):
        self.params.eps = value

    @property
    def dense_output(self):
        return self.params.dense_output

    @dense_output.setter
    def dense_output(self, value):
        self.params.dense_output = value

    @property
    def random_state(self):
        return self.params.random_state

    @random_state.setter
    def random_state(self, value):
        self.params.random_state = value

    @property
    def gaussian_method(self):
        return self.params.gaussian_method

    @gaussian_method.setter
    def gaussian_method(self, value):
        self.params.gaussian_method = value

    @property
    def density(self):
        return self.params.density

    @density.setter
    def density(self, value):
        self.params.density = value

    @cuml.internals.api_base_return_any()
    def fit(self, X, y=None):
        """
        Fit the model. This function generates the random matrix on GPU.

        Parameters
        ----------
            X : array-like (device or host) shape = (n_samples, n_features)
                Used to provide shape information.
                Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
                ndarray, cuda array interface compliant array like CuPy

        Returns
        -------
            The transformer itself with deducted 'auto' parameters and
            generated random matrix as attributes

        """
        _, n_samples, n_features, self.dtype = \
            input_to_cuml_array(X, check_dtype=[np.float32, np.float64])

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        self.params.n_samples = n_samples
        self.params.n_features = n_features

        if self.dtype == np.float32:
            RPROJfit[float](handle_[0], self.rand_matS, &self.params)
        else:
            RPROJfit[double](handle_[0], self.rand_matD, &self.params)

        self.handle.sync()

        return self

    @cuml.internals.api_base_return_array()
    def transform(self, X, convert_dtype=True):
        """
        Apply transformation on provided data. This function outputs
        a multiplication between the input matrix and the generated random
        matrix

        Parameters
        ----------
            X : array-like (device or host) shape = (n_samples, n_features)
                Dense matrix (floats or doubles) of shape (n_samples,
                n_features).
                Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
                ndarray, cuda array interface compliant array like CuPy
            convert_dtype : bool, optional (default = True)
                When set to True, the fit method will, when necessary, convert
                y to be the same data type as X if they differ. This will
                increase memory used for the method.

        Returns
        -------
            The output projected matrix of shape (n_samples, n_components)
            Result of multiplication between input matrix and random matrix

        """
        X_m, n_samples, n_features, dtype = \
            input_to_cuml_array(X, check_dtype=self.dtype,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None))
        cdef uintptr_t input_ptr = X_m.ptr

        X_new = CumlArray.empty((n_samples, self.params.n_components),
                                dtype=self.dtype,
                                order='F',
                                index=X_m.index)

        cdef uintptr_t output_ptr = X_new.ptr

        if self.params.n_features != n_features:
            raise ValueError("n_features must be same as on fitting: %d" %
                             self.params.n_features)

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        if dtype == np.float32:
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

        return X_new

    @cuml.internals.api_base_return_array(get_output_type=False)
    def fit_transform(self, X, convert_dtype=True):
        return self.fit(X).transform(X, convert_dtype)


class GaussianRandomProjection(Base,
                               BaseRandomProjection,
                               FMajorInputTagMixin):
    """
    Gaussian Random Projection method derivated from BaseRandomProjection
    class.

    Random projection is a dimensionality reduction technique. Random
    projection methods are powerful methods known for their simplicity,
    computational efficiency and restricted model size.
    This algorithm also has the advantage to preserve distances well between
    any two samples and is thus suitable for methods having this requirement.

    The components of the random matrix are drawn from N(0, 1 / n_components).

    Examples
    --------

    .. code-block:: python

        from cuml.random_projection import GaussianRandomProjection
        from sklearn.datasets import make_blobs
        from sklearn.svm import SVC

        # dataset generation
        data, target = make_blobs(n_samples=800, centers=400, n_features=3000,
                                  random_state=42)

        # model fitting
        model = GaussianRandomProjection(n_components=5,
                                         random_state=42).fit(data)

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
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.

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
        Seed used to initialize random generator
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    gaussian_method : boolean
        To be passed to base class in order to determine
        random matrix generation method

    Notes
    -----
    This class is unable to be used with ``sklearn.base.clone()`` and will
    raise an exception when called.

    Inspired by Scikit-learn's implementation :
    https://scikit-learn.org/stable/modules/random_projection.html

    """

    def __init__(self, *, handle=None, n_components='auto', eps=0.1,
                 random_state=None, verbose=False, output_type=None):

        Base.__init__(self,
                      handle=handle,
                      verbose=verbose,
                      output_type=output_type)

        BaseRandomProjection.__init__(
            self,
            gaussian_method=True,
            density=-1.0,
            n_components=n_components,
            eps=eps,
            dense_output=True,
            random_state=random_state)

    def get_param_names(self):
        return Base.get_param_names(self) + [
            "n_components",
            "eps",
            "random_state"
        ]


class SparseRandomProjection(Base,
                             BaseRandomProjection,
                             FMajorInputTagMixin):
    """
    Sparse Random Projection method derivated from BaseRandomProjection class.

    Random projection is a dimensionality reduction technique. Random
    projection methods are powerful methods known for their simplicity,
    computational efficiency and restricted model size.
    This algorithm also has the advantage to preserve distances well between
    any two samples and is thus suitable for methods having this requirement.

    Sparse random matrix is an alternative to dense random projection matrix
    (e.g. Gaussian) that guarantees similar embedding quality while being much
    more memory efficient and allowing faster computation of the projected data
    (with sparse enough matrices).
    If we note ``s = 1 / density`` the components of the random matrix are
    drawn from:

    - ``-sqrt(s) / sqrt(n_components)`` - with probability ``1 / 2s``
    - ``0`` - with probability ``1 - 1 / s``
    - ``+sqrt(s) / sqrt(n_components)`` - with probability ``1 / 2s``

    Examples
    --------

    .. code-block:: python

        from cuml.random_projection import SparseRandomProjection
        from sklearn.datasets import make_blobs
        from sklearn.svm import SVC

        # dataset generation
        data, target = make_blobs(n_samples=800, centers=400, n_features=3000,
                                  random_state=42)

        # model fitting
        model = SparseRandomProjection(n_components=5,
                                       random_state=42).fit(data)

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
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.

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
        Seed used to initialize random generator

    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    gaussian_method : boolean
        To be passed to base class in order to determine
        random matrix generation method

    Notes
    -----
    This class is unable to be used with ``sklearn.base.clone()`` and will
    raise an exception when called.

    Inspired by Scikit-learn's `implementation
    <https://scikit-learn.org/stable/modules/random_projection.html>`_.

    """

    def __init__(self, *, handle=None, n_components='auto', density='auto',
                 eps=0.1, dense_output=True, random_state=None,
                 verbose=False, output_type=None):

        Base.__init__(self,
                      handle=handle,
                      verbose=verbose,
                      output_type=output_type)

        BaseRandomProjection.__init__(
            self,
            gaussian_method=False,
            density=(density if density != 'auto' else -1.0),
            n_components=n_components,
            eps=eps,
            dense_output=dense_output,
            random_state=random_state)

    def get_param_names(self):
        return Base.get_param_names(self) + [
            "n_components",
            "density",
            "eps",
            "dense_output",
            "random_state"
        ]
