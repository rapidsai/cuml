# Copyright (c) 2018, NVIDIA CORPORATION.
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
from numba import cuda


cdef extern from "kalman_filter/kf_variables.h" namespace "kf::linear":
    enum Option:
        LongForm = 0
        ShortFormExplicit
        ShortFormImplicit

    cdef cppclass Variables[T]:
        pass


cdef extern from "kalman_filter/lkf_py.h" namespace "kf::linear" nogil:

    cdef size_t get_workspace_size_f32(Variables[float]&,
                                       int,
                                       int,
                                       Option,
                                       float*,
                                       float*,
                                       float*,
                                       float*,
                                       float*,
                                       float*,
                                       float*,
                                       float*)

    cdef void init_f32(Variables[float]&,
                       int,
                       int,
                       Option,
                       float*,
                       float*,
                       float*,
                       float*,
                       float*,
                       float*,
                       float*,
                       float*,
                       void*,
                       size_t&)

    cdef void predict_f32(Variables[float]&)

    cdef void update_f32(Variables[float]&,
                         float*)

    cdef size_t get_workspace_size_f64(Variables[double]&,
                                       int,
                                       int,
                                       Option,
                                       double*,
                                       double*,
                                       double*,
                                       double*,
                                       double*,
                                       double*,
                                       double*,
                                       double*)

    cdef void init_f64(Variables[double]&,
                       int,
                       int,
                       Option,
                       double*,
                       double*,
                       double*,
                       double*,
                       double*,
                       double*,
                       double*,
                       double*,
                       void*,
                       size_t&)

    cdef void predict_f64(Variables[double]&)

    cdef void update_f64(Variables[double]&,
                         double*)


class KalmanFilter:
    """
    Implements a Kalman filter. You are responsible for setting the
    various state variables to reasonable values; defaults  will
    not give you a functional filter.
    After construction the filter will have default matrices created for you,
    but you must specify the values for each.
    Examples
    --------
    .. code::
        from cuML import KalmanFilter
        f = KalmanFilter(dim_x=2, dim_z=1)
        f.x = np.array([[2.],    # position
                        [0.]])   # velocity
        f.F = np.array([[1.,1.],
                            [0.,1.]])
        f.H = np.array([[1.,0.]])
        f.P = np.array([[1000.,    0.],
                        [   0., 1000.] ])
        f.R = 5

    Now just perform the standard predict/update loop:
    while some_condition_is_true:
    .. code::
        z = get_sensor_reading()
        f.predict()
        f.update(z)

    Parameters
    ----------
    dim_x : int
        Number of state variables for the Kalman filter.
        This is used to set the default size of P, Q, and u
    dim_z : int
        Number of of measurement inputs.

    Attributes
    ----------
    x : numba device array(dim_x, 1)
        Current state estimate. Any call to update() or predict() updates
        this variable.
    P : numba device array(dim_x, dim_x)
        Current state covariance matrix. Any call to update() or predict()
        updates this variable.
    x_prior : numba device array(dim_x, 1)
        Prior (predicted) state estimate. The *_prior and *_post attributes
        are for convienence; they store the  prior and posterior of the
        current epoch. Read Only.
    P_prior : numba device array(dim_x, dim_x)
        Prior (predicted) state covariance matrix. Read Only.
    x_post : numba device array(dim_x, 1)
        Posterior (updated) state estimate. Read Only.
    P_post : numba device array(dim_x, dim_x)
        Posterior (updated) state covariance matrix. Read Only.
    z : numba device array
        Last measurement used in update(). Read only.
    R : numba device array(dim_z, dim_z)
        Measurement noise matrix
    Q : numba device array(dim_x, dim_x)
        Process noise matrix
    F : numba device array()
        State Transition matrix
    H : numba device array(dim_z, dim_x)
        Measurement function
    y : numba device array
        Residual of the update step. Read only.
    K : numba device array(dim_x, dim_z)
        Kalman gain of the update step. Read only.

    """

    def _get_ctype_ptr(self, obj):
        return obj.device_ctypes_pointer.value

    def _get_dtype(self, precision):
        return {
            'single': np.float32,
            'double': np.float64,
        }[precision]

    def __init__(self, dim_x, dim_z, solver='long', precision='single', seed=False):

        if solver in ['long', 'short_implicit', 'short_explicit']:
            self._algorithm = self._get_algorithm_c_name(solver)
        else:
            msg = "algorithm {!r} is not supported"
            raise TypeError(msg.format(solver))

        self.precision = precision
        self.dtype = self._get_dtype(precision)

        Phi = np.ones((dim_x, dim_x), dtype=self.dtype)
        x_up = np.zeros((dim_x, 1), dtype=self.dtype)
        x_est = np.zeros((dim_x, 1), dtype=self.dtype)
        P_up = np.eye(dim_x, dtype=self.dtype)
        P_est = np.eye(dim_x, dtype=self.dtype)
        Q = np.eye(dim_x, dtype=self.dtype)
        H = np.zeros((dim_z, dim_x), dtype=self.dtype)
        R = np.eye(dim_z, dtype=self.dtype)
        z = np.array([[0]*dim_z], dtype=self.dtype).T

        self.F = cuda.to_device(Phi)
        self.x = cuda.to_device(x_up)
        self.x_prev = cuda.to_device(x_est)
        self.P = cuda.to_device(P_up)
        self.P_prev = cuda.to_device(P_est)
        self.Q = cuda.to_device(Q)
        self.H = cuda.to_device(H)
        self.R = cuda.to_device(R)
        self.z = cuda.to_device(z)

        self.dim_x = dim_x
        self.dim_z = dim_z

        self._workspaceSize = -1

        cdef uintptr_t _Phi_ptr = self.F.device_ctypes_pointer.value
        cdef uintptr_t _x_up_ptr = self.x.device_ctypes_pointer.value
        cdef uintptr_t _x_est_ptr = self.x_prev.device_ctypes_pointer.value
        cdef uintptr_t _P_up_ptr = self.P.device_ctypes_pointer.value
        cdef uintptr_t _P_est_ptr = self.P_prev.device_ctypes_pointer.value
        cdef uintptr_t _Q_ptr = self.Q.device_ctypes_pointer.value
        cdef uintptr_t _H_ptr = self.H.device_ctypes_pointer.value
        cdef uintptr_t _R_ptr = self.R.device_ctypes_pointer.value
        cdef uintptr_t _z_ptr = self.z.device_ctypes_pointer.value

        cdef Variables[float] var_ptr
        cdef size_t workspace_size

        cdef int c_dim_x = dim_x
        cdef int c_dim_z = dim_z



        with nogil:

            workspace_size = get_workspace_size_f32(var_ptr,
                                                    <int> c_dim_x,
                                                    <int> c_dim_z,
                                                    <Option> LongForm,
                                                    <float*> _x_est_ptr,
                                                    <float*> _x_up_ptr,
                                                    <float*> _Phi_ptr,
                                                    <float*> _P_est_ptr,
                                                    <float*> _P_up_ptr,
                                                    <float*> _Q_ptr,
                                                    <float*> _R_ptr,
                                                    <float*> _H_ptr)

        self.workspace = cuda.to_device(np.zeros(workspace_size, dtype=self.dtype))
        self._workspace_size = workspace_size
        print(self.workspace)
        print(self._workspace_size)


    def _get_algorithm_c_name(self, algorithm):
        return {
            'long': LongForm,
            'short_implicit': ShortFormExplicit,
            'short_explicit': ShortFormImplicit,
        }[algorithm]


    def predict(self, B=None, F=None, Q=None):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations.
        Parameters
        ----------
        u : np.array
            Optional control vector. If not `None`, it is multiplied by B
            to create the control input into the system.
        B : np.array(dim_x, dim_z), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B`.
        F : np.array(dim_x, dim_x), or None
            Optional state transition matrix; a value of None
            will cause the filter to use `self.F`.
        Q : np.array(dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None will cause the
            filter to use `self.Q`.
        """

        cdef uintptr_t _Phi_ptr = self.F.device_ctypes_pointer.value
        cdef uintptr_t _x_up_ptr = self.x.device_ctypes_pointer.value
        cdef uintptr_t _x_est_ptr = self.x_prev.device_ctypes_pointer.value
        cdef uintptr_t _P_up_ptr = self.P.device_ctypes_pointer.value
        cdef uintptr_t _P_est_ptr = self.P_prev.device_ctypes_pointer.value
        cdef uintptr_t _Q_ptr = self.Q.device_ctypes_pointer.value
        cdef uintptr_t _H_ptr = self.H.device_ctypes_pointer.value
        cdef uintptr_t _R_ptr = self.R.device_ctypes_pointer.value
        cdef uintptr_t _z_ptr = self.z.device_ctypes_pointer.value

        cdef uintptr_t _ws_ptr = self.workspace.device_ctypes_pointer.value

        cdef Variables[float] var32
        cdef Variables[double] var64
        cdef size_t workspace_size, current_size

        current_size = self._workspace_size

        cdef int dim_x
        cdef int dim_z

        dim_x = self.dim_x
        dim_z = self.dim_z

        cdef Option algo
        algo = self._algorithm



        if self.precision == 'single':

            with nogil:

                workspace_size = get_workspace_size_f32(var32,
                                                        <int> dim_x,
                                                        <int> dim_z,
                                                        <Option> algo,
                                                        <float*> _x_est_ptr,
                                                        <float*> _x_up_ptr,
                                                        <float*> _Phi_ptr,
                                                        <float*> _P_est_ptr,
                                                        <float*> _P_up_ptr,
                                                        <float*> _Q_ptr,
                                                        <float*> _R_ptr,
                                                        <float*> _H_ptr)

                init_f32(var32,
                         <int> dim_x,
                         <int> dim_z,
                         <Option> LongForm,
                         <float*> _x_est_ptr,
                         <float*> _x_up_ptr,
                         <float*> _Phi_ptr,
                         <float*> _P_est_ptr,
                         <float*> _P_up_ptr,
                         <float*> _Q_ptr,
                         <float*> _R_ptr,
                         <float*> _H_ptr,
                         <void*> _ws_ptr,
                         <size_t&> workspace_size)

                predict_f32(var32)

        else:

            with nogil:

                workspace_size = get_workspace_size_f64(var64,
                                                        <int> dim_x,
                                                        <int> dim_z,
                                                        <Option> algo,
                                                        <double*> _x_est_ptr,
                                                        <double*> _x_up_ptr,
                                                        <double*> _Phi_ptr,
                                                        <double*> _P_est_ptr,
                                                        <double*> _P_up_ptr,
                                                        <double*> _Q_ptr,
                                                        <double*> _R_ptr,
                                                        <double*> _H_ptr)

                init_f64(var64,
                         <int> dim_x,
                         <int> dim_z,
                         <Option> LongForm,
                         <double*> _x_est_ptr,
                         <double*> _x_up_ptr,
                         <double*> _Phi_ptr,
                         <double*> _P_est_ptr,
                         <double*> _P_up_ptr,
                         <double*> _Q_ptr,
                         <double*> _R_ptr,
                         <double*> _H_ptr,
                         <void*> _ws_ptr,
                         <size_t&> workspace_size)


                predict_f64(var64)

            # if workspace_size != current_size:
        #     self.workspace = cuda.to_device(np.zeros(workspace_size, dtype=self.dtype))


    def update(self, z, R=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter.
        If z is None, nothing is computed. However, x_post and P_post are
        updated with the prior (x_prior, P_prior), and self.z is set to None.
        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.
        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.
        H : np.array, or None
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        """

        cdef uintptr_t _Phi_ptr = self.F.device_ctypes_pointer.value
        cdef uintptr_t _x_up_ptr = self.x.device_ctypes_pointer.value
        cdef uintptr_t _x_est_ptr = self.x_prev.device_ctypes_pointer.value
        cdef uintptr_t _P_up_ptr = self.P.device_ctypes_pointer.value
        cdef uintptr_t _P_est_ptr = self.P_prev.device_ctypes_pointer.value
        cdef uintptr_t _Q_ptr = self.Q.device_ctypes_pointer.value
        cdef uintptr_t _H_ptr = self.H.device_ctypes_pointer.value
        cdef uintptr_t _R_ptr = self.R.device_ctypes_pointer.value
        cdef uintptr_t _z_ptr = self.z.device_ctypes_pointer.value

        cdef Variables[float] var32
        cdef Variables[double] var64
        cdef size_t workspace_size, current_size

        current_size = self._workspace_size

        cdef int dim_x
        cdef int dim_z

        dim_x = self.dim_x
        dim_z = self.dim_z

        cdef uintptr_t _ws_ptr = self.workspace.device_ctypes_pointer.value
        cdef uintptr_t z_ptr = z.device_ctypes_pointer.value

        if self.precision == 'single':

            with nogil:

                workspace_size = get_workspace_size_f32(var32,
                                                        <int> dim_x,
                                                        <int> dim_z,
                                                        <Option> LongForm,
                                                        <float*> _x_est_ptr,
                                                        <float*> _x_up_ptr,
                                                        <float*> _Phi_ptr,
                                                        <float*> _P_est_ptr,
                                                        <float*> _P_up_ptr,
                                                        <float*> _Q_ptr,
                                                        <float*> _R_ptr,
                                                        <float*> _H_ptr)



                init_f32(var32,
                         <int> dim_x,
                         <int> dim_z,
                         <Option> LongForm,
                         <float*> _x_est_ptr,
                         <float*> _x_up_ptr,
                         <float*> _Phi_ptr,
                         <float*> _P_est_ptr,
                         <float*> _P_up_ptr,
                         <float*> _Q_ptr,
                         <float*> _R_ptr,
                         <float*> _H_ptr,
                         <void*> _ws_ptr,
                         <size_t&> workspace_size)

                update_f32(var32,
                           <float*> z_ptr)

        else:

            with nogil:

                workspace_size = get_workspace_size_f64(var64,
                                                        <int> dim_x,
                                                        <int> dim_z,
                                                        <Option> LongForm,
                                                        <double*> _x_est_ptr,
                                                        <double*> _x_up_ptr,
                                                        <double*> _Phi_ptr,
                                                        <double*> _P_est_ptr,
                                                        <double*> _P_up_ptr,
                                                        <double*> _Q_ptr,
                                                        <double*> _R_ptr,
                                                        <double*> _H_ptr)



                init_f64(var64,
                         <int> dim_x,
                         <int> dim_z,
                         <Option> LongForm,
                         <double*> _x_est_ptr,
                         <double*> _x_up_ptr,
                         <double*> _Phi_ptr,
                         <double*> _P_est_ptr,
                         <double*> _P_up_ptr,
                         <double*> _Q_ptr,
                         <double*> _R_ptr,
                         <double*> _H_ptr,
                         <void*> _ws_ptr,
                         <size_t&> workspace_size)

                update_f64(var64,
                           <double*> z_ptr)


    def __setattr__(self, name, value):
        if name in ["F", "x_up", "x", "P_up", "P", "Q", "H", "R", "z"]:
            if (isinstance(value, cudf.DataFrame)):
                val = value.as_gpu_matrix(order='C')

            elif (isinstance(value, cudf.Series)):
                val = value.to_gpu_array()

            elif (isinstance(value, np.ndarray) or cuda.devicearray.is_cuda_ndarray(value)):
                val = cuda.to_device(value)

            super(KalmanFilter, self).__setattr__(name, val)

        else:
            super(KalmanFilter, self).__setattr__(name, value)
