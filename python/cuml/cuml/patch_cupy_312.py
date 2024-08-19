# Copyright (c) 2024, NVIDIA CORPORATION.

import itertools
import binascii
import os
import hashlib
import time

import numpy

import cupy
from cupy import _core
from cupy._core import _fusion_interface
from cupy._core import fusion
from cupy._sorting import search
from cupy_backends.cuda.api import runtime


_UINT64_MAX = 0xFFFFFFFFFFFFFFFF


def copyto(dst, src, casting="same_kind", where=None):
    src_is_scalar = False
    src_type = type(src)

    if src_type in (bool, int, float, complex):
        dst_arr = numpy.empty((), dtype=dst.dtype)
        # NumPy 1.x and 2.0 make implementing copyto cast safety hard, so
        # test whether NumPy copy allows the copy operation:
        numpy.copyto(dst_arr, src, casting=casting)
        can_cast = True
        src_is_scalar = True
    elif src_type in (fusion._FusionVarScalar, _fusion_interface._ScalarProxy):
        src_dtype = src.dtype
        can_cast = numpy.can_cast(src_dtype, dst.dtype, casting)
        src_is_scalar = True
    elif isinstance(src, numpy.ndarray) or numpy.isscalar(src):
        if src.size != 1:
            raise ValueError(
                "non-scalar numpy.ndarray cannot be used for copyto"
            )
        src_dtype = src.dtype
        can_cast = numpy.can_cast(src, dst.dtype, casting)
        src = src.item()
        src_is_scalar = True
    else:
        src_dtype = src.dtype
        can_cast = numpy.can_cast(src_dtype, dst.dtype, casting)

    if not can_cast:
        raise TypeError(
            "Cannot cast %s to %s in %s casting mode"
            % (src_dtype, dst.dtype, casting)
        )

    if fusion._is_fusing():
        # TODO(kataoka): NumPy allows stripping leading unit dimensions.
        # But fusion array proxy does not currently support
        # `shape` and `squeeze`.

        if where is None:
            _core.elementwise_copy(src, dst)
        else:
            fusion._call_ufunc(search._where_ufunc, where, src, dst, dst)
        return

    if not src_is_scalar:
        # Check broadcast condition
        # - for fast-paths and
        # - for a better error message (than ufunc's).
        # NumPy allows stripping leading unit dimensions.
        if not all(
            [
                s in (d, 1)
                for s, d in itertools.zip_longest(
                    reversed(src.shape), reversed(dst.shape), fillvalue=1
                )
            ]
        ):
            raise ValueError(
                "could not broadcast input array "
                f"from shape {src.shape} into shape {dst.shape}"
            )
        squeeze_ndim = src.ndim - dst.ndim
        if squeeze_ndim > 0:
            # always succeeds because broadcast conition is checked.
            src = src.squeeze(tuple(range(squeeze_ndim)))

    if where is not None:
        _core.elementwise_copy(src, dst, _where=where)
        return

    if dst.size == 0:
        return

    if src_is_scalar:
        _core.elementwise_copy(src, dst)
        return

    if _can_memcpy(dst, src):
        dst.data.copy_from_async(src.data, src.nbytes)
        return

    device = dst.device
    prev_device = runtime.getDevice()
    try:
        runtime.setDevice(device.id)
        if src.device != device:
            src = src.copy()
        _core.elementwise_copy(src, dst)
    finally:
        runtime.setDevice(prev_device)


def _can_memcpy(dst, src):
    c_contiguous = dst.flags.c_contiguous and src.flags.c_contiguous
    f_contiguous = dst.flags.f_contiguous and src.flags.f_contiguous
    return (
        (c_contiguous or f_contiguous)
        and dst.dtype == src.dtype
        and dst.size == src.size
    )


# Replace the main cupy functions:
cupy.copyto = copyto
cupy._manipulation.basic.copyto = copyto

# FFTshifts used now invalid symbol in NuMpy (could also patch `NumPy instead...)


def fftshift(x, axes=None):
    x = cupy.asarray(x)
    if axes is None:
        axes = list(range(x.ndim))
    elif isinstance(axes, int):
        axes = (axes,)
    return cupy.roll(x, [x.shape[axis] // 2 for axis in axes], axes)


def ifftshift(x, axes=None):
    x = cupy.asarray(x)
    if axes is None:
        axes = list(range(x.ndim))
    elif isinstance(axes, int):
        axes = (axes,)
    return cupy.roll(x, [-(x.shape[axis] // 2) for axis in axes], axes)


cupy.fft.fftshift = fftshift
cupy.fft.ifftshift = ifftshift


def seed(self, seed=None):
    from cupy_backends.cuda.libs import curand

    if seed is None:
        try:
            seed_str = binascii.hexlify(os.urandom(8))
            seed = int(seed_str, 16)
        except NotImplementedError:
            seed = (time.time() * 1000000) % _UINT64_MAX
    else:
        if isinstance(seed, numpy.ndarray):
            seed = int(hashlib.md5(seed).hexdigest()[:16], 16)
        else:
            seed_arr = numpy.asarray(seed)
            if seed_arr.dtype.kind not in "biu":
                raise TypeError("Seed must be an integer.")
            seed = int(seed_arr)
            # Check that no integer overflow occurred during the cast
            if seed < 0 or seed >= 2**64:
                raise ValueError(
                    "Seed must be an integer between 0 and 2**64 - 1"
                )

    curand.setPseudoRandomGeneratorSeed(self._generator, seed)
    if self.method not in (
        curand.CURAND_RNG_PSEUDO_MT19937,
        curand.CURAND_RNG_PSEUDO_MTGP32,
    ):
        curand.setGeneratorOffset(self._generator, 0)

    self._rk_seed = seed


cupy.random.RandomState.seed = seed
