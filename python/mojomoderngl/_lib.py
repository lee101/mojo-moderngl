"""Load the Mojo shared library and declare its C ABI."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "capi.mojo")
LIB = os.environ.get("MOJOMODERNGL_LIB") or os.path.join(
    ROOT, "dist", "libmojo-moderngl.so"
)

I = ctypes.c_int64
F32 = ctypes.c_float
F64 = ctypes.c_double

_SIGNATURES = {
    "mmgl_copy_bytes": ([I, I, I], None),
    "mmgl_fill_pattern": ([I, I, I, I], None),
    "mmgl_gather_chunks": ([I, I, I, I, I, I], None),
    "mmgl_gather_chunk_range": ([I, I, I, I, I, I, I], None),
    "mmgl_scatter_chunks": ([I, I, I, I, I, I], None),
    "mmgl_transform_f32": ([I, I, I, I, I, F32, I], None),
    "mmgl_transform_f64": ([I, I, I, I, I, F64, I], None),
    "mmgl_transform_normals_f32": ([I, I, I, I, I], None),
    "mmgl_transform_normals_f64": ([I, I, I, I, I], None),
    "mmgl_normalize_f32": ([I, I, I, I], None),
    "mmgl_normalize_f64": ([I, I, I, I], None),
    "mmgl_bounds_f32": ([I, I, I, I], None),
    "mmgl_bounds_f64": ([I, I, I, I], None),
    "mmgl_deindex_f32": ([I, I, I, I, I], None),
    "mmgl_deindex_f64": ([I, I, I, I, I], None),
    "mmgl_compute_normals_f32": ([I, I, I, I, I, I], None),
    "mmgl_compute_normals_f64": ([I, I, I, I, I, I], None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(SRC)
    ):
        return LIB
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    return int(array.ctypes.data)


_PyBytes_FromStringAndSize = ctypes.pythonapi.PyBytes_FromStringAndSize
_PyBytes_FromStringAndSize.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t]
_PyBytes_FromStringAndSize.restype = ctypes.py_object
_PyBytes_AsString = ctypes.pythonapi.PyBytes_AsString
_PyBytes_AsString.argtypes = [ctypes.py_object]
_PyBytes_AsString.restype = ctypes.c_void_p


def empty_bytes(size: int) -> tuple[bytes, int]:
    result = _PyBytes_FromStringAndSize(None, size)
    return result, int(_PyBytes_AsString(result))
