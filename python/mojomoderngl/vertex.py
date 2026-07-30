"""Compute-heavy vertex array helpers implemented by Mojo kernels."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._lib import addr, lib


def _vertices(values: Any, components: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] not in components:
        allowed = ", ".join(map(str, components))
        raise ValueError(f"vertices must have shape (n, components), components={allowed}")
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("vertices must have dtype float32 or float64")
    return np.ascontiguousarray(array)


def _indices(values: Any) -> np.ndarray:
    index = np.asarray(values)
    if index.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if index.dtype.kind not in "iu":
        raise TypeError("indices must have an integer dtype")
    if index.size:
        info = np.iinfo(np.int64)
        if index.min() < info.min or index.max() > info.max:
            raise OverflowError("indices do not fit in int64")
    return np.ascontiguousarray(index, dtype=np.int64)


def _output(source: np.ndarray, out: Any | None) -> np.ndarray:
    if out is None:
        return np.empty_like(source)
    result = np.asarray(out)
    if (
        result.shape != source.shape
        or result.dtype != source.dtype
        or not result.flags.c_contiguous
        or not result.flags.writeable
    ):
        raise ValueError("out must be writable, C-contiguous, and match shape and dtype")
    return result


def _suffix(array: np.ndarray) -> str:
    return "f64" if array.dtype == np.float64 else "f32"


def transform_points(
    vertices: Any,
    matrix: Any,
    *,
    w: float = 1.0,
    perspective_divide: bool = False,
    out: Any | None = None,
) -> np.ndarray:
    """Apply a row-major 4x4 matrix to an ``(n, 2|3|4)`` vertex array."""
    source = _vertices(vertices, (2, 3, 4))
    transform = np.ascontiguousarray(matrix, dtype=source.dtype)
    if transform.shape != (4, 4):
        raise ValueError("matrix must have shape (4, 4)")
    result = _output(source, out)
    if source.size:
        fn = getattr(lib(), f"mmgl_transform_{_suffix(source)}")
        fn(
            addr(source),
            addr(transform),
            addr(result),
            source.shape[0],
            source.shape[1],
            w,
            int(perspective_divide),
        )
    return result


def transform_normals(
    normals: Any,
    matrix: Any,
    *,
    normalize: bool = True,
    out: Any | None = None,
) -> np.ndarray:
    """Transform normals by a 3x3 matrix or a 4x4 matrix's inverse transpose."""
    source = _vertices(normals, (3,))
    transform = np.asarray(matrix, dtype=source.dtype)
    if transform.shape == (4, 4):
        transform = np.linalg.inv(transform[:3, :3]).T
    elif transform.shape != (3, 3):
        raise ValueError("matrix must have shape (3, 3) or (4, 4)")
    transform = np.ascontiguousarray(transform, dtype=source.dtype)
    result = _output(source, out)
    if source.size:
        fn = getattr(lib(), f"mmgl_transform_normals_{_suffix(source)}")
        fn(
            addr(source),
            addr(transform),
            addr(result),
            source.shape[0],
            int(normalize),
        )
    return result


def normalize(vectors: Any, *, out: Any | None = None) -> np.ndarray:
    """Normalize each row, leaving zero-length rows unchanged."""
    source = _vertices(vectors, (2, 3, 4))
    result = _output(source, out)
    if source.size:
        fn = getattr(lib(), f"mmgl_normalize_{_suffix(source)}")
        fn(addr(source), addr(result), source.shape[0], source.shape[1])
    return result


def bounds(vertices: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return componentwise minimum and maximum vertex coordinates."""
    source = _vertices(vertices, (2, 3, 4))
    if not source.shape[0]:
        raise ValueError("vertices cannot be empty")
    result = np.empty((2, source.shape[1]), dtype=source.dtype)
    fn = getattr(lib(), f"mmgl_bounds_{_suffix(source)}")
    fn(addr(source), addr(result), source.shape[0], source.shape[1])
    return result[0], result[1]


def deindex(vertices: Any, indices: Any) -> np.ndarray:
    """Expand an indexed vertex array while preserving all 2-4 components."""
    source = _vertices(vertices, (2, 3, 4))
    index = _indices(indices)
    if index.size and (index.min() < 0 or index.max() >= source.shape[0]):
        raise IndexError("vertex index out of bounds")
    result = np.empty((index.size, source.shape[1]), dtype=source.dtype)
    if index.size:
        fn = getattr(lib(), f"mmgl_deindex_{_suffix(source)}")
        fn(addr(source), addr(index), addr(result), index.size, source.shape[1])
    return result


def compute_normals(
    vertices: Any, indices: Any | None = None, *, normalize: bool = True
) -> np.ndarray:
    """Generate area-weighted vertex normals for a triangle mesh."""
    source = _vertices(vertices, (3,))
    if indices is None:
        if source.shape[0] % 3:
            raise ValueError("unindexed triangle vertices must be a multiple of three")
        index = np.arange(source.shape[0], dtype=np.int64)
    else:
        raw_index = np.asarray(indices)
        if raw_index.ndim == 2 and raw_index.shape[1:] == (3,):
            raw_index = raw_index.reshape(-1)
        index = _indices(raw_index)
        if index.size % 3:
            raise ValueError("indices must contain complete triangles")
        if index.size and (index.min() < 0 or index.max() >= source.shape[0]):
            raise IndexError("vertex index out of bounds")
    result = np.empty_like(source)
    if source.shape[0]:
        fn = getattr(lib(), f"mmgl_compute_normals_{_suffix(source)}")
        fn(
            addr(source),
            addr(index),
            addr(result),
            source.shape[0],
            index.size,
            int(normalize),
        )
    return result
