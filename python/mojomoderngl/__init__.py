"""Mojo implementations of ModernGL buffer semantics and vertex math."""

from . import vertex
from .buffer import Buffer, Context, Error, InvalidObject, create_context
from .vertex import (
    bounds,
    compute_normals,
    deindex,
    normalize,
    transform_normals,
    transform_points,
)

POINTS = 0
LINES = 1
LINE_LOOP = 2
LINE_STRIP = 3
TRIANGLES = 4
TRIANGLE_STRIP = 5
TRIANGLE_FAN = 6

__all__ = [
    "Buffer",
    "Context",
    "Error",
    "InvalidObject",
    "create_context",
    "vertex",
    "transform_points",
    "transform_normals",
    "normalize",
    "bounds",
    "deindex",
    "compute_normals",
]

__version__ = "0.1.0"
