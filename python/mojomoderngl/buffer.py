"""CPU-resident implementations of ModernGL's Context and Buffer subset."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from ._lib import addr, empty_bytes, lib


class Error(RuntimeError):
    pass


class InvalidObject(Error):
    pass


def _readable_bytes(data: Any) -> np.ndarray:
    try:
        view = memoryview(data)
        if not view.c_contiguous:
            raise BufferError("buffer is not C-contiguous")
        return np.frombuffer(view.cast("B"), dtype=np.uint8)
    except (TypeError, BufferError, ValueError) as exc:
        raise TypeError("data must support the contiguous buffer protocol") from exc


def _writable_bytes(data: Any) -> np.ndarray:
    try:
        view = memoryview(data)
        if view.readonly:
            raise TypeError("buffer is read-only")
        if not view.c_contiguous:
            raise BufferError("buffer is not C-contiguous")
        return np.frombuffer(view.cast("B"), dtype=np.uint8)
    except (BufferError, ValueError) as exc:
        raise TypeError("buffer must be writable and C-contiguous") from exc


def _range(size: int, offset: int, capacity: int) -> tuple[int, int]:
    if offset < 0:
        raise Error("offset must be non-negative")
    if size == -1:
        size = capacity - offset
    if size < 0 or offset + size > capacity:
        raise Error("buffer range is out of bounds")
    return size, offset


class Buffer:
    """A byte buffer with the covered ModernGL Buffer methods and signatures."""

    def __init__(self, ctx: Context, storage: np.ndarray, dynamic: bool, glo: int):
        self.ctx = ctx
        self._storage: np.ndarray | None = storage
        self._dynamic = bool(dynamic)
        self._glo = glo
        self.extra: Any = None
        self._label: str | None = None

    def _array(self) -> np.ndarray:
        if self._storage is None:
            raise InvalidObject("buffer has been released")
        return self._storage

    @property
    def size(self) -> int:
        return int(self._array().size)

    @property
    def dynamic(self) -> bool:
        return self._dynamic

    @property
    def glo(self) -> int:
        return self._glo

    @property
    def label(self) -> str | None:
        return self._label

    @label.setter
    def label(self, value: str | None) -> None:
        if not isinstance(value, str) and value is not None:
            raise TypeError(
                f"Expected value to be a str or None, got {type(value).__name__}"
            )
        self._label = value

    def write(self, data: Any, offset: int = 0) -> None:
        source = _readable_bytes(data)
        storage = self._array()
        _range(int(source.size), offset, int(storage.size))
        if source.size:
            if np.shares_memory(source, storage):
                source = source.copy()
            lib().mmgl_copy_bytes(addr(source), addr(storage) + offset, source.size)

    def write_chunks(self, data: Any, start: int, step: int, count: int) -> None:
        source = _readable_bytes(data)
        storage = self._array()
        if count < 0:
            raise Error("count must be non-negative")
        if count == 0:
            if source.size:
                raise Error("data must be empty when count is zero")
            return
        if source.size % count:
            raise Error("data size must be divisible by count")
        chunk_size = int(source.size // count)
        self._check_chunks(chunk_size, start, step, count, int(storage.size))
        if source.size:
            if np.shares_memory(source, storage):
                source = source.copy()
            lib().mmgl_scatter_chunks(
                addr(source), addr(storage), chunk_size, start, step, count
            )

    def read(self, size: int = -1, offset: int = 0) -> bytes:
        storage = self._array()
        size, offset = _range(size, offset, int(storage.size))
        result, result_addr = empty_bytes(size)
        if size:
            lib().mmgl_copy_bytes(addr(storage) + offset, result_addr, size)
        return result

    def read_into(
        self, buffer: Any, size: int = -1, offset: int = 0, write_offset: int = 0
    ) -> None:
        storage = self._array()
        target = _writable_bytes(buffer)
        size, offset = _range(size, offset, int(storage.size))
        _range(size, write_offset, int(target.size))
        if size:
            if np.shares_memory(storage, target):
                target[write_offset : write_offset + size] = storage[
                    offset : offset + size
                ].copy()
            else:
                lib().mmgl_copy_bytes(
                    addr(storage) + offset, addr(target) + write_offset, size
                )

    def read_chunks(
        self, chunk_size: int, start: int, step: int, count: int
    ) -> bytes:
        storage = self._array()
        self._check_chunks(chunk_size, start, step, count, int(storage.size))
        total = chunk_size * count
        result, result_addr = empty_bytes(total)
        if total:
            lib().mmgl_gather_chunks(
                addr(storage), result_addr, chunk_size, start, step, count
            )
        return result

    def read_chunks_into(
        self,
        buffer: Any,
        chunk_size: int,
        start: int,
        step: int,
        count: int,
        write_offset: int = 0,
    ) -> None:
        storage = self._array()
        target = _writable_bytes(buffer)
        self._check_chunks(chunk_size, start, step, count, int(storage.size))
        total = chunk_size * count
        _range(total, write_offset, int(target.size))
        if total:
            if np.shares_memory(storage, target):
                temporary = np.empty(total, dtype=np.uint8)
                lib().mmgl_gather_chunks(
                    addr(storage), addr(temporary), chunk_size, start, step, count
                )
                target[write_offset : write_offset + total] = temporary
            else:
                lib().mmgl_gather_chunks(
                    addr(storage),
                    addr(target) + write_offset,
                    chunk_size,
                    start,
                    step,
                    count,
                )

    @staticmethod
    def _check_chunks(
        chunk_size: int, start: int, step: int, count: int, capacity: int
    ) -> None:
        if min(chunk_size, start, step, count) < 0:
            raise Error("chunk arguments must be non-negative")
        if count and start + (count - 1) * step + chunk_size > capacity:
            raise Error("chunk range is out of bounds")

    def clear(
        self, size: int = -1, offset: int = 0, chunk: Any | None = None
    ) -> None:
        storage = self._array()
        size, offset = _range(size, offset, int(storage.size))
        pattern = (
            np.zeros(1, dtype=np.uint8)
            if chunk is None
            else _readable_bytes(chunk)
        )
        if not pattern.size:
            raise Error("chunk cannot be empty")
        if size % pattern.size:
            raise Error("the chunk does not fit the size")
        if size:
            if np.shares_memory(pattern, storage):
                pattern = pattern.copy()
            lib().mmgl_fill_pattern(
                addr(storage) + offset, size, addr(pattern), int(pattern.size)
            )

    def orphan(self, size: int = -1) -> None:
        storage = self._array()
        if size == -1:
            size = int(storage.size)
        if size <= 0:
            raise Error("size must be positive")
        self._storage = np.zeros(size, dtype=np.uint8)

    def bind_to_uniform_block(
        self, binding: int = 0, offset: int = 0, size: int = -1
    ) -> None:
        raise NotImplementedError("OpenGL binding is outside this CPU port")

    def bind_to_storage_buffer(
        self, binding: int = 0, offset: int = 0, size: int = -1
    ) -> None:
        raise NotImplementedError("OpenGL binding is outside this CPU port")

    def release(self) -> None:
        self._storage = None

    def bind(self, *attribs: str, layout: str | None = None) -> tuple:
        return (self, layout, *attribs)

    def assign(self, index: int) -> tuple:
        return (self, index)


class Context:
    """A CPU context covering ModernGL's buffer allocation and copy API."""

    _ids = itertools.count(1)

    def __init__(self) -> None:
        self.extra: Any = None

    def buffer(
        self, data: Any | None = None, reserve: int = 0, dynamic: bool = False
    ) -> Buffer:
        if data is not None and reserve:
            raise Error("data and reserve are mutually exclusive")
        if data is None:
            if reserve <= 0:
                raise Error("missing data or reserve")
            storage = np.zeros(reserve, dtype=np.uint8)
        else:
            source = _readable_bytes(data)
            if not source.size:
                raise Error("buffer cannot be empty")
            storage = np.empty(source.size, dtype=np.uint8)
            lib().mmgl_copy_bytes(addr(source), addr(storage), source.size)
        return Buffer(self, storage, dynamic, next(self._ids))

    def copy_buffer(
        self,
        dst: Buffer,
        src: Buffer,
        size: int = -1,
        read_offset: int = 0,
        write_offset: int = 0,
    ) -> None:
        if not isinstance(dst, Buffer) or not isinstance(src, Buffer):
            raise TypeError("dst and src must be Buffer objects")
        source = src._array()
        target = dst._array()
        size, read_offset = _range(size, read_offset, int(source.size))
        _range(size, write_offset, int(target.size))
        if size:
            if np.shares_memory(source, target):
                target[write_offset : write_offset + size] = source[
                    read_offset : read_offset + size
                ].copy()
            else:
                lib().mmgl_copy_bytes(
                    addr(source) + read_offset, addr(target) + write_offset, size
                )

    def release(self) -> None:
        return None

    def gc(self) -> int:
        return 0


def create_context(
    require: int | None = None,
    standalone: bool = False,
    share: bool = False,
    **settings: Any,
) -> Context:
    del require, standalone, share, settings
    return Context()
