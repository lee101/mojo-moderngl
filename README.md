# mojo-moderngl

`mojo-moderngl` is a standalone Mojo port of the CPU-appropriate buffer and
vertex-data parts of [ModernGL](https://github.com/moderngl/moderngl). It keeps
the covered Python names and signatures close enough that code can use:

```python
import mojomoderngl as moderngl
```

This is intentionally not a second OpenGL driver. Buffers live in host memory,
and Mojo handles the byte movement and compute-heavy vertex loops.

## Covered subset

- `create_context()` and `Context.buffer()` with ModernGL's `data`, `reserve`,
  and `dynamic` arguments
- `Context.copy_buffer()`
- `Buffer.write()`, `read()`, `read_into()`, `clear()`, `orphan()`,
  `release()`, `bind()`, and `assign()`
- ModernGL's strided `Buffer.write_chunks()`, `read_chunks()`, and
  `read_chunks_into()` helpers
- `Buffer.size`, `dynamic`, `glo`, `label`, `extra`, and `ctx`
- float32 and float64 vertex transforms for 2D, 3D, and homogeneous 4D data
- inverse-transpose normal transforms, row normalization, bounds, indexed
  expansion, and area-weighted indexed normal generation

The port does not cover OpenGL context acquisition, shaders, rendering,
textures, framebuffers, GPU resource bindings, or the full `VertexArray` API.
`bind_to_uniform_block()` and `bind_to_storage_buffer()` therefore raise
`NotImplementedError`. Use upstream ModernGL when the data needs to remain on
the GPU.

## Install

Install the pinned Mojo nightly, Python, NumPy, pytest, and ModernGL 5.12.0:

```bash
pixi install
pixi run build
```

The build produces `dist/libmojo-moderngl.so`. The Pixi environment sets
`PYTHONPATH=python`, so no editable install is required.

## Usage

```python
import numpy as np
import mojomoderngl as moderngl

ctx = moderngl.create_context(standalone=True)
buffer = ctx.buffer(np.arange(12, dtype=np.float32))

copy = bytearray(buffer.size)
buffer.read_into(copy)
assert copy == np.arange(12, dtype=np.float32).tobytes()

points = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float32)
matrix = np.eye(4, dtype=np.float32)
matrix[:3, 3] = [10, 20, 30]

transformed = moderngl.transform_points(points, matrix)
low, high = moderngl.bounds(transformed)

assert np.array_equal(transformed, [[10, 20, 30], [11, 22, 33]])
assert np.array_equal(low, [10, 20, 30])
assert np.array_equal(high, [11, 22, 33])
```

Run the parity suite with `pixi run test`.

## Benchmarks

Measured on Linux 6.8.0-136-generic, x86_64, with an NVIDIA GeForce RTX 5090
OpenGL/EGL reference context. Times are the best of three runs from
`pixi run bench`. Buffer and transform-feedback comparisons include the final
readback to host memory; the ModernGL inputs are already GPU-resident.

| case | mojo-moderngl | reference | speedup | reference |
|---|---:|---:|---:|---|
| Buffer.clear + read (32 MiB) | 34.71 ms | 361.04 ms | 10.40x | ModernGL/EGL |
| Buffer.read_chunks (16 MiB) | 2.31 ms | 6.46 ms | 2.80x | ModernGL/EGL |
| Buffer.write_chunks + read | 33.87 ms | 38.59 ms | 1.14x | ModernGL/EGL |
| Affine transform (2M vec3) | 5.21 ms | 9.43 ms | 1.81x | ModernGL/EGL |
| Bounds (5M vec3) | 22.32 ms | 344.16 ms | 15.42x | NumPy |
| Indexed normals (500k faces) | 19.86 ms | 252.89 ms | 12.73x | NumPy |

Mojo leads every measured case in this run. The narrowest gain is the strided
write at 1.14x; thresholded parallel gathering moves the formerly slower
strided read to 2.80x over the ModernGL/EGL reference.

There is no optional GPU compute path. The measured kernels stay below roughly
two floating-point operations per byte moved, so device transfers cannot be
amortized. Keeping them on the CPU also avoids duplicating host-resident buffers
on the device.

## How it works

All kernels are in one Mojo compilation unit and exported with a C ABI.
Python owns every array allocation as a C-contiguous NumPy array. ctypes
passes the array address as a 64-bit integer, and the exported Mojo wrapper
reconstructs an `UnsafePointer[..., AnyOrigin[mut=True]]` inside the function.
Read methods allocate their final Python `bytes` object directly, so results
do not pass through a temporary NumPy array and a second copy. Strided reads of
at least 4 MiB split into eight disjoint chunk ranges handled by cached worker
threads; smaller reads stay serial to avoid launch overhead.

`Buffer` storage is a contiguous `uint8` array, so offsets, chunk sizes, and
copy ranges have byte semantics matching ModernGL. Vertex arrays are
row-major `(vertex_count, component_count)` float32 or float64 arrays.
Transform matrices are row-major; normals from a 4x4 model matrix use the
inverse transpose of its upper-left 3x3 block.

The 52-test suite checks byte-for-byte buffer behavior against ModernGL 5.12.0
on headless EGL, compares affine and normal transforms against real transform
feedback shaders, and validates the remaining kernels against NumPy formulas.
