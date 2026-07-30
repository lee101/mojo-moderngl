"""Honest CPU Mojo comparisons with ModernGL 5.12 and NumPy."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import moderngl  # noqa: E402
import mojomoderngl as mm  # noqa: E402


def timeit(fn, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def numpy_normals(vertices, indices):
    triangles = indices.reshape(-1, 3)
    a = vertices[triangles[:, 0]]
    b = vertices[triangles[:, 1]]
    c = vertices[triangles[:, 2]]
    face = np.cross(b - a, c - a)
    result = np.zeros_like(vertices)
    np.add.at(result, triangles[:, 0], face)
    np.add.at(result, triangles[:, 1], face)
    np.add.at(result, triangles[:, 2], face)
    length = np.linalg.norm(result, axis=1, keepdims=True)
    np.divide(result, length, out=result, where=length != 0)
    return result


def main():
    rng = np.random.default_rng(0)
    gl = moderngl.create_context(standalone=True, backend="egl")
    gl.simple_framebuffer((1, 1)).use()
    cpu = mm.create_context()
    rows = []

    nbytes = 32 * 1024 * 1024
    initial = rng.integers(0, 256, size=nbytes, dtype=np.uint8)
    a = cpu.buffer(initial)
    b = gl.buffer(initial)
    pattern = b"\x12\x34\x56\x78"

    def mojo_clear():
        a.clear(chunk=pattern)
        return a.read()

    def gl_clear():
        b.clear(chunk=pattern)
        return b.read()

    assert mojo_clear() == gl_clear()
    rows.append(("Buffer.clear + read (32 MiB)", timeit(mojo_clear), timeit(gl_clear), "ModernGL/EGL"))

    chunk_size, step, count = 16, 32, nbytes // 32

    def mojo_gather():
        return a.read_chunks(chunk_size, 0, step, count)

    def gl_gather():
        return b.read_chunks(chunk_size, 0, step, count)

    assert mojo_gather() == gl_gather()
    rows.append(("Buffer.read_chunks (16 MiB)", timeit(mojo_gather), timeit(gl_gather), "ModernGL/EGL"))

    payload = rng.integers(0, 256, size=chunk_size * count, dtype=np.uint8)

    def mojo_scatter():
        a.write_chunks(payload, 0, step, count)
        return a.read()

    def gl_scatter():
        b.write_chunks(payload, 0, step, count)
        return b.read()

    assert mojo_scatter() == gl_scatter()
    rows.append(("Buffer.write_chunks + read", timeit(mojo_scatter), timeit(gl_scatter), "ModernGL/EGL"))

    point_count = 2_000_000
    points = rng.normal(size=(point_count, 3)).astype("f4")
    matrix = np.array(
        [[1.2, 0.1, 0.0, 3.0], [0.0, 0.8, 0.2, -2.0], [0.3, 0.0, 1.5, 1.0], [0, 0, 0, 1]],
        dtype="f4",
    )
    program = gl.program(
        vertex_shader="""
            #version 330
            in vec3 in_position;
            uniform mat4 matrix;
            out vec3 out_position;
            void main() {
                out_position = (matrix * vec4(in_position, 1.0)).xyz;
            }
        """,
        varyings=["out_position"],
    )
    program["matrix"].write(matrix.T.tobytes())
    gl_source = gl.buffer(points.tobytes())
    gl_target = gl.buffer(reserve=points.nbytes)
    vao = gl.vertex_array(program, [(gl_source, "3f", "in_position")])
    mojo_target = np.empty_like(points)

    def mojo_transform():
        return mm.transform_points(points, matrix, out=mojo_target)

    def gl_transform():
        vao.transform(gl_target, mode=moderngl.POINTS, vertices=point_count)
        return gl_target.read()

    expected = np.frombuffer(gl_transform(), dtype="f4").reshape(-1, 3)
    assert np.allclose(mojo_transform(), expected, rtol=2e-6, atol=2e-6)
    rows.append(("Affine transform (2M vec3)", timeit(mojo_transform), timeit(gl_transform), "ModernGL/EGL"))

    bounds_points = rng.normal(size=(5_000_000, 3)).astype("f4")
    rows.append(
        (
            "Bounds (5M vec3)",
            timeit(lambda: mm.bounds(bounds_points)),
            timeit(lambda: (bounds_points.min(0), bounds_points.max(0))),
            "NumPy",
        )
    )

    vertex_count = 250_000
    face_count = 500_000
    mesh = rng.normal(size=(vertex_count, 3)).astype("f4")
    indices = rng.integers(0, vertex_count, size=face_count * 3, dtype=np.int64)
    mojo_ref = mm.compute_normals(mesh, indices)
    numpy_ref = numpy_normals(mesh, indices)
    assert np.allclose(mojo_ref, numpy_ref, rtol=3e-4, atol=3e-4)
    rows.append(
        (
            "Indexed normals (500k faces)",
            timeit(lambda: mm.compute_normals(mesh, indices)),
            timeit(lambda: numpy_normals(mesh, indices)),
            "NumPy",
        )
    )

    cpu_name = platform.processor() or platform.machine()
    print(f"Machine: {platform.system()} {platform.release()}, {cpu_name}")
    print(f"GPU/reference context: {gl.info.get('GL_RENDERER', 'unknown')}")
    print()
    print("| case | mojo-moderngl | reference | speedup | reference |")
    print("|---|---:|---:|---:|---|")
    for name, ours, reference, source in rows:
        ratio = reference / ours
        print(
            f"| {name} | {ours * 1e3:.2f} ms | {reference * 1e3:.2f} ms | "
            f"{ratio:.2f}x | {source} |"
        )


if __name__ == "__main__":
    main()
