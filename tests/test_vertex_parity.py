"""Vertex kernels use ModernGL transform feedback and NumPy references."""

import numpy as np
import pytest

import moderngl
import mojomoderngl as mm

rng = np.random.default_rng(7)


@pytest.fixture(scope="module")
def gl():
    try:
        ctx = moderngl.create_context(standalone=True, backend="egl")
        ctx.simple_framebuffer((1, 1)).use()
    except Exception as exc:
        pytest.skip(f"headless ModernGL transform feedback unavailable: {exc}")
    yield ctx
    ctx.release()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("components", [2, 3, 4])
def test_transform_points_matches_numpy(dtype, components):
    vertices = rng.normal(size=(301, components)).astype(dtype)
    matrix = rng.normal(size=(4, 4)).astype(dtype)
    got = mm.transform_points(vertices, matrix, w=1.25)
    homogeneous = np.zeros((len(vertices), 4), dtype=dtype)
    homogeneous[:, :components] = vertices
    if components < 4:
        homogeneous[:, 3] = 1.25
    ref = (homogeneous @ matrix.T)[:, :components]
    assert got.dtype == dtype
    assert np.allclose(got, ref, rtol=2e-6, atol=2e-6)


def test_transform_points_perspective_divide():
    vertices = rng.normal(size=(200, 3)).astype("f4")
    matrix = np.eye(4, dtype="f4")
    matrix[3] = [0.1, -0.2, 0.05, 2.0]
    got = mm.transform_points(vertices, matrix, perspective_divide=True)
    homogeneous = np.c_[vertices, np.ones(len(vertices), dtype="f4")]
    transformed = homogeneous @ matrix.T
    ref = transformed[:, :3] / transformed[:, 3, None]
    assert np.allclose(got, ref, rtol=2e-6, atol=2e-6)


def test_transform_points_reuses_output():
    vertices = rng.normal(size=(50, 3)).astype("f4")
    target = np.empty_like(vertices)
    returned = mm.transform_points(vertices, np.eye(4), out=target)
    assert returned is target
    assert np.array_equal(returned, vertices)


@pytest.mark.parametrize("count", [65535, 65539])
def test_transform_points_simd_tail_and_parallel_threshold(count):
    vertices = rng.normal(size=(count, 3)).astype("f4")
    matrix = np.array(
        [[1.2, 0.1, 0.0, 3.0], [0.0, 0.8, 0.2, -2.0], [0.3, 0.0, 1.5, 1.0], [0, 0, 0, 1]],
        dtype="f4",
    )
    got = mm.transform_points(vertices, matrix)
    ref = vertices @ matrix[:3, :3].T + matrix[:3, 3]
    assert np.allclose(got, ref, rtol=2e-6, atol=2e-6)


def test_affine_transform_matches_upstream_transform_feedback(gl):
    vertices = rng.normal(size=(4097, 3)).astype("f4")
    matrix = np.array(
        [
            [2.0, 0.1, 0.0, 3.0],
            [0.0, 3.0, 0.2, -2.0],
            [0.3, 0.0, 4.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
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
    source = gl.buffer(vertices.tobytes())
    target = gl.buffer(reserve=vertices.nbytes)
    vao = gl.vertex_array(program, [(source, "3f", "in_position")])
    vao.transform(target, mode=moderngl.POINTS, vertices=len(vertices))
    expected = np.frombuffer(target.read(), dtype="f4").reshape(-1, 3)
    got = mm.transform_points(vertices, matrix)
    assert np.allclose(got, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_transform_normals_matches_numpy(dtype):
    normals = rng.normal(size=(1000, 3)).astype(dtype)
    model = np.array(
        [[2.0, 0.2, 0.0, 4.0], [0.0, 3.0, 0.1, 2.0], [0.3, 0.0, 4.0, 1.0], [0, 0, 0, 1]],
        dtype=dtype,
    )
    got = mm.transform_normals(normals, model)
    normal_matrix = np.linalg.inv(model[:3, :3]).T
    ref = normals @ normal_matrix.T
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    assert np.allclose(got, ref, rtol=2e-6, atol=2e-6)


def test_normal_transform_matches_upstream_transform_feedback(gl):
    normals = rng.normal(size=(3073, 3)).astype("f4")
    matrix = np.array(
        [[1.5, 0.1, 0.0], [0.0, 0.8, 0.2], [0.3, 0.0, 2.0]], dtype="f4"
    )
    program = gl.program(
        vertex_shader="""
            #version 330
            in vec3 in_normal;
            uniform mat3 normal_matrix;
            out vec3 out_normal;
            void main() {
                out_normal = normalize(normal_matrix * in_normal);
            }
        """,
        varyings=["out_normal"],
    )
    program["normal_matrix"].write(matrix.T.tobytes())
    source = gl.buffer(normals.tobytes())
    target = gl.buffer(reserve=normals.nbytes)
    vao = gl.vertex_array(program, [(source, "3f", "in_normal")])
    vao.transform(target, mode=moderngl.POINTS, vertices=len(normals))
    expected = np.frombuffer(target.read(), dtype="f4").reshape(-1, 3)
    got = mm.transform_normals(normals, matrix)
    assert np.allclose(got, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("components", [2, 3, 4])
def test_normalize_matches_numpy_and_preserves_zero(components):
    vectors = rng.normal(size=(501, components)).astype("f4")
    vectors[17] = 0
    got = mm.normalize(vectors)
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    ref = np.divide(vectors, lengths, out=np.zeros_like(vectors), where=lengths != 0)
    assert np.allclose(got, ref, rtol=2e-6, atol=2e-6)
    assert np.array_equal(got[17], np.zeros(components))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_bounds_matches_numpy(dtype):
    vertices = rng.normal(size=(10007, 4)).astype(dtype)
    low, high = mm.bounds(vertices)
    assert np.array_equal(low, vertices.min(axis=0))
    assert np.array_equal(high, vertices.max(axis=0))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_deindex_matches_numpy(dtype):
    vertices = rng.normal(size=(1000, 4)).astype(dtype)
    indices = rng.integers(0, len(vertices), size=5003)
    assert np.array_equal(mm.deindex(vertices, indices), vertices[indices])


def numpy_normals(vertices, indices, normalize=True):
    triangles = indices.reshape(-1, 3)
    a, b, c = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    face = np.cross(b - a, c - a)
    result = np.zeros_like(vertices)
    np.add.at(result, triangles[:, 0], face)
    np.add.at(result, triangles[:, 1], face)
    np.add.at(result, triangles[:, 2], face)
    if normalize:
        length = np.linalg.norm(result, axis=1, keepdims=True)
        np.divide(result, length, out=result, where=length != 0)
    return result


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compute_normals_matches_numpy(dtype):
    vertices = rng.normal(size=(400, 3)).astype(dtype)
    indices = rng.integers(0, len(vertices), size=(700, 3), dtype=np.int64).ravel()
    got = mm.compute_normals(vertices, indices)
    ref = numpy_normals(vertices, indices)
    assert np.allclose(got, ref, rtol=2e-5, atol=2e-5)


def test_compute_normals_unindexed_triangle():
    triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="f4")
    expected = np.tile([0, 0, 1], (3, 1))
    assert np.array_equal(mm.compute_normals(triangle), expected)


def test_vertex_validation():
    with pytest.raises(ValueError):
        mm.transform_points(np.ones((3, 5)), np.eye(4))
    with pytest.raises(ValueError):
        mm.transform_points(np.ones((3, 3)), np.eye(3))
    with pytest.raises(ValueError):
        mm.bounds(np.empty((0, 3)))
    with pytest.raises(IndexError):
        mm.deindex(np.ones((3, 3)), [0, 3])
    with pytest.raises(ValueError):
        mm.compute_normals(np.ones((4, 3)))


@pytest.mark.parametrize(
    "values",
    [
        np.ones((3, 3), dtype=np.int64),
        np.ones((3, 3), dtype=np.float16),
        np.ones((3, 3), dtype=np.complex64),
    ],
)
def test_vertex_dtype_narrowing_is_rejected(values):
    with pytest.raises(TypeError, match="float32 or float64"):
        mm.transform_points(values, np.eye(4, dtype=np.float32))


def test_index_dtype_validation_precedes_ffi():
    vertices = np.ones((3, 3), dtype=np.float32)
    with pytest.raises(TypeError, match="integer"):
        mm.deindex(vertices, np.array([0.0, 1.0]))
    with pytest.raises(OverflowError, match="int64"):
        mm.deindex(vertices, np.array([2**63], dtype=np.uint64))
    with pytest.raises(ValueError, match="one-dimensional"):
        mm.deindex(vertices, np.array([[0, 1]], dtype=np.int64))
