"""Covered Buffer behavior is compared with ModernGL 5.12 on EGL."""

import inspect

import numpy as np
import pytest

import moderngl as upstream
import mojomoderngl as mojo


@pytest.fixture(scope="module")
def gl():
    try:
        ctx = upstream.create_context(standalone=True, backend="egl")
    except Exception as exc:
        pytest.skip(f"headless ModernGL context unavailable: {exc}")
    yield ctx
    ctx.release()


def both(gl, data):
    return mojo.create_context().buffer(data), gl.buffer(data)


def test_public_signatures_match_upstream():
    pairs = [
        (mojo.create_context, upstream.create_context),
        (mojo.Context.buffer, upstream.Context.buffer),
        (mojo.Context.copy_buffer, upstream.Context.copy_buffer),
        (mojo.Buffer.write, upstream.Buffer.write),
        (mojo.Buffer.read, upstream.Buffer.read),
        (mojo.Buffer.read_into, upstream.Buffer.read_into),
        (mojo.Buffer.write_chunks, upstream.Buffer.write_chunks),
        (mojo.Buffer.read_chunks, upstream.Buffer.read_chunks),
        (mojo.Buffer.read_chunks_into, upstream.Buffer.read_chunks_into),
        (mojo.Buffer.clear, upstream.Buffer.clear),
        (mojo.Buffer.orphan, upstream.Buffer.orphan),
        (mojo.Buffer.bind_to_uniform_block, upstream.Buffer.bind_to_uniform_block),
        (mojo.Buffer.bind_to_storage_buffer, upstream.Buffer.bind_to_storage_buffer),
        (mojo.Buffer.release, upstream.Buffer.release),
        (mojo.Buffer.bind, upstream.Buffer.bind),
        (mojo.Buffer.assign, upstream.Buffer.assign),
    ]
    for ours, theirs in pairs:
        assert list(inspect.signature(ours).parameters) == list(
            inspect.signature(theirs).parameters
        )


def test_buffer_creation_and_properties(gl):
    a, b = both(gl, b"vertex-data")
    assert a.read() == b.read() == b"vertex-data"
    assert a.size == b.size == 11
    dynamic = mojo.create_context().buffer(reserve=64, dynamic=True)
    assert dynamic.size == 64
    assert dynamic.dynamic is True
    assert dynamic.extra is None
    assert dynamic.ctx is not None
    assert isinstance(dynamic.glo, int)


def test_write_and_read_ranges(gl):
    a, b = both(gl, bytes(range(32)))
    a.write(b"abcdef", offset=7)
    b.write(b"abcdef", offset=7)
    assert a.read() == b.read()
    assert a.read(11, offset=5) == b.read(11, offset=5)
    assert a.read(offset=17) == b.read(offset=17)


def test_read_into_with_offsets(gl):
    a, b = both(gl, bytes(range(64)))
    ours = bytearray(b"x" * 40)
    theirs = bytearray(b"x" * 40)
    a.read_into(ours, size=17, offset=9, write_offset=6)
    b.read_into(theirs, size=17, offset=9, write_offset=6)
    assert ours == theirs


@pytest.mark.parametrize("chunk", [b"Z", b"xy", b"rgb", b"12345678"])
def test_pattern_clear_matches_upstream(gl, chunk):
    size = len(chunk) * 7
    a, b = both(gl, bytes(range(64)))
    a.clear(size=size, offset=8, chunk=chunk)
    b.clear(size=size, offset=8, chunk=chunk)
    assert a.read() == b.read()


def test_default_clear_all_matches_upstream(gl):
    a, b = both(gl, bytes(range(64)))
    a.clear()
    b.clear()
    assert a.read() == b.read() == bytes(64)


def test_read_chunks_matches_upstream(gl):
    a, b = both(gl, bytes(range(128)))
    assert a.read_chunks(7, 3, 17, 6) == b.read_chunks(7, 3, 17, 6)


def test_write_chunks_matches_upstream(gl):
    a, b = both(gl, bytes(range(128)))
    payload = bytes(range(35))
    a.write_chunks(payload, 4, 19, 5)
    b.write_chunks(payload, 4, 19, 5)
    assert a.read() == b.read()


@pytest.mark.parametrize("count", [33, 262145])
def test_chunk_roundtrip_simd_tail_and_parallel_threshold(count):
    chunk_size = 16
    start = 5
    step = 19
    capacity = start + (count - 1) * step + chunk_size
    payload = np.arange(count * chunk_size, dtype=np.uint8)
    buf = mojo.create_context().buffer(reserve=capacity)
    buf.write_chunks(payload, start, step, count)
    assert buf.read_chunks(chunk_size, start, step, count) == payload.tobytes()


def test_read_chunks_into():
    buf = mojo.create_context().buffer(bytes(range(128)))
    target = bytearray(b"x" * 64)
    buf.read_chunks_into(target, 5, 2, 13, 7, write_offset=9)
    expected = bytearray(b"x" * 64)
    expected[9:44] = b"".join(
        bytes(range(128))[2 + i * 13 : 7 + i * 13] for i in range(7)
    )
    assert target == expected


def test_copy_buffer_matches_upstream(gl):
    our_ctx = mojo.create_context()
    a_src = our_ctx.buffer(bytes(range(80)))
    a_dst = our_ctx.buffer(b"x" * 80)
    b_src = gl.buffer(bytes(range(80)))
    b_dst = gl.buffer(b"x" * 80)
    our_ctx.copy_buffer(a_dst, a_src, size=31, read_offset=11, write_offset=23)
    gl.copy_buffer(b_dst, b_src, size=31, read_offset=11, write_offset=23)
    assert a_dst.read() == b_dst.read()


def test_bind_and_assign_match_upstream(gl):
    a, b = both(gl, b"1234")
    assert a.bind("position", "color", layout="2f 3f1")[1:] == b.bind(
        "position", "color", layout="2f 3f1"
    )[1:]
    assert a.assign(4)[1:] == b.assign(4)[1:]


def test_label_extra_and_orphan():
    buf = mojo.create_context().buffer(b"abcdefgh")
    buf.label = "positions"
    buf.extra = {"owner": "mesh"}
    assert buf.label == "positions"
    assert buf.extra == {"owner": "mesh"}
    buf.orphan(17)
    assert buf.size == 17
    assert buf.read() == bytes(17)


def test_release_invalidates_buffer():
    buf = mojo.create_context().buffer(b"abcd")
    buf.release()
    with pytest.raises(mojo.InvalidObject):
        buf.read()


def test_validation_errors():
    ctx = mojo.create_context()
    with pytest.raises(mojo.Error):
        ctx.buffer()
    with pytest.raises(mojo.Error):
        ctx.buffer(b"abc", reserve=3)
    buf = ctx.buffer(b"abcdefgh")
    with pytest.raises(mojo.Error):
        buf.write(b"too long", offset=2)
    with pytest.raises(mojo.Error):
        buf.clear(5, chunk=b"xy")
    with pytest.raises(mojo.Error):
        buf.read_chunks(4, 6, 4, 2)
    with pytest.raises(NotImplementedError):
        buf.bind_to_uniform_block()
    with pytest.raises(NotImplementedError):
        buf.bind_to_storage_buffer()


def test_numpy_buffer_protocol_roundtrip():
    source = np.arange(20, dtype=np.float32)
    buf = mojo.create_context().buffer(source)
    target = np.empty_like(source)
    buf.read_into(target)
    assert np.array_equal(source, target)


def test_overlapping_ranges_are_safe():
    ctx = mojo.create_context()
    buf = ctx.buffer(bytes(range(32)))
    storage = buf._array()
    buf.write(storage[:16], offset=4)
    assert buf.read() == bytes(range(4)) + bytes(range(16)) + bytes(range(20, 32))

    ctx.copy_buffer(buf, buf, size=20, read_offset=0, write_offset=5)
    expected = bytearray(bytes(range(4)) + bytes(range(16)) + bytes(range(20, 32)))
    expected[5:25] = expected[:20]
    assert buf.read() == expected

    buf.read_into(storage, size=20, offset=0, write_offset=7)
    expected[7:27] = expected[:20]
    assert buf.read() == expected

    pattern = storage[12:16]
    expected_pattern = bytes(pattern)
    buf.clear(size=24, offset=0, chunk=pattern)
    assert buf.read(24) == expected_pattern * 6


def test_overlapping_chunk_destinations_are_deterministic():
    count = 300_000
    chunk_size = 16
    step = 8
    payload = np.arange(count * chunk_size, dtype=np.uint8)
    capacity = (count - 1) * step + chunk_size
    buf = mojo.create_context().buffer(reserve=capacity)
    buf.write_chunks(payload, 0, step, count)
    expected = np.zeros(capacity, dtype=np.uint8)
    for chunk in range(count):
        start = chunk * step
        expected[start : start + chunk_size] = payload[
            chunk * chunk_size : (chunk + 1) * chunk_size
        ]
    assert buf.read() == expected.tobytes()
