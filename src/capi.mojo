"""C ABI for CPU-resident ModernGL buffer and vertex helpers."""

from std.math import sqrt
from std.sys.info import simd_width_of

comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime F32Ptr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()
comptime BYTE_WIDTH = W * 8


def copy_bytes(src: U8Ptr, dst: U8Ptr, n: Int):
    var i = 0
    while i + BYTE_WIDTH <= n:
        dst.store(i, src.load[width=BYTE_WIDTH](i))
        i += BYTE_WIDTH
    while i < n:
        dst[i] = src[i]
        i += 1


def fill_pattern(dst: U8Ptr, n: Int, pattern: U8Ptr, pattern_size: Int):
    if pattern_size == 1:
        var value = SIMD[DType.uint8, BYTE_WIDTH](pattern[0])
        var i = 0
        while i + BYTE_WIDTH <= n:
            dst.store(i, value)
            i += BYTE_WIDTH
        while i < n:
            dst[i] = pattern[0]
            i += 1
        return
    copy_bytes(pattern, dst, pattern_size)
    var filled = pattern_size
    while filled * 2 <= n:
        copy_bytes(dst, dst + filled, filled)
        filled *= 2
    if filled < n:
        copy_bytes(dst, dst + filled, n - filled)


def gather_chunk_range(
    src: U8Ptr,
    dst: U8Ptr,
    chunk_size: Int,
    start: Int,
    step: Int,
    begin: Int,
    end: Int,
):
    if chunk_size == 16:
        var chunk = begin
        while chunk + 2 <= end:
            var source_offset = start + chunk * step
            var first = src.load[width=16](source_offset)
            var second = src.load[width=16](source_offset + step)
            dst.store(chunk * 16, first.join(second))
            chunk += 2
        while chunk < end:
            var source_offset = start + chunk * step
            dst.store(chunk * 16, src.load[width=16](source_offset))
            chunk += 1
        return
    if chunk_size == 32:
        for chunk in range(begin, end):
            var source_offset = start + chunk * step
            dst.store(chunk * 32, src.load[width=32](source_offset))
        return
    for chunk in range(begin, end):
        copy_bytes(src + start + chunk * step, dst + chunk * chunk_size, chunk_size)


def gather_chunks(
    src: U8Ptr,
    dst: U8Ptr,
    chunk_size: Int,
    start: Int,
    step: Int,
    count: Int,
):
    gather_chunk_range(src, dst, chunk_size, start, step, 0, count)


def scatter_chunk_range(
    src: U8Ptr,
    dst: U8Ptr,
    chunk_size: Int,
    start: Int,
    step: Int,
    begin: Int,
    end: Int,
):
    if chunk_size == 16:
        var chunk = begin
        while chunk + 2 <= end:
            var dest_offset = start + chunk * step
            var pair = src.load[width=32](chunk * 16)
            dst.store(dest_offset, pair.slice[16]())
            dst.store(dest_offset + step, pair.slice[16, offset=16]())
            chunk += 2
        while chunk < end:
            var dest_offset = start + chunk * step
            dst.store(dest_offset, src.load[width=16](chunk * 16))
            chunk += 1
        return
    if chunk_size == 32:
        for chunk in range(begin, end):
            var dest_offset = start + chunk * step
            dst.store(dest_offset, src.load[width=32](chunk * 32))
        return
    for chunk in range(begin, end):
        copy_bytes(src + chunk * chunk_size, dst + start + chunk * step, chunk_size)


def scatter_chunks(
    src: U8Ptr,
    dst: U8Ptr,
    chunk_size: Int,
    start: Int,
    step: Int,
    count: Int,
):
    scatter_chunk_range(src, dst, chunk_size, start, step, 0, count)


def transform_f32_range(
    src: F32Ptr,
    matrix: F32Ptr,
    dst: F32Ptr,
    begin: Int,
    end: Int,
    components: Int,
    default_w: Float32,
    divide: Bool,
):
    var i = begin
    if components == 3 and not divide:
        var m0 = SIMD[DType.float32, W](matrix[0])
        var m1 = SIMD[DType.float32, W](matrix[1])
        var m2 = SIMD[DType.float32, W](matrix[2])
        var m3 = SIMD[DType.float32, W](matrix[3] * default_w)
        var m4 = SIMD[DType.float32, W](matrix[4])
        var m5 = SIMD[DType.float32, W](matrix[5])
        var m6 = SIMD[DType.float32, W](matrix[6])
        var m7 = SIMD[DType.float32, W](matrix[7] * default_w)
        var m8 = SIMD[DType.float32, W](matrix[8])
        var m9 = SIMD[DType.float32, W](matrix[9])
        var m10 = SIMD[DType.float32, W](matrix[10])
        var m11 = SIMD[DType.float32, W](matrix[11] * default_w)
        comptime if W == 4:
            while i + W <= end:
                var base = i * 3
                var a = src.load[width=W](base)
                var b = src.load[width=W](base + W)
                var c = src.load[width=W](base + 2 * W)
                var x = SIMD[DType.float32, W](a[0], a[3], b[2], c[1])
                var y = SIMD[DType.float32, W](a[1], b[0], b[3], c[2])
                var z = SIMD[DType.float32, W](a[2], b[1], c[0], c[3])
                var tx = m0 * x + m1 * y + m2 * z + m3
                var ty = m4 * x + m5 * y + m6 * z + m7
                var tz = m8 * x + m9 * y + m10 * z + m11
                dst.store(
                    base,
                    SIMD[DType.float32, W](tx[0], ty[0], tz[0], tx[1]),
                )
                dst.store(
                    base + W,
                    SIMD[DType.float32, W](ty[1], tz[1], tx[2], ty[2]),
                )
                dst.store(
                    base + 2 * W,
                    SIMD[DType.float32, W](tz[2], tx[3], ty[3], tz[3]),
                )
                i += W
        else:
            while i + W <= end:
                var base = i * 3
                var x = (src + base).strided_load[width=W](3)
                var y = (src + base + 1).strided_load[width=W](3)
                var z = (src + base + 2).strided_load[width=W](3)
                (dst + base).strided_store(
                    m0 * x + m1 * y + m2 * z + m3, 3
                )
                (dst + base + 1).strided_store(
                    m4 * x + m5 * y + m6 * z + m7, 3
                )
                (dst + base + 2).strided_store(
                    m8 * x + m9 * y + m10 * z + m11, 3
                )
                i += W

    while i < end:
        var base = i * components
        var x = src[base]
        var y = src[base + 1]
        var z = Float32(0.0)
        var w = default_w
        if components >= 3:
            z = src[base + 2]
        if components == 4:
            w = src[base + 3]
        var tx = matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3] * w
        var ty = matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7] * w
        var tz = matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11] * w
        var tw = matrix[12] * x + matrix[13] * y + matrix[14] * z + matrix[15] * w
        if divide and tw != 0.0:
            tx /= tw
            ty /= tw
            tz /= tw
            tw = 1.0
        dst[base] = tx
        dst[base + 1] = ty
        if components >= 3:
            dst[base + 2] = tz
        if components == 4:
            dst[base + 3] = tw
        i += 1


def transform_f32(
    src: F32Ptr,
    matrix: F32Ptr,
    dst: F32Ptr,
    n: Int,
    components: Int,
    default_w: Float32,
    divide: Bool,
):
    transform_f32_range(
        src, matrix, dst, 0, n, components, default_w, divide
    )


def transform_f64_range(
    src: F64Ptr,
    matrix: F64Ptr,
    dst: F64Ptr,
    begin: Int,
    end: Int,
    components: Int,
    default_w: Float64,
    divide: Bool,
):
    var i = begin
    if components == 3 and not divide:
        var m0 = SIMD[DType.float64, W](matrix[0])
        var m1 = SIMD[DType.float64, W](matrix[1])
        var m2 = SIMD[DType.float64, W](matrix[2])
        var m3 = SIMD[DType.float64, W](matrix[3] * default_w)
        var m4 = SIMD[DType.float64, W](matrix[4])
        var m5 = SIMD[DType.float64, W](matrix[5])
        var m6 = SIMD[DType.float64, W](matrix[6])
        var m7 = SIMD[DType.float64, W](matrix[7] * default_w)
        var m8 = SIMD[DType.float64, W](matrix[8])
        var m9 = SIMD[DType.float64, W](matrix[9])
        var m10 = SIMD[DType.float64, W](matrix[10])
        var m11 = SIMD[DType.float64, W](matrix[11] * default_w)
        while i + W <= end:
            var base = i * 3
            var x = (src + base).strided_load[width=W](3)
            var y = (src + base + 1).strided_load[width=W](3)
            var z = (src + base + 2).strided_load[width=W](3)
            (dst + base).strided_store(
                m0 * x + m1 * y + m2 * z + m3, 3
            )
            (dst + base + 1).strided_store(
                m4 * x + m5 * y + m6 * z + m7, 3
            )
            (dst + base + 2).strided_store(
                m8 * x + m9 * y + m10 * z + m11, 3
            )
            i += W

    while i < end:
        var base = i * components
        var x = src[base]
        var y = src[base + 1]
        var z = 0.0
        var w = default_w
        if components >= 3:
            z = src[base + 2]
        if components == 4:
            w = src[base + 3]
        var tx = matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3] * w
        var ty = matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7] * w
        var tz = matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11] * w
        var tw = matrix[12] * x + matrix[13] * y + matrix[14] * z + matrix[15] * w
        if divide and tw != 0.0:
            tx /= tw
            ty /= tw
            tz /= tw
            tw = 1.0
        dst[base] = tx
        dst[base + 1] = ty
        if components >= 3:
            dst[base + 2] = tz
        if components == 4:
            dst[base + 3] = tw
        i += 1


def transform_f64(
    src: F64Ptr,
    matrix: F64Ptr,
    dst: F64Ptr,
    n: Int,
    components: Int,
    default_w: Float64,
    divide: Bool,
):
    transform_f64_range(
        src, matrix, dst, 0, n, components, default_w, divide
    )


def transform_normals_f32(
    src: F32Ptr, matrix: F32Ptr, dst: F32Ptr, n: Int, do_normalize: Bool
):
    for i in range(n):
        var base = i * 3
        var x = src[base]
        var y = src[base + 1]
        var z = src[base + 2]
        var tx = matrix[0] * x + matrix[1] * y + matrix[2] * z
        var ty = matrix[3] * x + matrix[4] * y + matrix[5] * z
        var tz = matrix[6] * x + matrix[7] * y + matrix[8] * z
        if do_normalize:
            var length = sqrt(tx * tx + ty * ty + tz * tz)
            if length != 0.0:
                tx /= length
                ty /= length
                tz /= length
        dst[base] = tx
        dst[base + 1] = ty
        dst[base + 2] = tz


def transform_normals_f64(
    src: F64Ptr, matrix: F64Ptr, dst: F64Ptr, n: Int, do_normalize: Bool
):
    for i in range(n):
        var base = i * 3
        var x = src[base]
        var y = src[base + 1]
        var z = src[base + 2]
        var tx = matrix[0] * x + matrix[1] * y + matrix[2] * z
        var ty = matrix[3] * x + matrix[4] * y + matrix[5] * z
        var tz = matrix[6] * x + matrix[7] * y + matrix[8] * z
        if do_normalize:
            var length = sqrt(tx * tx + ty * ty + tz * tz)
            if length != 0.0:
                tx /= length
                ty /= length
                tz /= length
        dst[base] = tx
        dst[base + 1] = ty
        dst[base + 2] = tz


def normalize_f32(src: F32Ptr, dst: F32Ptr, n: Int, components: Int):
    for i in range(n):
        var base = i * components
        var length2 = Float32(0.0)
        for j in range(components):
            var value = src[base + j]
            length2 += value * value
        var length = sqrt(length2)
        var inverse = Float32(1.0)
        if length != 0.0:
            inverse = 1.0 / length
        for j in range(components):
            dst[base + j] = src[base + j] * inverse


def normalize_f64(src: F64Ptr, dst: F64Ptr, n: Int, components: Int):
    for i in range(n):
        var base = i * components
        var length2 = 0.0
        for j in range(components):
            var value = src[base + j]
            length2 += value * value
        var length = sqrt(length2)
        var inverse = 1.0
        if length != 0.0:
            inverse = 1.0 / length
        for j in range(components):
            dst[base + j] = src[base + j] * inverse


def bounds_f32(src: F32Ptr, dst: F32Ptr, n: Int, components: Int):
    for j in range(components):
        dst[j] = src[j]
        dst[components + j] = src[j]
    for i in range(1, n):
        var base = i * components
        for j in range(components):
            var value = src[base + j]
            if value < dst[j]:
                dst[j] = value
            if value > dst[components + j]:
                dst[components + j] = value


def bounds_f64(src: F64Ptr, dst: F64Ptr, n: Int, components: Int):
    for j in range(components):
        dst[j] = src[j]
        dst[components + j] = src[j]
    for i in range(1, n):
        var base = i * components
        for j in range(components):
            var value = src[base + j]
            if value < dst[j]:
                dst[j] = value
            if value > dst[components + j]:
                dst[components + j] = value


def deindex_f32(
    src: F32Ptr, indices: I64Ptr, dst: F32Ptr, count: Int, components: Int
):
    for i in range(count):
        var source_base = Int(indices[i]) * components
        var dest_base = i * components
        for j in range(components):
            dst[dest_base + j] = src[source_base + j]


def deindex_f64(
    src: F64Ptr, indices: I64Ptr, dst: F64Ptr, count: Int, components: Int
):
    for i in range(count):
        var source_base = Int(indices[i]) * components
        var dest_base = i * components
        for j in range(components):
            dst[dest_base + j] = src[source_base + j]


def compute_normals_f32(
    vertices: F32Ptr,
    indices: I64Ptr,
    dst: F32Ptr,
    vertex_count: Int,
    index_count: Int,
    do_normalize: Bool,
):
    for i in range(vertex_count * 3):
        dst[i] = 0.0
    for face in range(index_count // 3):
        var ia = Int(indices[face * 3])
        var ib = Int(indices[face * 3 + 1])
        var ic = Int(indices[face * 3 + 2])
        var ax = vertices[ia * 3]
        var ay = vertices[ia * 3 + 1]
        var az = vertices[ia * 3 + 2]
        var ux = vertices[ib * 3] - ax
        var uy = vertices[ib * 3 + 1] - ay
        var uz = vertices[ib * 3 + 2] - az
        var vx = vertices[ic * 3] - ax
        var vy = vertices[ic * 3 + 1] - ay
        var vz = vertices[ic * 3 + 2] - az
        var nx = uy * vz - uz * vy
        var ny = uz * vx - ux * vz
        var nz = ux * vy - uy * vx
        for index in range(3):
            var vertex = Int(indices[face * 3 + index]) * 3
            dst[vertex] += nx
            dst[vertex + 1] += ny
            dst[vertex + 2] += nz
    if do_normalize:
        normalize_f32(dst, dst, vertex_count, 3)


def compute_normals_f64(
    vertices: F64Ptr,
    indices: I64Ptr,
    dst: F64Ptr,
    vertex_count: Int,
    index_count: Int,
    do_normalize: Bool,
):
    for i in range(vertex_count * 3):
        dst[i] = 0.0
    for face in range(index_count // 3):
        var ia = Int(indices[face * 3])
        var ib = Int(indices[face * 3 + 1])
        var ic = Int(indices[face * 3 + 2])
        var ax = vertices[ia * 3]
        var ay = vertices[ia * 3 + 1]
        var az = vertices[ia * 3 + 2]
        var ux = vertices[ib * 3] - ax
        var uy = vertices[ib * 3 + 1] - ay
        var uz = vertices[ib * 3 + 2] - az
        var vx = vertices[ic * 3] - ax
        var vy = vertices[ic * 3 + 1] - ay
        var vz = vertices[ic * 3 + 2] - az
        var nx = uy * vz - uz * vy
        var ny = uz * vx - ux * vz
        var nz = ux * vy - uy * vx
        for index in range(3):
            var vertex = Int(indices[face * 3 + index]) * 3
            dst[vertex] += nx
            dst[vertex + 1] += ny
            dst[vertex + 2] += nz
    if do_normalize:
        normalize_f64(dst, dst, vertex_count, 3)


@export("mmgl_copy_bytes")
def mmgl_copy_bytes(src: Int, dst: Int, n: Int) abi("C"):
    copy_bytes(
        U8Ptr(unsafe_from_address=src), U8Ptr(unsafe_from_address=dst), n
    )


@export("mmgl_fill_pattern")
def mmgl_fill_pattern(
    dst: Int, n: Int, pattern: Int, pattern_size: Int
) abi("C"):
    fill_pattern(
        U8Ptr(unsafe_from_address=dst),
        n,
        U8Ptr(unsafe_from_address=pattern),
        pattern_size,
    )


@export("mmgl_gather_chunks")
def mmgl_gather_chunks(
    src: Int,
    dst: Int,
    chunk_size: Int,
    start: Int,
    step: Int,
    count: Int,
) abi("C"):
    gather_chunks(
        U8Ptr(unsafe_from_address=src),
        U8Ptr(unsafe_from_address=dst),
        chunk_size,
        start,
        step,
        count,
    )


@export("mmgl_gather_chunk_range")
def mmgl_gather_chunk_range(
    src: Int,
    dst: Int,
    chunk_size: Int,
    start: Int,
    step: Int,
    begin: Int,
    end: Int,
) abi("C"):
    gather_chunk_range(
        U8Ptr(unsafe_from_address=src),
        U8Ptr(unsafe_from_address=dst),
        chunk_size,
        start,
        step,
        begin,
        end,
    )


@export("mmgl_scatter_chunks")
def mmgl_scatter_chunks(
    src: Int,
    dst: Int,
    chunk_size: Int,
    start: Int,
    step: Int,
    count: Int,
) abi("C"):
    scatter_chunks(
        U8Ptr(unsafe_from_address=src),
        U8Ptr(unsafe_from_address=dst),
        chunk_size,
        start,
        step,
        count,
    )


@export("mmgl_transform_f32")
def mmgl_transform_f32(
    src: Int,
    matrix: Int,
    dst: Int,
    n: Int,
    components: Int,
    default_w: Float32,
    divide: Int,
) abi("C"):
    transform_f32(
        F32Ptr(unsafe_from_address=src),
        F32Ptr(unsafe_from_address=matrix),
        F32Ptr(unsafe_from_address=dst),
        n,
        components,
        default_w,
        divide != 0,
    )


@export("mmgl_transform_f64")
def mmgl_transform_f64(
    src: Int,
    matrix: Int,
    dst: Int,
    n: Int,
    components: Int,
    default_w: Float64,
    divide: Int,
) abi("C"):
    transform_f64(
        F64Ptr(unsafe_from_address=src),
        F64Ptr(unsafe_from_address=matrix),
        F64Ptr(unsafe_from_address=dst),
        n,
        components,
        default_w,
        divide != 0,
    )


@export("mmgl_transform_normals_f32")
def mmgl_transform_normals_f32(
    src: Int, matrix: Int, dst: Int, n: Int, do_normalize: Int
) abi("C"):
    transform_normals_f32(
        F32Ptr(unsafe_from_address=src),
        F32Ptr(unsafe_from_address=matrix),
        F32Ptr(unsafe_from_address=dst),
        n,
        do_normalize != 0,
    )


@export("mmgl_transform_normals_f64")
def mmgl_transform_normals_f64(
    src: Int, matrix: Int, dst: Int, n: Int, do_normalize: Int
) abi("C"):
    transform_normals_f64(
        F64Ptr(unsafe_from_address=src),
        F64Ptr(unsafe_from_address=matrix),
        F64Ptr(unsafe_from_address=dst),
        n,
        do_normalize != 0,
    )


@export("mmgl_normalize_f32")
def mmgl_normalize_f32(
    src: Int, dst: Int, n: Int, components: Int
) abi("C"):
    normalize_f32(
        F32Ptr(unsafe_from_address=src),
        F32Ptr(unsafe_from_address=dst),
        n,
        components,
    )


@export("mmgl_normalize_f64")
def mmgl_normalize_f64(
    src: Int, dst: Int, n: Int, components: Int
) abi("C"):
    normalize_f64(
        F64Ptr(unsafe_from_address=src),
        F64Ptr(unsafe_from_address=dst),
        n,
        components,
    )


@export("mmgl_bounds_f32")
def mmgl_bounds_f32(
    src: Int, dst: Int, n: Int, components: Int
) abi("C"):
    bounds_f32(
        F32Ptr(unsafe_from_address=src),
        F32Ptr(unsafe_from_address=dst),
        n,
        components,
    )


@export("mmgl_bounds_f64")
def mmgl_bounds_f64(
    src: Int, dst: Int, n: Int, components: Int
) abi("C"):
    bounds_f64(
        F64Ptr(unsafe_from_address=src),
        F64Ptr(unsafe_from_address=dst),
        n,
        components,
    )


@export("mmgl_deindex_f32")
def mmgl_deindex_f32(
    src: Int, indices: Int, dst: Int, count: Int, components: Int
) abi("C"):
    deindex_f32(
        F32Ptr(unsafe_from_address=src),
        I64Ptr(unsafe_from_address=indices),
        F32Ptr(unsafe_from_address=dst),
        count,
        components,
    )


@export("mmgl_deindex_f64")
def mmgl_deindex_f64(
    src: Int, indices: Int, dst: Int, count: Int, components: Int
) abi("C"):
    deindex_f64(
        F64Ptr(unsafe_from_address=src),
        I64Ptr(unsafe_from_address=indices),
        F64Ptr(unsafe_from_address=dst),
        count,
        components,
    )


@export("mmgl_compute_normals_f32")
def mmgl_compute_normals_f32(
    vertices: Int,
    indices: Int,
    dst: Int,
    vertex_count: Int,
    index_count: Int,
    do_normalize: Int,
) abi("C"):
    compute_normals_f32(
        F32Ptr(unsafe_from_address=vertices),
        I64Ptr(unsafe_from_address=indices),
        F32Ptr(unsafe_from_address=dst),
        vertex_count,
        index_count,
        do_normalize != 0,
    )


@export("mmgl_compute_normals_f64")
def mmgl_compute_normals_f64(
    vertices: Int,
    indices: Int,
    dst: Int,
    vertex_count: Int,
    index_count: Int,
    do_normalize: Int,
) abi("C"):
    compute_normals_f64(
        F64Ptr(unsafe_from_address=vertices),
        I64Ptr(unsafe_from_address=indices),
        F64Ptr(unsafe_from_address=dst),
        vertex_count,
        index_count,
        do_normalize != 0,
    )
