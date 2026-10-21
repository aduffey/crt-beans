import numpy as np
import taichi as ti
import taichi.math as tm
from imageio.v3 import imread, imwrite
from common import texelFetchRepeat, linear_to_srgb

phosphors = ti.Vector.field(n=3, dtype=float, shape=3)
phosphors[0] = tm.vec3(1.0, 0.0, 0.0)
phosphors[1] = tm.vec3(0.0, 1.0, 0.0)
phosphors[2] = tm.vec3(0.0, 0.0, 1.0)


def blend_mask(
    mask: np.ndarray,
    pixel_value: np.ndarray,
    mask_coverage: float,
    mitigation: float = 1.0,
) -> np.ndarray:
    """
    Blend the mask with the input pixel values, matching the logic of
    blend_mask in slang-shaders/shaders/mask.inc.

    Parameters:
        mask: Mask RGB values, shape (..., 3), in range [0, 1].
        pixel_value: Original image RGB values, shape (..., 3), in range [0, 1].
        mask_coverage: Inverse of the integral of the mask (e.g. 2.0, 3.0).
        mitigation: Level of brightness loss mitigation in [0.0, 1.0].
                    0.0: No brightness loss mitigation (pure mask multiplication).
                    1.0: Full brightness loss mitigation.

    Returns:
        Blended image array of same shape and float32 dtype.
    """
    if np.isclose(mask_coverage, 1.0) or mask_coverage <= 1.0:
        return pixel_value

    cov = np.float32(mask_coverage)
    mit = np.float32(mitigation)
    strength = (1.0 / cov) * (1.0 - mit) + mit
    v = pixel_value * strength

    a = np.minimum((v - 1.0) / (1.0 - cov), v)
    b = np.maximum((1.0 - cov * v) / (1.0 - cov), 0.0)
    return (cov * a * mask + b)


@ti.func
def blend_mask_taichi(
    mask: tm.vec3, pixel_value: tm.vec3, mask_coverage: float, mitigation: float
) -> tm.vec3:
    strength = tm.mix(1.0 / mask_coverage, 1.0, mitigation)
    v = pixel_value * strength
    a = tm.min((v - 1.0) / (1.0 - mask_coverage), v)
    b = tm.max((1.0 - mask_coverage * v) / (1.0 - mask_coverage), tm.vec3(0.0))
    return mask_coverage * a * mask + b


def generate_subpixel_mask(shape: tuple[int, ...], mask_pattern: int) -> tuple[np.ndarray, float]:
    if mask_pattern == 0:
        mask_tile = np.array([[1, 0, 1], [0, 1, 0]])  # <=1080p
        mask_coverage = 2.0
    elif mask_pattern == 1:
        mask_tile = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])  # 1080p/1440p
        mask_coverage = 3.0
    elif mask_pattern == 2:
        mask_tile = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1]])  # 1440p/4k
        mask_coverage = 2.0
    elif mask_pattern == 3:
        mask_tile = np.array([[1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0]])  # 4k, lower TVL
        mask_coverage = 15.0 / 6.0
    else:
        raise AssertionError('Unhandled mask pattern')
    # TODO ? For BGR subpixel arrangement, just transpose red and blue.
    out_height, out_width = shape[:2]
    mask = np.broadcast_to(mask_tile[np.arange(out_width) % mask_tile.shape[0]], (out_height, out_width, 3))
    return mask.astype(np.float32), float(mask_coverage)


def generate_coverage_mask(shape: tuple[int, ...], mask_triads: float) -> tuple[np.ndarray, float]:
    out_height, out_width = shape[:2]
    field_out = ti.Vector.field(n=3, dtype=float, shape=(out_height, out_width))
    coverage_mask_fragment(field_out, mask_triads)
    return field_out.to_numpy(), 3.0


@ti.kernel
def coverage_mask_fragment(field_out: ti.template(), mask_triads: float):
    (out_height, out_width) = field_out.shape
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = coverage_mask_taichi(vTexCoord, OutputSize, mask_triads)
    return


@ti.func
def coverage_mask_taichi(vTexCoord: tm.vec2, OutputSize: tm.vec4, mask_triads: float) -> tm.vec3:
    # Working in [0, mask_triads + 1) coordinate space to find the phosphor edges.
    x = vTexCoord.x * mask_triads
    phosphor_left_edge = tm.floor(x + tm.vec3(1.0 / 3.0, 0.0, -1.0 / 3.0)) + tm.vec3(0.0, 1.0 / 3.0, 2.0 / 3.0)
    # Working in [0, OutputSize.x + 1) coordinate space for the coverage.
    x = vTexCoord.x * OutputSize.x
    phosphor_left_edge = phosphor_left_edge / mask_triads * OutputSize.x
    phosphor_right_edge = phosphor_left_edge + OutputSize.x / mask_triads / 3.0
    mask = tm.clamp(min(x + 0.5, phosphor_right_edge) - max(x - 0.5, phosphor_left_edge), 0.0, 1.0)
    return mask


def generate_bandlimit_mask(shape: tuple[int, ...], mask_triads: float) -> tuple[np.ndarray, float]:
    out_height, out_width = shape[:2]
    field_out = ti.Vector.field(n=3, dtype=float, shape=(out_height, out_width))
    bandlimit_mask_fragment(field_out, mask_triads)
    return field_out.to_numpy(), 3.0


@ti.kernel
def bandlimit_mask_fragment(field_out: ti.template(), mask_triads: float):
    (out_height, out_width) = field_out.shape
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = bandlimit_mask_taichi2(vTexCoord, OutputSize, mask_triads)
    return


@ti.func
def bandlimit_mask_taichi2(vTexCoord: tm.vec2, OutputSize: tm.vec4, mask_triads: float) -> tm.vec3:
    # With unrolled loops and common terms collapsed for performance
    w = (mask_triads * 3.0) / OutputSize.x
    x = mask_triads * 3.0 * vTexCoord.x
    mask = tm.vec3(0.0)
    if w < 0.5:
        x1 = tm.clamp((x - tm.round(x)) / w, -1.0, 1.0)
        p0 = phosphors[int(tm.round(x) - 1.0) % 3]
        p1 = phosphors[int(tm.round(x)) % 3]
        mask = np.pi * (p0 + p1) + (p1 - p0) * (np.pi * x1 + tm.sin(np.pi * x1))
        mask /= 2.0 * np.pi
    elif w < 1.0:
        x1 = tm.clamp((x - tm.floor(x)) / w, -1.0, 1.0)
        x2 = tm.clamp((x - (tm.floor(x) + 1.0)) / w, -1.0, 1.0)
        p0 = phosphors[int(tm.floor(x) - 1.0) % 3]
        p1 = phosphors[int(tm.floor(x)) % 3]
        p2 = phosphors[int(tm.floor(x) + 1.0) % 3]
        mask = np.pi * (p0 + p2) + \
            (p1 - p0) * (np.pi * x1 + tm.sin(np.pi * x1)) + \
            (p2 - p1) * (np.pi * x2 + tm.sin(np.pi * x2))
        mask /= 2.0 * np.pi
    else:  # w < 1.5
        x1 = tm.clamp((x - (tm.round(x) - 1.0)) / w, -1.0, 1.0)
        x2 = tm.clamp((x - tm.round(x)) / w, -1.0, 1.0)
        x3 = tm.clamp((x - (tm.round(x) + 1.0)) / w, -1.0, 1.0)
        p0 = phosphors[int(tm.round(x) - 2.0) % 3]
        p1 = phosphors[int(tm.round(x) - 1.0) % 3]
        p2 = phosphors[int(tm.round(x)) % 3]
        p3 = phosphors[int(tm.round(x) + 1.0) % 3]
        mask = np.pi * (p0 + p3) + \
            (p1 - p0) * (np.pi * x1 + tm.sin(np.pi * x1)) + \
            (p2 - p1) * (np.pi * x2 + tm.sin(np.pi * x2)) + \
            (p3 - p2) * (np.pi * x3 + tm.sin(np.pi * x3))
        mask /= 2.0 * np.pi

    return tm.clamp(mask, 0.0, 1.0)


def generate_additive_mask(shape: tuple[int, ...], mask_triads: float) -> tuple[np.ndarray, float]:
    out_height, out_width = shape[:2]
    field_out = ti.Vector.field(n=3, dtype=float, shape=(out_height, out_width))
    additive_mask_fragment(field_out, mask_triads)

    if mask_triads * 4 < out_width:
        mask_coverage = 3.0
    elif mask_triads * 2 < out_width:
        mask_coverage = 2.0
    else:
        mask_coverage = 1.0

    return field_out.to_numpy(), mask_coverage


@ti.kernel
def additive_mask_fragment(field_out: ti.template(), mask_triads: float):
    (out_height, out_width) = field_out.shape
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = additive_mask_taichi(vTexCoord, OutputSize, mask_triads)
    return


@ti.func
def additive_mask_taichi(vTexCoord: tm.vec2, OutputSize: tm.vec4, mask_triads: float) -> tm.vec3:
    offset = tm.vec3(0.0, -1 / (3 * mask_triads), -2 / (3 * mask_triads))
    half = OutputSize.z * 0.5
    x = vTexCoord.x
    mask = tm.vec3(1.0, 1.0, 1.0)
    if mask_triads * 4 < OutputSize.x:  # XXX
        # Area under offset
        mask = (1.0 / 3.0 + 0.080163) * OutputSize.z
        # Area under first harmonic
        right = tm.sin(2 * np.pi * mask_triads * (x + offset + half))
        left = tm.sin(2 * np.pi * mask_triads * (x + offset - half))
        mask += 0.5513288954217920 * (right - left) / (2 * np.pi * mask_triads)
        # Area under second harmonic
        right = tm.sin(4 * np.pi * mask_triads * (x + offset + half))
        left = tm.sin(4 * np.pi * mask_triads * (x + offset - half))
        mask += 0.2756644477108960 * (right - left) / (4 * np.pi * mask_triads)
        mask /= (1.24049 * OutputSize.z)
    elif mask_triads * 2 < OutputSize.x:
        mask = 0.5 * OutputSize.z + 0.5 * \
                (tm.sin(2 * np.pi * mask_triads * (x + offset + half)) - \
                tm.sin(2 * np.pi * mask_triads * (x + offset - half))) / (2 * np.pi * mask_triads)
        mask /= OutputSize.z

    return tm.clamp(mask, 0.0, 1.0)


def generate_additive2_mask(shape: tuple[int, ...], mask_triads: float) -> tuple[np.ndarray, float]:
    out_height, out_width = shape[:2]
    field_out = ti.Vector.field(n=3, dtype=float, shape=(out_height, out_width))
    additive2_mask_fragment(field_out, mask_triads)

    def smoothstep(a, b, x):
        # Scale, bias and saturate x to 0..1 range
        x = np.clip((x - a) / (b - a), 0.0, 1.0)
        # Evaluate polynomial
        return x * x * (3.0 - 2.0 * x)

    mask_coverage = 1.0 / (0.5 - smoothstep(4.0, 5.0, (out_width / mask_triads)) / 8.0)

    return field_out.to_numpy(), mask_coverage


@ti.kernel
def additive2_mask_fragment(field_out: ti.template(), mask_triads: float):
    (out_height, out_width) = field_out.shape
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = additive2_mask_taichi(vTexCoord, OutputSize, mask_triads)
    return


@ti.func
def additive2_mask_taichi(vTexCoord: tm.vec2, OutputSize: tm.vec4, mask_triads: float) -> tm.vec3:
    x = vTexCoord.x

    P = 1.0 / mask_triads
    phase = 2.0 * tm.pi * x / P

    cs = tm.cos(phase)
    sn = tm.sin(phase)

    c1 = tm.vec3(
        cs,
        -0.5 * cs + 0.86602540378 * sn,
        -0.5 * cs - 0.86602540378 * sn
    );

    # cos(2*theta) = 2*cos(theta)^2 - 1
    c2 = 2.0 * c1 * c1 - 1.0

    # Enable the second harmonic as it becomes sufficiently separated from Nyquist.
    t = tm.smoothstep(4.0, 5.0, (OutputSize.x / mask_triads))

    # Pixel integration attenuates each harmonic.
    a1 = 0.5 * tm.sin(tm.pi * 1.0 / (OutputSize.x / mask_triads)) / (tm.pi * 1.0 / (OutputSize.x / mask_triads))
    a2 = 0.125 * t * tm.sin(tm.pi * 2.0 / (OutputSize.x / mask_triads)) / (tm.pi * 2.0 / (OutputSize.x / mask_triads))

    return tm.vec3(0.5 - 0.125 * t) + a1 * c1 + a2 * c2;


def generate_tiled_mask(shape: tuple[int, ...], mask_type: str, mask_triads: float) -> tuple[np.ndarray, float]:
    if mask_type == 'aperture':
        mask_tile = imread('mask-aperture.png')[:, :, 0:3].astype(np.float32) / 255.0
    elif mask_type == 'slot':
        mask_tile = imread('mask-slot.png')[:, :, 0:3].astype(np.float32) / 255.0
    else:
        raise AssertionError(f'Unhandled mask type: {mask_type}')
    mask_coverage = float(1.0 / np.average(mask_tile))
    out_height, out_width = shape[:2]
    mask = generate_mask(mask_tile, (out_height, out_width, 3), mask_triads)
    return mask.astype(np.float32), mask_coverage


def generate_mask(mask, out_shape, triads: float):
    (mask_height, mask_width, mask_planes) = mask.shape
    (out_height, out_width, out_planes) = out_shape
    scale = mask_width / 2 * triads / out_width
    print("scale: {}".format(scale))
    field_in = ti.Vector.field(n=3, dtype=float, shape=(mask_height, mask_width))
    field_in.from_numpy(mask)
    field_out = ti.Vector.field(n=3, dtype=float, shape=(out_width, mask_height))
    lanczos3_downscale(field_in, field_out, scale)
    field_in = field_out
    field_out = ti.Vector.field(n=3, dtype=float, shape=(out_height, out_width))
    lanczos3_downscale(field_in, field_out, scale)
    return field_out.to_numpy()


@ti.kernel
def lanczos3_downscale(field_in: ti.template(), field_out: ti.template(), scale: float):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = lanczos3_taichi(vTexCoord, field_in, SourceSize, OutputSize, scale)
    return


@ti.func
def lanczos3_taichi(vTexCoord: tm.vec2, Source, SourceSize: tm.vec4, OutputSize: tm.vec4, scale: float) -> tm.vec3:
    kernel_size = 3  # 1, 2, or 3
    x_pos = vTexCoord.y * OutputSize.y * scale
    y_pos = int(tm.floor(vTexCoord.x * SourceSize.y))
    weight_sum = 0.0
    value = tm.vec3(0.0)
    for x in range(int(tm.round(x_pos - kernel_size * scale)), int(tm.round(x_pos + kernel_size * scale))):  # TODO bounds? round to int
        distance_x = (x_pos - x - 0.5) / scale
        assert abs(distance_x) <= kernel_size
        weight = 1.0 if distance_x == 0.0 else \
            (kernel_size * tm.sin(np.pi * distance_x) * tm.sin(np.pi * distance_x / kernel_size)) / (np.pi * np.pi * distance_x * distance_x)
        weight_sum += weight
        value += weight * texelFetchRepeat(Source, tm.ivec2(x, y_pos))
    return value / weight_sum


def apply_mask(
    img: np.ndarray,
    mask_type: str,
    mask_pattern: int = 1,
    tiled_mask_type: str = 'slot',
    mask_triads: float = 550.0,
    mask_amount: float = 1.0,
    *,
    mask_mitigation: float = None,
) -> np.ndarray:
    if mask_type in (None, 'none', 'disabled', False):
        return img

    mitigation = mask_mitigation if mask_mitigation is not None else mask_amount
    shape = img.shape[:2]

    if mask_type == 'subpixel':
        mask, coverage = generate_subpixel_mask(shape, mask_pattern)
    elif mask_type == 'bandlimit':
        mask, coverage = generate_bandlimit_mask(shape, mask_triads)
    elif mask_type == 'coverage':
        mask, coverage = generate_coverage_mask(shape, mask_triads)
    elif mask_type == 'additive':
        mask, coverage = generate_additive_mask(shape, mask_triads)
    elif mask_type == 'additive2':
        mask, coverage = generate_additive2_mask(shape, mask_triads)
    elif mask_type == 'tiled':
        mask, coverage = generate_tiled_mask(shape, tiled_mask_type, mask_triads)
    else:
        raise ValueError(
            f"Unknown mask type: {mask_type}. Supported types: 'subpixel', 'bandlimit', 'coverage', 'additive', 'tiled', 'none'."
        )

    return blend_mask(mask, img, coverage, mitigation)
