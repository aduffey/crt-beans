from imageio.v3 import imwrite, imread
import numpy as np
import argparse
import functools
import taichi as ti
import taichi.math as tm


ti.init()


def srgb_to_gamma(img, out_gamma):
    """sRGB uint8 to simple gamma float"""
    out = img.astype(np.float32) / 255
    out = np.where(out <= 0.04045, out / 12.92, np.power((out + 0.055) / 1.055, 2.4))
    out = np.power(out, 1.0 / out_gamma)
    return out


def gamma_to_gamma(img, in_gamma, out_gamma):
    return np.power(np.power(np.clip(img, 0.0, 1.0), in_gamma), 1.0 / out_gamma)


def gamma_to_yiq(img):
    """Native gamma to YIQ float"""
    rgb2yiq = np.array([[0.30, 0.59, 0.11],
                        [0.599, -0.2773, -0.3217],
                        [0.213, -0.5251, 0.3121]])
    return np.dot(img, rgb2yiq.T.copy())


def gamma_to_linear(img, in_gamma):
    """Simple gamma float to linear float"""
    return np.power(np.clip(img, 0.0, 1.0), in_gamma)


def yiq_to_linear(img, in_gamma):
    """YIQ float to linear float"""
    yiq2rgb = np.linalg.inv(np.array([[0.30, 0.59, 0.11],
                                      [0.599, -0.2773, -0.3217],
                                      [0.213, -0.5251, 0.3121]]))
    out = np.dot(img, yiq2rgb.T.copy())
    out = np.clip(out, 0.0, 1.0)
    out = np.power(out, in_gamma)
    return out


def linear_to_srgb(img):
    """Linear float to sRGB uint8"""
    img_clipped = np.clip(img, 0.0, 1.0)
    out = np.where(img_clipped <= 0.0031308, img_clipped * 12.92, 1.055 * (np.power(img_clipped, (1.0 / 2.4))) - 0.055)
    out = np.around(out * 255).astype(np.uint8)
    return out


@ti.func
def texelFetch(Source, vTexCoords: tm.ivec2) -> tm.vec3:
    """Fetch pixel from img at (x, y), or 0 if outside range"""
    y_size, x_size = Source.shape
    val = tm.vec3(0.0)
    if not (vTexCoords.x < 0 or vTexCoords.x >= x_size or vTexCoords.y < 0 or vTexCoords.y >= y_size):
        val = Source[vTexCoords.y, vTexCoords.x]
    return val


@ti.func
def texelFetchRepeat(Source, vTexCoords: tm.ivec2) -> tm.vec3:
    """Fetch pixel from img at (x, y), with the texture repeated infinitely on both axes"""
    y_size, x_size = Source.shape
    return Source[vTexCoords.y % y_size, vTexCoords.x % x_size]


@ti.func
def texture(Source, vTexCoords: tm.vec2) -> tm.vec3:
    """Sample from Source at (x, y) using bilinear interpolation.

    Outside of the Source is considered to be 0.
    """
    lookup_coords = vTexCoords.x * Source.shape
    coords = tm.round(lookup_coords, tm.ivec2)
    v11 = texelFetch(Source, coords)
    v01 = texelFetch(Source, coords - ivec2(1, 0))
    v10 = texelFetch(Source, coords - ivec2(0, 1))
    v00 = texelFetch(Source, coords - ivec2(1, 1))
    row1 = tm.mix(v10, v11, lookup_coords.y - coords.y + 0.5)
    row0 = tm.mix(v01, v00, lookup_coords.y - coords.y + 0.5)
    return tm.mix(row0, row1, lookup_coords.x - coords.x + 0.5)


def filter_fragment(image_in, output_dim):
    (in_height, in_width, in_planes) = image_in.shape
    (out_height, out_width, out_planes) = output_dim
    field_in = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_in.from_numpy(image_in)
    field_out = ti.Vector.field(n=3, dtype=float, shape=(out_height, out_width))
    output_field = taichi_filter_fragment(field_in, field_out)
    return field_out.to_numpy()


@ti.kernel
def taichi_filter_fragment(field_in: ti.template(), field_out: ti.template()):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = filter_sim2(vTexCoord, field_in, SourceSize)
    return


@ti.func
def filter_sim(vTexCoord: tm.vec2, Source, SourceSize: tm.vec4) -> tm.vec3:
    max_L = tm.max(tm.max(L.r, L.g), L.b)
    L_rcp = 1.0 / L

    filtered = tm.vec3(0.0)
    pix_y = int(tm.floor(vTexCoord.y * SourceSize.y))
    t = vTexCoord.x
    for pix_x in range(int(tm.floor(SourceSize.x * (vTexCoord.x - max_L))),
                       int(tm.floor(SourceSize.x * (vTexCoord.x + max_L))) + 1):
        s = texelFetch(Source, tm.ivec2(pix_x, pix_y))
        t0 = tm.vec3(pix_x * SourceSize.z)
        t1 = t0 + tm.vec3(SourceSize.z)
        t0 = tm.clamp(t0, t - L, t + L)
        t1 = tm.clamp(t1, t - L, t + L)
        # Integral of s * (1 / L) * (0.5 + 0.5 * cos(PI * (t - t_x) / L)) dt_x over t0 to t1
        filtered += 0.5 * s * L_rcp * (t1 - t0 + (L / np.pi) *
                                       (tm.sin(L_rcp * ((np.pi * t) - np.pi * t0)) - tm.sin(L_rcp * ((np.pi * t) - np.pi * t1))))
    return filtered


@ti.func
def filter_sim2(vTexCoord: tm.vec2, Source, SourceSize: tm.vec4) -> tm.vec3:
    max_L = tm.max(tm.max(L.r, L.g), L.b)

    filtered = tm.vec3(0.0)
    pix_y = int(tm.floor(vTexCoord.y * SourceSize.y))
    t = vTexCoord.x

    # Set up loop bounds
    start = int(tm.floor(SourceSize.x * (vTexCoord.x - max_L)))
    end = int(tm.floor(SourceSize.x * (vTexCoord.x + max_L))) + 1

    # Set up the first common term for the left. The right side of the integral
    # for iteration i is the left side of the integral for iteration i + 1, so
    # we can avoid computing the term twice.
    t0 = start * SourceSize.z
    u0 = tm.clamp((t - t0) / L, -1.0, 1.0)
    left = np.pi * u0 + tm.sin(np.pi * u0)
    for pix_x in range(start, end):
        # Integral of s / L * (0.5 + 0.5 * cos(PI * (t - t_x) / L)) dt_x over t0 to t1
        s = texelFetch(Source, tm.ivec2(pix_x, pix_y))
        t1 = (pix_x + 1) * SourceSize.z
        u1 = tm.clamp((t - t1) / L, -1.0, 1.0)
        right = np.pi * u1 + tm.sin(np.pi * u1)
        filtered += s * (left - right)
        left = right
    return filtered / (np.pi * 2.0)


def spot_fragment(image_in, output_dim):
    (in_height, in_width, in_planes) = image_in.shape
    (out_height, out_width, out_planes) = output_dim
    field_in = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_in.from_numpy(image_in)
    field_out = ti.Vector.field(n=3, dtype=float, shape=(out_height, out_width))
    output_field = taichi_spot_fragment(field_in, field_out)
    return field_out.to_numpy()


@ti.kernel
def taichi_spot_fragment(field_in: ti.template(), field_out: ti.template()):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = spot_sim(vTexCoord, field_in, SourceSize, OutputSize)
    return


@ti.func
def spot_sim(vTexCoord: tm.vec2, img, SourceSize: tm.vec4, OutputSize: tm.vec4) -> tm.vec3:
    # Overscan
    vTexCoord = (1.0 - tm.vec2(OVERSCAN_HORIZONTAL, OVERSCAN_VERTICAL)) * (vTexCoord - 0.5) + 0.5

    # Distance units (including for delta) are *scanlines heights*. This means
    # we need to adjust x distances by the aspect ratio. Overscan needs to be
    # taken into account because it can change the aspect ratio.
    upper_sample_y = int(tm.round(vTexCoord.y * SourceSize.y))
    lower_sample_y = upper_sample_y - 1
    delta = OutputSize.x * OutputSize.w * SourceSize.y * SourceSize.z * (1 - OVERSCAN_VERTICAL) / (1 - OVERSCAN_HORIZONTAL)
    upper_distance_y = (upper_sample_y + 0.5) - vTexCoord.y * SourceSize.y
    lower_distance_y = (lower_sample_y + 0.5) - vTexCoord.y * SourceSize.y

    output = tm.vec3(0.0)
    for sample_x in range(int(tm.round(vTexCoord.x * SourceSize.x - (MAX_SPOT_SIZE / delta))),
                          int(tm.round(vTexCoord.x * SourceSize.x + (MAX_SPOT_SIZE / delta)))):
        upper_sample = texelFetch(img, tm.ivec2(sample_x, upper_sample_y))
        lower_sample = texelFetch(img, tm.ivec2(sample_x, lower_sample_y))
        distance_x = delta * ((sample_x + 0.5) - vTexCoord.x * SourceSize.x)
        output += spot3(upper_sample, distance_x, upper_distance_y)
        output += spot3(lower_sample, distance_x, lower_distance_y)
    return delta * output


@ti.func
def spot1(sample, distance_x, distance_y):
    width_rcp = 1.0 / tm.mix(MAX_SPOT_SIZE * MIN_SPOT_SIZE, MAX_SPOT_SIZE, tm.sqrt(sample))
    x = tm.clamp(abs(distance_x) * width_rcp, 0.0, 1.0)
    y = tm.clamp(abs(distance_y) * width_rcp, 0.0, 1.0)
    return sample * MAX_SPOT_SIZE * width_rcp**2 * (0.5 * tm.cos(np.pi * x) + 0.5) * (0.5 * tm.cos(np.pi * y) + 0.5)


@ti.func
def spot2(sample, distance_x, distance_y):
    width_rcp = 1.0 / tm.mix(MAX_SPOT_SIZE * MIN_SPOT_SIZE, MAX_SPOT_SIZE, tm.sqrt(sample))
    x = tm.min(abs(distance_x) * width_rcp - 0.5, 0.5)
    y = tm.min(abs(distance_y) * width_rcp - 0.5, 0.5)
    return sample * MAX_SPOT_SIZE * width_rcp**2 * (2.0 * (x * abs(x) - x) + 0.5) * (2.0 * (y * abs(y) - y) + 0.5)


@ti.func
def spot3(sample, distance_x, distance_y):
    width_rcp = 1.0 / tm.mix(MAX_SPOT_SIZE * MIN_SPOT_SIZE, MAX_SPOT_SIZE, tm.sqrt(sample))
    x = tm.clamp(abs(distance_x) * width_rcp, 0.0, 1.0)
    y = tm.clamp(abs(distance_y) * width_rcp, 0.0, 1.0)
    return sample * MAX_SPOT_SIZE * width_rcp**2 * ((x * x) * (2.0 * x - 3.0) + 1.0) * ((y * y) * (2.0 * y - 3.0) + 1.0)


@ti.func
def spot_sim_gauss(vTexCoord: tm.vec2, img, SourceSize: tm.vec4, OutputSize: tm.vec4) -> tm.vec3:
    # Overscan
    vTexCoord = (1.0 - tm.vec2(OVERSCAN_HORIZONTAL, OVERSCAN_VERTICAL)) * (vTexCoord - 0.5) + 0.5

    # Distance units (including for delta) are *scanlines heights*. This means
    # we need to adjust x distances by the aspect ratio. Overscan needs to be
    # taken into account because it can change the aspect ratio.
    delta = OutputSize.x * OutputSize.w * SourceSize.y * SourceSize.z * (1 - OVERSCAN_VERTICAL) / (1 - OVERSCAN_HORIZONTAL)
    upper_sample_y = int(tm.round(vTexCoord.y * SourceSize.y))
    lower_sample_y = upper_sample_y - 1
    # The >= is important here. We need to make sure that we work in the same
    # direction that the upper_sample_y is rounded from.
    third_sample_y = upper_sample_y - 2 if tm.fract(vTexCoord.y * SourceSize.y) >= 0.5 else upper_sample_y + 1
    upper_distance_y = (upper_sample_y + 0.5) - vTexCoord.y * SourceSize.y
    lower_distance_y = (lower_sample_y + 0.5) - vTexCoord.y * SourceSize.y
    third_distance_y = (third_sample_y + 0.5) - vTexCoord.y * SourceSize.y

    output = tm.vec3(0.0)
    max_sigma = MAX_SPOT_SIZE / (2 * tm.sqrt(2 * tm.log(2)))
    for sample_x in range(int(tm.round(vTexCoord.x * SourceSize.x - ((3 * max_sigma) / delta))),
                          int(tm.round(vTexCoord.x * SourceSize.x + ((3 * max_sigma) / delta)))):
        upper_sample = texelFetch(img, tm.ivec2(sample_x, upper_sample_y))
        lower_sample = texelFetch(img, tm.ivec2(sample_x, lower_sample_y))
        third_sample = texelFetch(img, tm.ivec2(sample_x, third_sample_y))
        distance_x = delta * ((sample_x + 0.5) - vTexCoord.x * SourceSize.x)
        output += spot_gauss(upper_sample, distance_x, upper_distance_y)
        output += spot_gauss(lower_sample, distance_x, lower_distance_y)
        output += spot_gauss(third_sample, distance_x, third_distance_y)
    output = delta * output
    # Rescale output so that the maximum value is one.
    peak = (tm.sqrt(np.pi / 2) * abs(max_sigma)) / (max_sigma * max_sigma) * (1.0 + 2.0 * tm.exp(-1.0 / (2 * max_sigma * max_sigma)))
    return output / peak


@ti.func
def spot_sim_gauss4(vTexCoord: tm.vec2, img, SourceSize: tm.vec4, OutputSize: tm.vec4) -> tm.vec3:
    # Overscan
    vTexCoord = (1.0 - tm.vec2(OVERSCAN_HORIZONTAL, OVERSCAN_VERTICAL)) * (vTexCoord - 0.5) + 0.5

    # Distance units (including for delta) are *scanlines heights*. This means
    # we need to adjust x distances by the aspect ratio. Overscan needs to be
    # taken into account because it can change the aspect ratio.
    delta = OutputSize.x * OutputSize.w * SourceSize.y * SourceSize.z * (1 - OVERSCAN_VERTICAL) / (1 - OVERSCAN_HORIZONTAL)

    upper_sample_y = int(tm.round(vTexCoord.y * SourceSize.y))
    upper_sample_y2 = upper_sample_y + 1
    lower_sample_y = upper_sample_y - 1
    lower_sample_y2 = upper_sample_y - 2

    upper_distance_y = (upper_sample_y + 0.5) - vTexCoord.y * SourceSize.y
    upper_distance_y2 = (upper_sample_y2 + 0.5) - vTexCoord.y * SourceSize.y
    lower_distance_y = (lower_sample_y + 0.5) - vTexCoord.y * SourceSize.y
    lower_distance_y2 = (lower_sample_y2 + 0.5) - vTexCoord.y * SourceSize.y

    output = tm.vec3(0.0)
    max_sigma = MAX_SPOT_SIZE / (2 * tm.sqrt(2 * tm.log(2)))
    for sample_x in range(int(tm.round(vTexCoord.x * SourceSize.x - ((3 * max_sigma) / delta))),
                          int(tm.round(vTexCoord.x * SourceSize.x + ((3 * max_sigma) / delta)))):
        upper_sample = texelFetch(img, tm.ivec2(sample_x, upper_sample_y))
        upper_sample2 = texelFetch(img, tm.ivec2(sample_x, upper_sample_y2))
        lower_sample = texelFetch(img, tm.ivec2(sample_x, lower_sample_y))
        lower_sample2 = texelFetch(img, tm.ivec2(sample_x, lower_sample_y2))
        distance_x = delta * ((sample_x + 0.5) - vTexCoord.x * SourceSize.x)
        output += spot_gauss(upper_sample, distance_x, upper_distance_y)
        output += spot_gauss(upper_sample2, distance_x, upper_distance_y2)
        output += spot_gauss(lower_sample, distance_x, lower_distance_y)
        output += spot_gauss(lower_sample2, distance_x, lower_distance_y2)
    output = delta * output
    # Rescale output so that the maximum value is one.
    peak = (tm.sqrt(np.pi / 2) * abs(max_sigma) / (max_sigma * max_sigma) * tm.exp(-2 / (max_sigma * max_sigma)) * \
            (2 * tm.exp(3 / (2 * max_sigma * max_sigma)) + tm.exp(2 / (max_sigma * max_sigma)) + 1))
    return output / peak


@ti.func
def spot_sim_gauss5(vTexCoord: tm.vec2, img, SourceSize: tm.vec4, OutputSize: tm.vec4) -> tm.vec3:
    # TODO
    # Overscan
    vTexCoord = (1.0 - tm.vec2(OVERSCAN_HORIZONTAL, OVERSCAN_VERTICAL)) * (vTexCoord - 0.5) + 0.5

    # Distance units (including for delta) are *scanlines heights*. This means
    # we need to adjust x distances by the aspect ratio. Overscan needs to be
    # taken into account because it can change the aspect ratio.
    delta = OutputSize.x * OutputSize.w * SourceSize.y * SourceSize.z * (1 - OVERSCAN_VERTICAL) / (1 - OVERSCAN_HORIZONTAL)

    upper_sample_y = int(tm.round(vTexCoord.y * SourceSize.y))
    upper_sample_y2 = upper_sample_y + 1
    lower_sample_y = upper_sample_y - 1
    lower_sample_y2 = upper_sample_y - 2

    upper_distance_y = (upper_sample_y + 0.5) - vTexCoord.y * SourceSize.y
    upper_distance_y2 = (upper_sample_y2 + 0.5) - vTexCoord.y * SourceSize.y
    lower_distance_y = (lower_sample_y + 0.5) - vTexCoord.y * SourceSize.y
    lower_distance_y2 = (lower_sample_y2 + 0.5) - vTexCoord.y * SourceSize.y

    output = tm.vec3(0.0)
    max_sigma = MAX_SPOT_SIZE / (2 * tm.sqrt(2 * tm.log(2)))
    for sample_x in range(int(tm.round(vTexCoord.x * SourceSize.x - ((3 * max_sigma) / delta))),
                          int(tm.round(vTexCoord.x * SourceSize.x + ((3 * max_sigma) / delta)))):
        upper_sample = texelFetch(img, tm.ivec2(sample_x, upper_sample_y))
        upper_sample2 = texelFetch(img, tm.ivec2(sample_x, upper_sample_y2))
        lower_sample = texelFetch(img, tm.ivec2(sample_x, lower_sample_y))
        lower_sample2 = texelFetch(img, tm.ivec2(sample_x, lower_sample_y2))
        distance_x = delta * ((sample_x + 0.5) - vTexCoord.x * SourceSize.x)
        output += spot_gauss(upper_sample, distance_x, upper_distance_y)
        output += spot_gauss(upper_sample2, distance_x, upper_distance_y2)
        output += spot_gauss(lower_sample, distance_x, lower_distance_y)
        output += spot_gauss(lower_sample2, distance_x, lower_distance_y2)
    output = delta * output
    # Rescale output so that the maximum value is one.
    peak = (tm.sqrt(np.pi / 2) * abs(max_sigma)) / (max_sigma * max_sigma) * (1.0 + 2.0 * tm.exp(-1.0 / (2 * max_sigma * max_sigma)))
    return output / peak


@ti.func
def spot_gauss(sample, distance_x, distance_y):
    width = tm.mix(MAX_SPOT_SIZE * MIN_SPOT_SIZE, MAX_SPOT_SIZE, tm.sqrt(sample))
    sigma = width / (2 * tm.sqrt(2 * tm.log(2)))
    x = distance_x
    y = distance_y
    # We don't need pi, as in the normal distribution, because we are
    # rescaling later (we can ignore constant factors).
    return sample / (2 * sigma * sigma) * tm.exp(-(x * x + y * y) / (2 * sigma * sigma))


def spot_analytical_fragment(image_in, output_dim):
    (in_height, in_width, in_planes) = image_in.shape
    (out_height, out_width, out_planes) = output_dim
    field_in = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_in.from_numpy(image_in)
    field_out = ti.Vector.field(n=3, dtype=float, shape=(out_height, out_width))
    output_field = taichi_spot_analytical_fragment(field_in, field_out)
    return field_out.to_numpy()


@ti.kernel
def taichi_spot_analytical_fragment(field_in: ti.template(), field_out: ti.template()):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = spot_sim_analytical(vTexCoord, field_in, SourceSize, OutputSize)
    return

@ti.func
def spot_sim_analytical(vTexCoord: tm.vec2, img, SourceSize: tm.vec4, OutputSize: tm.vec4) -> tm.vec3:
    # Overscan
    vTexCoord = (1.0 - tm.vec2(OVERSCAN_HORIZONTAL, OVERSCAN_VERTICAL)) * (vTexCoord - 0.5) + 0.5

    # Distance units (including for delta) are *scanlines heights*. This means
    # we need to adjust x distances by the aspect ratio. Overscan needs to be
    # taken into account because it can change the aspect ratio.
    upper_sample_y = int(tm.round(vTexCoord.y * SourceSize.y))
    lower_sample_y = upper_sample_y - 1
    delta = OutputSize.x * OutputSize.w * SourceSize.y * SourceSize.z * (1 - OVERSCAN_VERTICAL) / (1 - OVERSCAN_HORIZONTAL)
    upper_distance_y = (upper_sample_y + 0.5) - vTexCoord.y * SourceSize.y
    lower_distance_y = (lower_sample_y + 0.5) - vTexCoord.y * SourceSize.y

    output = tm.vec3(0.0)
    x = vTexCoord.x * SourceSize.x * delta
    for sample_x in range(int(tm.floor(vTexCoord.x * SourceSize.x - (MAX_SPOT_SIZE / delta))),
                          int(tm.floor(vTexCoord.x * SourceSize.x + (MAX_SPOT_SIZE / delta))) + 1):
        upper_sample = texelFetch(img, tm.ivec2(sample_x, upper_sample_y))
        lower_sample = texelFetch(img, tm.ivec2(sample_x, lower_sample_y))
        x0 = delta * sample_x
        x1 = x0 + delta
        output += spot_analytical(upper_sample, x, x0, x1, upper_distance_y)
        output += spot_analytical(lower_sample, x, x0, x1, lower_distance_y)
    return output


@ti.func
def spot_analytical(sample: tm.vec3, x: float, x0_: float, x1_: float, y_: float):
    w = tm.mix(MAX_SPOT_SIZE * MIN_SPOT_SIZE, MAX_SPOT_SIZE, tm.sqrt(sample))
    x0 = tm.clamp((x - x0_) / w, -1.0, 1.0)
    x1 = tm.clamp((x - x1_) / w, -1.0, 1.0)
    y = tm.clamp(abs(y_) / w, 0.0, 1.0)
    return MAX_SPOT_SIZE / (2.0 * np.pi) * sample * 1.0 / w * (0.5 * tm.cos(np.pi * y) + 0.5) * \
        (np.pi * x0 + tm.sin(np.pi * x0) - np.pi * x1 - tm.sin(np.pi * x1))


@ti.func
def spot_analytical2(sample: tm.vec3, x: float, x0_: float, x1_: float, y_: float):
    w = tm.mix(MAX_SPOT_SIZE * MIN_SPOT_SIZE, MAX_SPOT_SIZE, tm.sqrt(sample))
    x0 = tm.clamp((x - x0_) / w, -1.0, 1.0)
    x1 = tm.clamp((x - x1_) / w, -1.0, 1.0)
    y = tm.clamp(abs(y_) / w, 0.0, 1.0)
    return MAX_SPOT_SIZE * sample * 1.0 / w * ((y * y) * (2.0 * y - 3.0) + 1.0) * (
        ((x0 * x0) * (tm.sign(x0) * 0.5 * (x0 * x0) - x0) + x0) -
        ((x1 * x1) * (tm.sign(x1) * 0.5 * (x1 * x1) - x1) + x1)
    )


def spot_fast_fragment(image_in, output_dim):
    (in_height, in_width, in_planes) = image_in.shape
    (out_height, out_width, out_planes) = output_dim
    field_in = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_in.from_numpy(image_in)
    field_intermediate = ti.Vector.field(n=3, dtype=float, shape=(in_height, out_width))
    field_out = ti.Vector.field(n=3, dtype=float, shape=(out_height, out_width))

    taichi_spot_fast_horizontal_fragment(field_in, field_intermediate)
    imwrite("intermediate.png", linear_to_srgb(field_intermediate.to_numpy()))
    taichi_spot_fast_vertical_fragment(field_intermediate, field_out)
    return field_out.to_numpy()


@ti.kernel
def taichi_spot_fast_horizontal_fragment(field_in: ti.template(), field_out: ti.template()):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)

    # Distance units (including for delta) are *scanlines heights*. This means
    # we need to adjust x distances by the aspect ratio. Overscan needs to be
    # taken into account because it can change the aspect ratio.
    # delta = OutputSize.x * OutputSize.w * SourceSize.y * SourceSize.z * (1 - OVERSCAN_VERTICAL) / (1 - OVERSCAN_HORIZONTAL)
    delta = 2880 / 2160 * SourceSize.y * SourceSize.z * (1 - OVERSCAN_VERTICAL) / (1 - OVERSCAN_HORIZONTAL)

    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = spot_sim_fast_horizontal(vTexCoord, field_in, SourceSize, OutputSize, delta)
    return


@ti.kernel
def taichi_spot_fast_vertical_fragment(field_in: ti.template(), field_out: ti.template()):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)

    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = spot_sim_fast_vertical(vTexCoord, field_in, SourceSize, OutputSize)
    return


@ti.func
def spot_sim_fast_horizontal(vTexCoord: tm.vec2, img, SourceSize: tm.vec4, OutputSize: tm.vec4, delta: float) -> tm.vec3:
    sample_y = int(vTexCoord.y * SourceSize.y)
    output = tm.vec3(0.0)
    x = vTexCoord.x * SourceSize.x * delta
    for sample_x in range(int(tm.floor(vTexCoord.x * SourceSize.x - (MAX_SPOT_SIZE / delta))),
                          int(tm.floor(vTexCoord.x * SourceSize.x + (MAX_SPOT_SIZE / delta))) + 1):
        sample = texelFetch(img, tm.ivec2(sample_x, sample_y))
        x0 = delta * sample_x
        x1 = x0 + delta
        output += spot_fast_horizontal(sample, x, x0, x1)
    return output


@ti.func
def spot_sim_fast_vertical(vTexCoord: tm.vec2, img, SourceSize: tm.vec4, OutputSize: tm.vec4) -> tm.vec3:
    # Overscan
    vTexCoord = (1.0 - tm.vec2(OVERSCAN_HORIZONTAL, OVERSCAN_VERTICAL)) * (vTexCoord - 0.5) + 0.5

    # Distance units (including for delta) are *scanlines heights*. This means
    # we need to adjust x distances by the aspect ratio. Overscan needs to be
    # taken into account because it can change the aspect ratio.
    upper_sample_y = int(tm.round(vTexCoord.y * SourceSize.y))
    lower_sample_y = upper_sample_y - 1
    upper_distance_y = (upper_sample_y + 0.5) - vTexCoord.y * SourceSize.y
    lower_distance_y = (lower_sample_y + 0.5) - vTexCoord.y * SourceSize.y

    upper_sample = texelFetch(img, tm.ivec2(int(vTexCoord.x * SourceSize.x), upper_sample_y))
    lower_sample = texelFetch(img, tm.ivec2(int(vTexCoord.x * SourceSize.x), lower_sample_y))
    return spot_fast_vertical(upper_sample, upper_distance_y) + spot_fast_vertical(lower_sample, lower_distance_y)


@ti.func
def spot_fast_horizontal(sample: tm.vec3, x: float, x0_: float, x1_: float):
    w = tm.mix(MAX_SPOT_SIZE * MIN_SPOT_SIZE, MAX_SPOT_SIZE, tm.sqrt(sample))
    x0 = tm.clamp((x - x0_) / w, -1.0, 1.0)
    x1 = tm.clamp((x - x1_) / w, -1.0, 1.0)
    return 1 / (2.0 * np.pi) * sample * (np.pi * x0 + tm.sin(np.pi * x0) - np.pi * x1 - tm.sin(np.pi * x1))


@ti.func
def spot_fast_vertical(sample: tm.vec3, y_: float):
    w = tm.mix(MAX_SPOT_SIZE * MIN_SPOT_SIZE, MAX_SPOT_SIZE, tm.sqrt(sample))
    y = tm.clamp(abs(y_) / w, 0.0, 1.0)
    return sample * MAX_SPOT_SIZE / w * (0.5 * tm.cos(np.pi * y) + 0.5)


f16vec3 = ti.types.vector(3, ti.f16)


@ti.func
def spot_sim_f16(vTexCoord: tm.vec2, img, SourceSize: tm.vec4, OutputSize: tm.vec4) -> tm.vec3:
    # Overscan
    vTexCoord = (1.0 - tm.vec2(OVERSCAN_HORIZONTAL, OVERSCAN_VERTICAL)) * (vTexCoord - 0.5) + 0.5

    # Distance units (including for delta) are *scanlines heights*. This means
    # we need to adjust x distances by the aspect ratio. Overscan needs to be
    # taken into account because it can change the aspect ratio.
    # Check if we should be deinterlacing.
    upper_sample_y = int(tm.round(vTexCoord.y * SourceSize.y))
    lower_sample_y = upper_sample_y - 1
    delta = OutputSize.x * OutputSize.w * SourceSize.y * SourceSize.z * (1 - OVERSCAN_VERTICAL) / (1 - OVERSCAN_HORIZONTAL)
    upper_distance_y = ti.f16((upper_sample_y + 0.5) - vTexCoord.y * SourceSize.y)
    lower_distance_y = ti.f16((lower_sample_y + 0.5) - vTexCoord.y * SourceSize.y)

    output = tm.vec3(0.0)
    for sample_x in range(int(tm.round(vTexCoord.x * SourceSize.x - (MAX_SPOT_SIZE / delta))),
                          int(tm.round(vTexCoord.x * SourceSize.x + (MAX_SPOT_SIZE / delta)))):
        upper_sample = f16vec3(texelFetch(img, tm.ivec2(sample_x, upper_sample_y)))
        lower_sample = f16vec3(texelFetch(img, tm.ivec2(sample_x, lower_sample_y)))
        distance_x = ti.f16(delta * ((sample_x + 0.5) - vTexCoord.x * SourceSize.x))
        output += spot3_float16(upper_sample, distance_x, upper_distance_y)
        output += spot3_float16(lower_sample, distance_x, lower_distance_y)
    return delta * output


@ti.func
def spot3_float16(sample: f16vec3, distance_x: f16vec3, distance_y: f16vec3) -> f16vec3:
    width_rcp = ti.f16(1.0) / tm.mix(MAX_SPOT_SIZE * MIN_SPOT_SIZE, MAX_SPOT_SIZE, tm.sqrt(sample))
    x = tm.clamp(abs(distance_x) * width_rcp, ti.f16(0.0), ti.f16(1.0))
    y = tm.clamp(abs(distance_y) * width_rcp, ti.f16(0.0), ti.f16(1.0))
    return sample * MAX_SPOT_SIZE * width_rcp**2 * \
            ((x * x) * (ti.f16(2.0) * x - ti.f16(3.0)) + ti.f16(1.0)) * \
            ((y * y) * (ti.f16(2.0) * y - ti.f16(3.0)) + ti.f16(1.0))


def box_blur(image_in, radius):
    """Do several box blurs on the image, approximating a gaussian blur.

    This is a very fast blur for large images. The speed is not
    dependent on the radius."""
    (in_height, in_width, in_planes) = image_in.shape
    field_in = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_in.from_numpy(image_in)
    field_out = ti.Vector.field(n=3, dtype=float, shape=(in_width, in_height))
    for i in range(4):
        taichi_box_blur(field_in, field_out, radius)
        taichi_box_blur(field_out, field_in, radius)
    return field_in.to_numpy()


@ti.kernel
def taichi_box_blur(field_in: ti.template(), field_out: ti.template(), radius: int):
    """Do a 1D horizontal box blur on field_in, writing the transposed result to field_out"""
    (in_height, in_width) = field_in.shape
    width = 2 * radius + 1
    for y in range(in_height):
        running_sum = tm.vec3(0.0)
        # TODO If radius or width is > in_width?
        for x in range(radius):
            running_sum += field_in[y, x]
        for x in range(radius, width):
            running_sum += field_in[y, x]
            field_out[x - radius, y] = running_sum / width
        for x in range(width, in_width):
            running_sum += field_in[y, x]
            running_sum -= field_in[y, x - width]
            field_out[x - radius, y] = running_sum / width
        for x in range(in_width, in_width + radius):
            running_sum -= field_in[y, x - width]
            field_out[x - radius, y] = running_sum / width
    return


def gaussian_blur(image_in, sigma):
    (in_height, in_width, in_planes) = image_in.shape
    field_in = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_in.from_numpy(image_in)
    field_out = ti.Vector.field(n=3, dtype=float, shape=(in_width, in_height))
    gaussian_fragment(field_in, field_out, sigma)
    gaussian_fragment(field_out, field_in, sigma)
    return field_in.to_numpy()


@ti.kernel
def gaussian_fragment(field_in: ti.template(), field_out: ti.template(), sigma: float):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = gaussian_taichi(vTexCoord, field_in, SourceSize, OutputSize, sigma)
    return


@ti.func
def gaussian_taichi(vTexCoord: tm.vec2, Source, SourceSize: tm.vec4, OutputSize: tm.vec4, sigma: float) -> tm.vec3:
    pos = vTexCoord.yx * SourceSize.xy
    weight_sum = 0.0
    value = tm.vec3(0.0)
    center = tm.ivec2(int(tm.round(pos.x)), int(tm.floor(pos.y)))
    for x in range(center.x - int(tm.ceil(4 * sigma)), center.x + int(tm.ceil(4 * sigma)) + 1):
        distance_x = pos.x - x - 0.5
        weight = tm.exp(-(distance_x * distance_x) / (2 * sigma * sigma))
        weight_sum += weight
        value += weight * texelFetch(Source, tm.ivec2(x, center.y))
    return value / weight_sum


def subpixel_mask(img_in):
    if MASK_PATTERN == 0:
        mask_tile = np.array([[1, 0, 1], [0, 1, 0]])  # <=1080p
        mask_coverage = 2
    elif MASK_PATTERN == 1:
        mask_tile = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])  # 1080p/1440p
        mask_coverage = 3
    elif MASK_PATTERN == 2:
        mask_tile = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1]])  # 1440p/4k
        mask_coverage = 2
    elif MASK_PATTERN == 3:
        mask_tile = np.array([[1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0]])  # 4k, lower TVL
        mask_coverage = 15 / 6
    else:
        raise AssertionError('Unhandled mask pattern')
    # TODO ? For BGR subpixel arrangement, just transpose red and blue.
    mask = np.broadcast_to(mask_tile[np.arange(OUTPUT_RESOLUTION[1]) % mask_tile.shape[0]], (OUTPUT_RESOLUTION[0], OUTPUT_RESOLUTION[1], 3))

    # Piecewise phase-in. The original image starts phasing in only when we've
    # maxed the brightness we can get from our masked image.
    # s = mask_coverage / (mask_coverage - 1)
    # weight = np.clip(-s * img_in + s, 0.0, 1.0)
    # return (1 - weight) * img_in + mask_coverage * mask * weight * img_in

    # Cubic phase-in. Keeps the mask strength higher for longer than linear
    # but has no discontinuity like piecewise.
    a = np.clip((img_in - 1) / (1 - mask_coverage), 0, img_in)
    b = np.clip((1 - mask_coverage * img_in) / (1 - mask_coverage), 0, img_in)
    return mask_coverage * a * mask + b


def coverage_mask(image_in):
    (in_height, in_width, in_planes) = image_in.shape
    field_in = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_in.from_numpy(image_in)
    field_out = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    coverage_mask_fragment(field_in, field_out)
    return field_out.to_numpy()


@ti.kernel
def coverage_mask_fragment(field_in: ti.template(), field_out: ti.template()):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = coverage_mask_taichi(vTexCoord, field_in, SourceSize, OutputSize)
    return


@ti.func
def coverage_mask_taichi(vTexCoord: tm.vec2, Source, SourceSize: tm.vec4, OutputSize: tm.vec4):
    mask_coverage = 3.0

    # Working in [0, MASK_TRIADS + 1) coordinate space to find the phosphor edges.
    x = vTexCoord.x * MASK_TRIADS
    phosphor_left_edge = tm.floor(x + tm.vec3(1.0 / 3.0, 0.0, -1.0 / 3.0)) + tm.vec3(0.0, 1.0 / 3.0, 2.0 / 3.0)
    # Working in [0, OutputSize.x + 1) coordinate space for the coverage.
    x = vTexCoord.x * OutputSize.x
    phosphor_left_edge = phosphor_left_edge / MASK_TRIADS * OutputSize.x
    phosphor_right_edge = phosphor_left_edge + OutputSize.x / MASK_TRIADS / 3.0 # ?? correct ??
    mask = tm.clamp(min(x + 0.5, phosphor_right_edge) - max(x - 0.5, phosphor_left_edge), 0.0, 1.0)

    pixel_coords = tm.ivec2(int(vTexCoord.x * SourceSize.x), int(vTexCoord.y * SourceSize.y))
    pixel_value = texelFetch(Source, pixel_coords)

    # Cubic phase-in. Keeps the mask strength higher for longer than linear
    # but has no discontinuity like piecewise.
    s = mask_coverage / (mask_coverage - 1.0);
    a = -s + 2.0;
    b = s - 3.0;
    weight = a * (pixel_value * pixel_value * pixel_value) + b * (pixel_value * pixel_value) + 1.0;
    return pixel_value - pixel_value * weight * (1.0 - mask_coverage * mask);


def bandlimit_mask(image_in):
    (in_height, in_width, in_planes) = image_in.shape
    field_in = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_in.from_numpy(image_in)
    field_out = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    bandlimit_mask_fragment(field_in, field_out)
    return field_out.to_numpy()


@ti.kernel
def bandlimit_mask_fragment(field_in: ti.template(), field_out: ti.template()):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = bandlimit_mask_taichi2(vTexCoord, field_in, SourceSize, OutputSize)
    return


phosphors = ti.Vector.field(n=3, dtype=float, shape=3)
phosphors[0] = tm.vec3(1.0, 0.0, 0.0)
phosphors[1] = tm.vec3(0.0, 1.0, 0.0)
phosphors[2] = tm.vec3(0.0, 0.0, 1.0)


@ti.func
def bandlimit_mask_taichi(vTexCoord: tm.vec2, Source, SourceSize: tm.vec4, OutputSize: tm.vec4):
    w = (MASK_TRIADS * 3.0) / OutputSize.x
    x = MASK_TRIADS * 3.0 * vTexCoord.x
    mask = tm.vec3(0.0)
    for offset in range(-2, 3):
        x0 = tm.floor(x) + offset
        x1 = tm.ceil(x) + offset
        x0 = tm.clamp((x - x0) / w, -1.0, 1.0)
        x1 = tm.clamp((x - x1) / w, -1.0, 1.0)
        weight = np.pi * x0 + tm.sin(np.pi * x0) - np.pi * x1 - tm.sin(np.pi * x1)
        # weight = tm.sin(np.pi / 2.0 * x0) - tm.sin(np.pi / 2.0 * x1)  # For filter kernel pi / 4 * cos(pi / 2 * x)
        if vTexCoord.x < 0.01 and vTexCoord.y * OutputSize.y < 1:
            print(w, weight)
        # print((int(MASK_TRIADS * 3.0 * vTexCoord.x) + offset) % 3)
        mask += weight * phosphors[(int(x) + offset) % 3]
    mask /= 2 * np.pi
    # mask /= 2  # For filter kernel pi / 4 * cos(pi / 2 * x)
    if vTexCoord.x < 0.01 and vTexCoord.y * OutputSize.y < 1:
        print(mask)

    pixel_coords = tm.ivec2(int(vTexCoord.x * SourceSize.x), int(vTexCoord.y * SourceSize.y))
    pixel_value = texelFetch(Source, pixel_coords)

    mask_coverage = 3.0

    # Cubic phase-in. Keeps the mask strength higher for longer than linear
    # but has no discontinuity like piecewise.
    s = mask_coverage / (mask_coverage - 1.0)
    a = -s + 2.0
    b = s - 3.0
    weight = a * (pixel_value * pixel_value * pixel_value) + b * (pixel_value * pixel_value) + 1.
    return pixel_value - pixel_value * weight * (1.0 - mask_coverage * mask)


@ti.func
def bandlimit_mask_taichi2(vTexCoord: tm.vec2, Source, SourceSize: tm.vec4, OutputSize: tm.vec4):
    # With unrolled loops and common terms collapsed for performance
    w = (MASK_TRIADS * 3.0) / OutputSize.x
    x = MASK_TRIADS * 3.0 * vTexCoord.x
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

    if mask.x < 0 or mask.x > 1 or mask.y < 0 or mask.y > 1 or mask.z < 0 or mask.z > 1:
        print(mask)

    pixel_coords = tm.ivec2(int(vTexCoord.x * SourceSize.x), int(vTexCoord.y * SourceSize.y))
    pixel_value = texelFetch(Source, pixel_coords)

    mask_coverage = 3.0

    a = tm.clamp((pixel_value - 1) / (1 - mask_coverage), 0, pixel_value)
    b = tm.clamp((1 - mask_coverage * pixel_value) / (1 - mask_coverage), 0, pixel_value)
    return mask_coverage * a * mask + b

    # # Cubic phase-in. Keeps the mask strength higher for longer than linear
    # # but has no discontinuity like piecewise.
    # return pixel_value * pixel_value + mask_coverage * mask * (1.0 - pixel_value) * pixel_value  # XXX
    # s = mask_coverage / (mask_coverage - 1.0)
    # a = -s + 2.0
    # b = s - 3.0
    # weight = a * (pixel_value * pixel_value * pixel_value) + b * (pixel_value * pixel_value) + 1.0
    # return pixel_value - pixel_value * weight * (1.0 - mask_coverage * mask)


def additive_mask(image_in):
    (in_height, in_width, in_planes) = image_in.shape
    field_in = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_in.from_numpy(image_in)
    field_out = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    additive_mask_fragment(field_in, field_out)
    return field_out.to_numpy()


@ti.kernel
def additive_mask_fragment(field_in: ti.template(), field_out: ti.template()):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = additive_mask_taichi(vTexCoord, field_in, SourceSize, OutputSize)
    return


@ti.func
def additive_mask_taichi(vTexCoord: tm.vec2, Source, SourceSize: tm.vec4, OutputSize: tm.vec4):
    offset = tm.vec3(0.0, -1 / (3 * MASK_TRIADS), -2 / (3 * MASK_TRIADS))
    half = OutputSize.z * 0.5
    x = vTexCoord.x
    mask = tm.vec3(1.0, 1.0, 1.0)
    mask_coverage = 1.0
    if MASK_TRIADS * 4 < OutputSize.x:  # XXX
        # mask = 1 / 3 + \
        #     2 / np.pi * tm.sin(np.pi / 3) * tm.cos(2 * np.pi * MASK_TRIADS * (x + offset)) + \
        #     1 / np.pi * tm.sin((2 / 3) * np.pi) * tm.cos(4 * np.pi * MASK_TRIADS * (x + offset))

        # mask = 1 / 3 + \
        #     0.5513288954217920 * tm.sin(2 * np.pi * MASK_TRIADS * (x + offset)) + \
        #     0.2756644477108960 * tm.sin(4 * np.pi * MASK_TRIADS * (x + offset))
        # mask = (mask + 0.080163) / 1.24049

        # mask = 1.0 / 3.0 * OutputSize.z + 0.5513288954217920 * \
        #         (tm.sin(2 * np.pi * MASK_TRIADS * (x + offset + half)) - \
        #         tm.sin(2 * np.pi * MASK_TRIADS * (x + offset - half))) / (2 * np.pi * MASK_TRIADS) + \
        #         0.2756644477108960 * (tm.sin(4 * np.pi * MASK_TRIADS * (x + offset + half)) - \
        #         tm.sin(4 * np.pi * MASK_TRIADS * (x + offset - half))) / (4 * np.pi * MASK_TRIADS)
        # mask /= OutputSize.z

        # Area under offset
        mask = (1.0 / 3.0 + 0.080163) * OutputSize.z
        # Area under first harmonic
        right = tm.sin(2 * np.pi * MASK_TRIADS * (x + offset + half))
        left = tm.sin(2 * np.pi * MASK_TRIADS * (x + offset - half))
        mask += 0.5513288954217920 * (right - left) / (2 * np.pi * MASK_TRIADS)
        # Area under second harmonic
        right = tm.sin(4 * np.pi * MASK_TRIADS * (x + offset + half))
        left = tm.sin(4 * np.pi * MASK_TRIADS * (x + offset - half))
        mask += 0.2756644477108960 * (right - left) / (4 * np.pi * MASK_TRIADS)
        mask /= (1.24049 * OutputSize.z)

        mask_coverage = 3
        # mask_coverage = 1 / (1 / 3 + 0.080163)
    elif MASK_TRIADS * 2 < OutputSize.x:
        # mask = 0.5 + 0.5 * tm.cos(2 * np.pi * MASK_TRIADS * (x + offset))
        mask = 0.5 * OutputSize.z + 0.5 * \
                (tm.sin(2 * np.pi * MASK_TRIADS * (x + offset + half)) - \
                tm.sin(2 * np.pi * MASK_TRIADS * (x + offset - half))) / (2 * np.pi * MASK_TRIADS)
        # mask = 1 / 3 + \
        #     0.5513288954217920 * tm.cos(2 * np.pi * MASK_TRIADS * (x + offset))
        # mask = 1.0 / 3.0 * OutputSize.z + 0.5513288954217920 * \
        #         (tm.sin(2 * np.pi * MASK_TRIADS * (x + offset + half)) - \
        #         tm.sin(2 * np.pi * MASK_TRIADS * (x + offset - half))) / (2 * np.pi * MASK_TRIADS)
        mask /= OutputSize.z
        mask_coverage = 2

    if vTexCoord.x < 0.01 and vTexCoord.y * OutputSize.y < 1:
        print(x + offset + half, (x + offset - half))
        print(tm.sin(2 * np.pi * MASK_TRIADS * (x + offset + half)) - tm.sin(2 * np.pi * MASK_TRIADS * (x + offset - half)),
            tm.sin(4 * np.pi * MASK_TRIADS * (x + offset + half)) - tm.sin(4 * np.pi * MASK_TRIADS * (x + offset - half)))
        print(mask, mask.x + mask.y + mask.z)

    pixel_coords = tm.ivec2(int(vTexCoord.x * SourceSize.x), int(vTexCoord.y * SourceSize.y))
    pixel_value = texelFetch(Source, pixel_coords)

    a = tm.clamp((pixel_value - 1) / (1 - mask_coverage), 0, pixel_value)
    b = tm.clamp((1 - mask_coverage * pixel_value) / (1 - mask_coverage), 0, pixel_value)
    return mask_coverage * a * mask + b
    # # return mask * pixel_value
    # #
    # return pixel_value * pixel_value + mask_coverage * mask * (1.0 - pixel_value) * pixel_value
    # #
    # s = mask_coverage / (mask_coverage - 1)
    # weight = tm.clamp(-s * pixel_value + s, 0.0, 1.0)
    # return (1.0 - weight) * pixel_value + mask_coverage * mask * weight * pixel_value

    # # Cubic phase-in. Keeps the mask strength higher for longer than linear
    # # but has no discontinuity like piecewise.
    # s = mask_coverage / (mask_coverage - 1.0)
    # a = -s + 2.0
    # b = s - 3.0
    # weight = a * (pixel_value * pixel_value * pixel_value) + b * (pixel_value * pixel_value) + 1.0
    # return pixel_value - pixel_value * weight * (1.0 - mask_coverage * mask)


@ti.func
def smoothstep(edge0: tm.vec3, edge1: tm.vec3, x: float):
    t= tm.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def tiled_mask(img_in):
    if MASK_TYPE == 'aperture':
        mask_tile = imread('mask-aperture.png')[:, :, 0:3].astype(np.float32) / 255.0
    elif MASK_TYPE == 'slot':
        mask_tile = imread('mask-slot.png')[:, :, 0:3].astype(np.float32) / 255.0
    else:
        raise AssertionError('Unhandled mask type')
    mask_coverage = 1 / np.average(mask_tile)  # TODO axis?
    mask = generate_mask(mask_tile, (OUTPUT_RESOLUTION[0], OUTPUT_RESOLUTION[1], 3), MASK_TRIADS)
    imwrite('mask_resized.png', linear_to_srgb(mask))  # DEBUG

    # Cubic phase-in. Keeps the mask strength higher for longer than linear
    # but has no discontinuity like piecewise.
    s = mask_coverage / (mask_coverage - 1);
    a = -s + 2.0;
    b = s - 3.0;
    weight = a * np.power(img_in, 3.0) + b * np.power(img_in, 2.0) + 1.0;
    return img_in * ((1 - weight) + mask_coverage * mask * weight);


def generate_mask(mask, out_shape, triads):
    (mask_height, mask_width, mask_planes) = mask.shape
    (out_height, out_width, out_planes) = out_shape
    scale = mask_width / 2 * triads / out_width
    print("scale: {}".format(scale))
    field_in = ti.Vector.field(n=3, dtype=float, shape=(mask_height, mask_width))
    field_in.from_numpy(mask)
    field_out = ti.Vector.field(n=3, dtype=float, shape=(out_width, mask_height))
    lanczos3_downscale(field_in, field_out, scale)
    # imwrite('mask_resized_pass1.png', linear_to_srgb(field_out.to_numpy()))  # DEBUG
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
        # if y_pos == 5 and x_pos / scale < 1:
        #     print(f'x_pos: {x_pos}, x: {x}, distance_x: {distance_x}, weight: {weight}, texel: {texelFetchRepeat(Source, tm.ivec2(x, y_pos))}')
    return value / weight_sum


def tone_map(srgb):
    # sRGB -> ACES
    acesInput = np.array([
            [0.59719, 0.35458, 0.04823],
            [0.07600, 0.90834, 0.01566],
            [0.02840, 0.13383, 0.83777]])

    # ACES -> sRGB
    acesOutput = np.array([
            [ 1.60475, -0.53108, -0.07367],
            [-0.10208,  1.10813, -0.00605],
            [-0.00327, -0.07276,  1.07602]])

    v = np.tensordot(srgb, acesInput, axes=1)
    print(v.shape)
    a = v * (v + 0.0245786) - 0.000090537
    b = v * (0.983729 * v + 0.4329510) + 0.238081
    print((a/b).shape)
    return np.tensordot(a / b, acesOutput, axes=1)


# For low-pass
SAMPLES = 1024 #2880  #907  #1400
USE_YIQ = False
GAMMA = 2.4
# -6dB cutoff is at 1 / 2L in cycles. We want CUTOFF * 53.33e-6 cycles (CUTOFF bandwidth and NTSC standard active line time of 53.33us).
# CUTOFF = np.array([5.0e6, 0.6e6, 0.6e6])  # Hz
CUTOFF = np.array([4.0e6, 4.0e6, 4.0e6])  # Hz
Lnp = 1 / (CUTOFF * 53.33e-6 * 2)
L = tm.vec3(Lnp[0], Lnp[1], Lnp[2])

# For scanlines, interlacing, and overscan
OUTPUT_RESOLUTION = (2160, 2880)  #(8640, 11520) (2160, 2880) (1440, 1920) (1080, 1440) (800, 1067) (720, 960)
MAX_SPOT_SIZE = 0.9
MIN_SPOT_SIZE = 0.4
OVERSCAN_HORIZONTAL = 0.0
OVERSCAN_VERTICAL = 0.0

# For subpixel masks
MASK_PATTERN = 1  # 0 for 2-pixel (1080p), 1 for 3-pixel (1440p), 2 for 4-pixel (4k), 3 for 5-pixel (4k)

# For tiled masks
MASK_TYPE = 'slot'  # 'slot' or 'aperture'
MASK_TRIADS = 550
MASK_AMOUNT = 1.0

# Diffusion
BLUR_SIGMA = 0.05
BLUR_AMOUNT = 0.04  #0.15  #0.13


# TODO Clean this up. Parameterize options.
def simulate(img):
    # Read image
    image_height, image_width, planes = img.shape

    if USE_YIQ:
        img = gamma_to_yiq(img)

    # Horizontal low pass filter
    print('Low pass filtering...')
    img_filtered = filter_fragment(img, (image_height, SAMPLES, 3))
    # imwrite('filtered.png', linear_to_srgb(img_filtered))  # DEBUG

    # DEBUG -- Write Y, I, and Q planes to separate images
    # y_mask = np.array([True, False, False])
    # i_mask = np.array([False, True, False])
    # q_mask = np.array([False, False, True])
    # y = img_filtered.copy()
    # y[:, :, i_mask] = 0
    # y[:, :, q_mask] = 0
    # imwrite('y.png', linear_to_srgb(yiq_to_linear(y, GAMMA)))
    # i = img_filtered.copy()
    # i[:, :, y_mask] = 0.5
    # i[:, :, q_mask] = 0
    # imwrite('i.png', linear_to_srgb(yiq_to_linear(i, GAMMA)))
    # q = img_filtered.copy()
    # q[:, :, y_mask] = 0.5
    # q[:, :, i_mask] = 0
    # imwrite('q.png', linear_to_srgb(yiq_to_linear(q, GAMMA)))

    # To linear RGB
    if USE_YIQ:
        img_filtered_linear = yiq_to_linear(img_filtered, GAMMA)
    else:
        img_filtered_linear = gamma_to_linear(img_filtered, GAMMA)

    # Mimic CRT spot
    print('Simulating CRT spot...')
    img_spot = spot_fragment(img_filtered_linear, (OUTPUT_RESOLUTION[0], OUTPUT_RESOLUTION[1], 3))

    # Mask
    print('Masking...')
    # img_masked = tiled_mask(img_spot)
    # img_masked = subpixel_mask(img_spot)
    # img_masked = coverage_mask(img_spot)
    img_masked = bandlimit_mask(img_spot)
    # img_masked = additive_mask(img_spot)
    # img_masked = img_spot

    # Diffusion
    print('Blurring...')
    sigma = BLUR_SIGMA * OUTPUT_RESOLUTION[0]
    # box_radius = int(np.round((np.sqrt(3 * sigma * sigma + 1) - 1) / 2))
    # blurred = box_blur(img_masked, box_radius)
    # blurred = gaussian_blur(img_masked, sigma=sigma)
    #imwrite('blurred.png', linear_to_srgb(blurred))  # DEBUG
    # img_diffused = img_masked + (blurred - img_masked) * BLUR_AMOUNT
    img_diffused = img_masked

    # return tone_map(4*img_diffused)
    return img_diffused


# TODO Clean this up. Parameterize options.
def simulate_analytical(img):
    # Read image
    image_height, image_width, planes = img.shape

    # To linear RGB
    img_linear = gamma_to_linear(img, GAMMA)

    # Mimic CRT spot
    print('Simulating CRT spot...')
    img_spot = spot_analytical_fragment(img_linear, (OUTPUT_RESOLUTION[0], OUTPUT_RESOLUTION[1], 3))

    return img_spot


# TODO Clean this up. Parameterize options.
def simulate_fast(img):
    # Read image
    image_height, image_width, planes = img.shape

    # To linear RGB
    img_linear = gamma_to_linear(img, GAMMA)

    # Mimic CRT spot
    print('Simulating CRT spot...')
    img_spot = spot_fast_fragment(img_linear, (OUTPUT_RESOLUTION[0], OUTPUT_RESOLUTION[1], 3))

    return img_spot


def main():
    print('L = {}'.format(L))

    parser = argparse.ArgumentParser(description='Generate a CRT-simulated image')
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    # Read image
    img_original = imread(args.input)
    img_float = img_original.astype(np.float32) / 255

    # Simulate
    img_crt = simulate_fast(img_float)

    # To sRGB
    print('Color transform and save...')
    img_final_srgb = linear_to_srgb(img_crt)

    imwrite(args.output, img_final_srgb)


if __name__ == '__main__':
    main()
