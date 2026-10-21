import numpy as np
import taichi as ti
import taichi.math as tm
from imageio.v3 import imwrite
from common import texelFetch, linear_to_srgb

f16vec3 = ti.types.vector(3, ti.f16)


def parse_interlacing_mode(interlacing_type: str | int | float) -> int:
    if isinstance(interlacing_type, (int, float)):
        return int(interlacing_type)
    t = str(interlacing_type).lower().strip()
    if t == 'bob':
        return 0
    elif t == 'weave':
        return 1
    elif t in ('vga', 'doubled'):
        return 2
    elif t in ('off', 'none', 'disabled'):
        return 3
    return 0


@ti.func
def get_lines_closest(v: float, source_y: float) -> tm.vec4:
    upper_sample_y = tm.round(v * source_y)
    lower_sample_y = upper_sample_y - 1.0
    upper_distance_y = (upper_sample_y + 0.5) - v * source_y
    lower_distance_y = upper_distance_y - 1.0
    return tm.vec4(upper_sample_y, lower_sample_y, upper_distance_y, lower_distance_y)


@ti.func
def get_lines_odd(v: float, source_y: float) -> tm.vec4:
    upper_sample_y = tm.ceil(0.5 * v * source_y - 0.25) * 2.0
    lower_sample_y = upper_sample_y - 2.0
    upper_distance_y = 0.5 * ((upper_sample_y + 0.5) - v * source_y)
    lower_distance_y = upper_distance_y - 1.0
    return tm.vec4(upper_sample_y, lower_sample_y, upper_distance_y, lower_distance_y)


@ti.func
def get_lines_even(v: float, source_y: float) -> tm.vec4:
    upper_sample_y = tm.ceil(0.5 * v * source_y + 0.25) * 2.0 - 1.0
    lower_sample_y = upper_sample_y - 2.0
    upper_distance_y = 0.5 * ((upper_sample_y + 0.5) - v * source_y)
    lower_distance_y = upper_distance_y - 1.0
    return tm.vec4(upper_sample_y, lower_sample_y, upper_distance_y, lower_distance_y)


@ti.func
def get_lines_doubled(v: float, source_y: float) -> tm.vec4:
    upper_distance_y = tm.ceil(2.0 * (v * source_y) - 0.5) + 0.5 - 2.0 * (v * source_y)
    lower_distance_y = upper_distance_y - 1.0
    upper_sample_y = tm.ceil(v * source_y + 0.25) - 1.0
    lower_sample_y = tm.ceil(v * source_y - 0.25) - 1.0
    return tm.vec4(upper_sample_y, lower_sample_y, upper_distance_y, lower_distance_y)


def spot_fragment(image_in, output_dim, max_spot_size: float, min_spot_size: float,
                  overscan_horizontal: float, overscan_vertical: float,
                  interlace_mode: int = 0, odd_field_first: int = 1, frame_count: int = 0):
    (in_height, in_width, in_planes) = image_in.shape
    (out_height, out_width, out_planes) = output_dim
    field_in = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_in.from_numpy(image_in)
    field_out = ti.Vector.field(n=3, dtype=float, shape=(out_height, out_width))
    taichi_spot_fragment(field_in, field_out, max_spot_size, min_spot_size,
                         overscan_horizontal, overscan_vertical,
                         interlace_mode, odd_field_first, frame_count)
    return field_out.to_numpy()


@ti.kernel
def _kernel_spot_fragment(field_in: ti.template(), field_out: ti.template(),
                          max_spot_size: float, min_spot_size: float,
                          overscan_horizontal: float, overscan_vertical: float,
                          interlace_mode: int, odd_field_first: int, frame_count: int):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = spot_sim(vTexCoord, field_in, SourceSize, OutputSize,
                                   max_spot_size, min_spot_size,
                                   overscan_horizontal, overscan_vertical,
                                   interlace_mode, odd_field_first, frame_count)
    return


def taichi_spot_fragment(field_in, field_out, max_spot_size: float, min_spot_size: float,
                         overscan_horizontal: float, overscan_vertical: float,
                         interlace_mode: int = 0, odd_field_first: int = 1, frame_count: int = 0):
    _kernel_spot_fragment(field_in, field_out, max_spot_size, min_spot_size,
                          overscan_horizontal, overscan_vertical,
                          interlace_mode, odd_field_first, frame_count)



@ti.func
def evaluate_spot_sampled(vTexCoord: tm.vec2, img, SourceSize: tm.vec4, delta: float, lines: tm.vec4,
                          max_spot_size: float, min_spot_size: float) -> tm.vec3:
    upper_sample_y = int(lines.x)
    lower_sample_y = int(lines.y)
    upper_distance_y = lines.z
    lower_distance_y = lines.w

    output = tm.vec3(0.0)
    for sample_x in range(int(tm.round(vTexCoord.x * SourceSize.x - (max_spot_size / delta))),
                           int(tm.round(vTexCoord.x * SourceSize.x + (max_spot_size / delta)))):
        upper_sample = texelFetch(img, tm.ivec2(sample_x, upper_sample_y))
        lower_sample = texelFetch(img, tm.ivec2(sample_x, lower_sample_y))
        distance_x = delta * ((sample_x + 0.5) - vTexCoord.x * SourceSize.x)
        output += spot1(upper_sample, distance_x, upper_distance_y, max_spot_size, min_spot_size)
        output += spot1(lower_sample, distance_x, lower_distance_y, max_spot_size, min_spot_size)
    return delta * output


@ti.func
def spot_sim(vTexCoord: tm.vec2, img, SourceSize: tm.vec4, OutputSize: tm.vec4,
             max_spot_size: float, min_spot_size: float,
             overscan_horizontal: float, overscan_vertical: float,
             interlace_mode: int, odd_field_first: int, frame_count: int) -> tm.vec3:
    # Overscan
    vTexCoord = (1.0 - tm.vec2(overscan_horizontal, overscan_vertical)) * (vTexCoord - 0.5) + 0.5

    delta = OutputSize.x * OutputSize.w * SourceSize.y * SourceSize.z * (1 - overscan_vertical) / (1 - overscan_horizontal)
    if interlace_mode <= 1 and SourceSize.y > 300.0:
        delta = 0.5 * delta
    elif interlace_mode == 2 and SourceSize.y < 350.0:
        delta = 2.0 * delta

    output = tm.vec3(0.0)
    if interlace_mode == 2 and SourceSize.y < 350.0:
        lines = get_lines_doubled(vTexCoord.y, SourceSize.y)
        output = evaluate_spot_sampled(vTexCoord, img, SourceSize, delta, lines, max_spot_size, min_spot_size)
    elif interlace_mode == 3 or SourceSize.y <= 300.0:
        lines = get_lines_closest(vTexCoord.y, SourceSize.y)
        output = evaluate_spot_sampled(vTexCoord, img, SourceSize, delta, lines, max_spot_size, min_spot_size)
    elif interlace_mode == 1:
        lines_odd = get_lines_odd(vTexCoord.y, SourceSize.y)
        lines_even = get_lines_even(vTexCoord.y, SourceSize.y)
        output = 0.5 * (evaluate_spot_sampled(vTexCoord, img, SourceSize, delta, lines_odd, max_spot_size, min_spot_size) +
                        evaluate_spot_sampled(vTexCoord, img, SourceSize, delta, lines_even, max_spot_size, min_spot_size))
    else:
        phase = 1 if odd_field_first == 0 else 0
        if (frame_count + phase) % 2 == 0:
            lines = get_lines_odd(vTexCoord.y, SourceSize.y)
            output = evaluate_spot_sampled(vTexCoord, img, SourceSize, delta, lines, max_spot_size, min_spot_size)
        else:
            lines = get_lines_even(vTexCoord.y, SourceSize.y)
            output = evaluate_spot_sampled(vTexCoord, img, SourceSize, delta, lines, max_spot_size, min_spot_size)
    return output


@ti.func
def spot1(sample, distance_x, distance_y, max_spot_size: float, min_spot_size: float):
    width_rcp = 1.0 / tm.mix(max_spot_size * min_spot_size, max_spot_size, tm.sqrt(sample))
    x = tm.clamp(abs(distance_x) * width_rcp, 0.0, 1.0)
    y = tm.clamp(abs(distance_y) * width_rcp, 0.0, 1.0)
    return sample * max_spot_size * width_rcp**2 * (0.5 * tm.cos(np.pi * x) + 0.5) * (0.5 * tm.cos(np.pi * y) + 0.5)


@ti.func
def spot2(sample, distance_x, distance_y, max_spot_size: float, min_spot_size: float):
    width_rcp = 1.0 / tm.mix(max_spot_size * min_spot_size, max_spot_size, tm.sqrt(sample))
    x = tm.min(abs(distance_x) * width_rcp - 0.5, 0.5)
    y = tm.min(abs(distance_y) * width_rcp - 0.5, 0.5)
    return sample * max_spot_size * width_rcp**2 * (2.0 * (x * abs(x) - x) + 0.5) * (2.0 * (y * abs(y) - y) + 0.5)


@ti.func
def spot3(sample, distance_x, distance_y, max_spot_size: float, min_spot_size: float):
    width_rcp = 1.0 / tm.mix(max_spot_size * min_spot_size, max_spot_size, tm.sqrt(sample))
    x = tm.clamp(abs(distance_x) * width_rcp, 0.0, 1.0)
    y = tm.clamp(abs(distance_y) * width_rcp, 0.0, 1.0)
    return sample * max_spot_size * width_rcp**2 * ((x * x) * (2.0 * x - 3.0) + 1.0) * ((y * y) * (2.0 * y - 3.0) + 1.0)


@ti.func
def spot_sim_gauss(vTexCoord: tm.vec2, img, SourceSize: tm.vec4, OutputSize: tm.vec4,
                   max_spot_size: float, min_spot_size: float,
                   overscan_horizontal: float, overscan_vertical: float) -> tm.vec3:
    # Overscan
    vTexCoord = (1.0 - tm.vec2(overscan_horizontal, overscan_vertical)) * (vTexCoord - 0.5) + 0.5

    # Distance units (including for delta) are *scanlines heights*. This means
    # we need to adjust x distances by the aspect ratio. Overscan needs to be
    # taken into account because it can change the aspect ratio.
    delta = OutputSize.x * OutputSize.w * SourceSize.y * SourceSize.z * (1 - overscan_vertical) / (1 - overscan_horizontal)
    upper_sample_y = int(tm.round(vTexCoord.y * SourceSize.y))
    lower_sample_y = upper_sample_y - 1
    # The >= is important here. We need to make sure that we work in the same
    # direction that the upper_sample_y is rounded from.
    third_sample_y = upper_sample_y - 2 if tm.fract(vTexCoord.y * SourceSize.y) >= 0.5 else upper_sample_y + 1
    upper_distance_y = (upper_sample_y + 0.5) - vTexCoord.y * SourceSize.y
    lower_distance_y = (lower_sample_y + 0.5) - vTexCoord.y * SourceSize.y
    third_distance_y = (third_sample_y + 0.5) - vTexCoord.y * SourceSize.y

    output = tm.vec3(0.0)
    max_sigma = max_spot_size / (2 * tm.sqrt(2 * tm.log(2)))
    for sample_x in range(int(tm.round(vTexCoord.x * SourceSize.x - ((3 * max_sigma) / delta))),
                           int(tm.round(vTexCoord.x * SourceSize.x + ((3 * max_sigma) / delta)))):
        upper_sample = texelFetch(img, tm.ivec2(sample_x, upper_sample_y))
        lower_sample = texelFetch(img, tm.ivec2(sample_x, lower_sample_y))
        third_sample = texelFetch(img, tm.ivec2(sample_x, third_sample_y))
        distance_x = delta * ((sample_x + 0.5) - vTexCoord.x * SourceSize.x)
        output += spot_gauss(upper_sample, distance_x, upper_distance_y, max_spot_size, min_spot_size)
        output += spot_gauss(lower_sample, distance_x, lower_distance_y, max_spot_size, min_spot_size)
        output += spot_gauss(third_sample, distance_x, third_distance_y, max_spot_size, min_spot_size)
    output = delta * output
    # Rescale output so that the maximum value is one.
    peak = (tm.sqrt(np.pi / 2) * abs(max_sigma)) / (max_sigma * max_sigma) * (1.0 + 2.0 * tm.exp(-1.0 / (2 * max_sigma * max_sigma)))
    return output / peak


@ti.func
def spot_sim_gauss4(vTexCoord: tm.vec2, img, SourceSize: tm.vec4, OutputSize: tm.vec4,
                    max_spot_size: float, min_spot_size: float,
                    overscan_horizontal: float, overscan_vertical: float) -> tm.vec3:
    # Overscan
    vTexCoord = (1.0 - tm.vec2(overscan_horizontal, overscan_vertical)) * (vTexCoord - 0.5) + 0.5

    # Distance units (including for delta) are *scanlines heights*. This means
    # we need to adjust x distances by the aspect ratio. Overscan needs to be
    # taken into account because it can change the aspect ratio.
    delta = OutputSize.x * OutputSize.w * SourceSize.y * SourceSize.z * (1 - overscan_vertical) / (1 - overscan_horizontal)

    upper_sample_y = int(tm.round(vTexCoord.y * SourceSize.y))
    upper_sample_y2 = upper_sample_y + 1
    lower_sample_y = upper_sample_y - 1
    lower_sample_y2 = upper_sample_y - 2

    upper_distance_y = (upper_sample_y + 0.5) - vTexCoord.y * SourceSize.y
    upper_distance_y2 = (upper_sample_y2 + 0.5) - vTexCoord.y * SourceSize.y
    lower_distance_y = (lower_sample_y + 0.5) - vTexCoord.y * SourceSize.y
    lower_distance_y2 = (lower_sample_y2 + 0.5) - vTexCoord.y * SourceSize.y

    output = tm.vec3(0.0)
    max_sigma = max_spot_size / (2 * tm.sqrt(2 * tm.log(2)))
    for sample_x in range(int(tm.round(vTexCoord.x * SourceSize.x - ((3 * max_sigma) / delta))),
                           int(tm.round(vTexCoord.x * SourceSize.x + ((3 * max_sigma) / delta)))):
        upper_sample = texelFetch(img, tm.ivec2(sample_x, upper_sample_y))
        upper_sample2 = texelFetch(img, tm.ivec2(sample_x, upper_sample_y2))
        lower_sample = texelFetch(img, tm.ivec2(sample_x, lower_sample_y))
        lower_sample2 = texelFetch(img, tm.ivec2(sample_x, lower_sample_y2))
        distance_x = delta * ((sample_x + 0.5) - vTexCoord.x * SourceSize.x)
        output += spot_gauss(upper_sample, distance_x, upper_distance_y, max_spot_size, min_spot_size)
        output += spot_gauss(upper_sample2, distance_x, upper_distance_y2, max_spot_size, min_spot_size)
        output += spot_gauss(lower_sample, distance_x, lower_distance_y, max_spot_size, min_spot_size)
        output += spot_gauss(lower_sample2, distance_x, lower_distance_y2, max_spot_size, min_spot_size)
    output = delta * output
    # Rescale output so that the maximum value is one.
    peak = (tm.sqrt(np.pi / 2) * abs(max_sigma) / (max_sigma * max_sigma) * tm.exp(-2 / (max_sigma * max_sigma)) * \
            (2 * tm.exp(3 / (2 * max_sigma * max_sigma)) + tm.exp(2 / (max_sigma * max_sigma)) + 1))
    return output / peak


@ti.func
def spot_sim_gauss5(vTexCoord: tm.vec2, img, SourceSize: tm.vec4, OutputSize: tm.vec4,
                    max_spot_size: float, min_spot_size: float,
                    overscan_horizontal: float, overscan_vertical: float) -> tm.vec3:
    # Overscan
    vTexCoord = (1.0 - tm.vec2(overscan_horizontal, overscan_vertical)) * (vTexCoord - 0.5) + 0.5

    # Distance units (including for delta) are *scanlines heights*. This means
    # we need to adjust x distances by the aspect ratio. Overscan needs to be
    # taken into account because it can change the aspect ratio.
    delta = OutputSize.x * OutputSize.w * SourceSize.y * SourceSize.z * (1 - overscan_vertical) / (1 - overscan_horizontal)

    upper_sample_y = int(tm.round(vTexCoord.y * SourceSize.y))
    upper_sample_y2 = upper_sample_y + 1
    lower_sample_y = upper_sample_y - 1
    lower_sample_y2 = upper_sample_y - 2

    upper_distance_y = (upper_sample_y + 0.5) - vTexCoord.y * SourceSize.y
    upper_distance_y2 = (upper_sample_y2 + 0.5) - vTexCoord.y * SourceSize.y
    lower_distance_y = (lower_sample_y + 0.5) - vTexCoord.y * SourceSize.y
    lower_distance_y2 = (lower_sample_y2 + 0.5) - vTexCoord.y * SourceSize.y

    output = tm.vec3(0.0)
    max_sigma = max_spot_size / (2 * tm.sqrt(2 * tm.log(2)))
    for sample_x in range(int(tm.round(vTexCoord.x * SourceSize.x - ((3 * max_sigma) / delta))),
                           int(tm.round(vTexCoord.x * SourceSize.x + ((3 * max_sigma) / delta)))):
        upper_sample = texelFetch(img, tm.ivec2(sample_x, upper_sample_y))
        upper_sample2 = texelFetch(img, tm.ivec2(sample_x, upper_sample_y2))
        lower_sample = texelFetch(img, tm.ivec2(sample_x, lower_sample_y))
        lower_sample2 = texelFetch(img, tm.ivec2(sample_x, lower_sample_y2))
        distance_x = delta * ((sample_x + 0.5) - vTexCoord.x * SourceSize.x)
        output += spot_gauss(upper_sample, distance_x, upper_distance_y, max_spot_size, min_spot_size)
        output += spot_gauss(upper_sample2, distance_x, upper_distance_y2, max_spot_size, min_spot_size)
        output += spot_gauss(lower_sample, distance_x, lower_distance_y, max_spot_size, min_spot_size)
        output += spot_gauss(lower_sample2, distance_x, lower_distance_y2, max_spot_size, min_spot_size)
    output = delta * output
    # Rescale output so that the maximum value is one.
    peak = (tm.sqrt(np.pi / 2) * abs(max_sigma)) / (max_sigma * max_sigma) * (1.0 + 2.0 * tm.exp(-1.0 / (2 * max_sigma * max_sigma)))
    return output / peak


@ti.func
def spot_gauss(sample, distance_x, distance_y, max_spot_size: float, min_spot_size: float):
    width = tm.mix(max_spot_size * min_spot_size, max_spot_size, tm.sqrt(sample))
    sigma = width / (2 * tm.sqrt(2 * tm.log(2)))
    x = distance_x
    y = distance_y
    # We don't need pi, as in the normal distribution, because we are
    # rescaling later (we can ignore constant factors).
    return sample / (2 * sigma * sigma) * tm.exp(-(x * x + y * y) / (2 * sigma * sigma))


def spot_analytical_fragment(image_in, output_dim, max_spot_size: float, min_spot_size: float,
                             overscan_horizontal: float, overscan_vertical: float,
                             interlace_mode: int = 0, odd_field_first: int = 1, frame_count: int = 0):
    (in_height, in_width, in_planes) = image_in.shape
    (out_height, out_width, out_planes) = output_dim
    field_in = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_in.from_numpy(image_in)
    field_out = ti.Vector.field(n=3, dtype=float, shape=(out_height, out_width))
    taichi_spot_analytical_fragment(field_in, field_out, max_spot_size, min_spot_size,
                                    overscan_horizontal, overscan_vertical,
                                    interlace_mode, odd_field_first, frame_count)
    return field_out.to_numpy()


@ti.kernel
def _kernel_spot_analytical_fragment(field_in: ti.template(), field_out: ti.template(),
                                      max_spot_size: float, min_spot_size: float,
                                      overscan_horizontal: float, overscan_vertical: float,
                                      interlace_mode: int, odd_field_first: int, frame_count: int):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = spot_sim_analytical(vTexCoord, field_in, SourceSize, OutputSize,
                                              max_spot_size, min_spot_size,
                                              overscan_horizontal, overscan_vertical,
                                              interlace_mode, odd_field_first, frame_count)
    return


def taichi_spot_analytical_fragment(field_in, field_out, max_spot_size: float, min_spot_size: float,
                                    overscan_horizontal: float, overscan_vertical: float,
                                    interlace_mode: int = 0, odd_field_first: int = 1, frame_count: int = 0):
    _kernel_spot_analytical_fragment(field_in, field_out, max_spot_size, min_spot_size,
                                     overscan_horizontal, overscan_vertical,
                                     interlace_mode, odd_field_first, frame_count)



@ti.func
def evaluate_spot_analytical(vTexCoord: tm.vec2, img, SourceSize: tm.vec4, delta: float, lines: tm.vec4,
                             max_spot_size: float, min_spot_size: float) -> tm.vec3:
    upper_sample_y = int(lines.x)
    lower_sample_y = int(lines.y)
    upper_distance_y = lines.z
    lower_distance_y = lines.w

    output = tm.vec3(0.0)
    x = vTexCoord.x * SourceSize.x * delta
    for sample_x in range(int(tm.floor(vTexCoord.x * SourceSize.x - (max_spot_size / delta))),
                           int(tm.floor(vTexCoord.x * SourceSize.x + (max_spot_size / delta))) + 1):
        upper_sample = texelFetch(img, tm.ivec2(sample_x, upper_sample_y))
        lower_sample = texelFetch(img, tm.ivec2(sample_x, lower_sample_y))
        x0 = delta * sample_x
        x1 = x0 + delta
        output += spot_analytical(upper_sample, x, x0, x1, upper_distance_y, max_spot_size, min_spot_size)
        output += spot_analytical(lower_sample, x, x0, x1, lower_distance_y, max_spot_size, min_spot_size)
    return output


@ti.func
def spot_sim_analytical(vTexCoord: tm.vec2, img, SourceSize: tm.vec4, OutputSize: tm.vec4,
                        max_spot_size: float, min_spot_size: float,
                        overscan_horizontal: float, overscan_vertical: float,
                        interlace_mode: int, odd_field_first: int, frame_count: int) -> tm.vec3:
    # Overscan
    vTexCoord = (1.0 - tm.vec2(overscan_horizontal, overscan_vertical)) * (vTexCoord - 0.5) + 0.5

    delta = OutputSize.x * OutputSize.w * SourceSize.y * SourceSize.z * (1 - overscan_vertical) / (1 - overscan_horizontal)
    if interlace_mode <= 1 and SourceSize.y > 300.0:
        delta = 0.5 * delta
    elif interlace_mode == 2 and SourceSize.y < 350.0:
        delta = 2.0 * delta

    output = tm.vec3(0.0)
    if interlace_mode == 2 and SourceSize.y < 350.0:
        lines = get_lines_doubled(vTexCoord.y, SourceSize.y)
        output = evaluate_spot_analytical(vTexCoord, img, SourceSize, delta, lines, max_spot_size, min_spot_size)
    elif interlace_mode == 3 or SourceSize.y <= 300.0:
        lines = get_lines_closest(vTexCoord.y, SourceSize.y)
        output = evaluate_spot_analytical(vTexCoord, img, SourceSize, delta, lines, max_spot_size, min_spot_size)
    elif interlace_mode == 1:
        lines_odd = get_lines_odd(vTexCoord.y, SourceSize.y)
        lines_even = get_lines_even(vTexCoord.y, SourceSize.y)
        output = 0.5 * (evaluate_spot_analytical(vTexCoord, img, SourceSize, delta, lines_odd, max_spot_size, min_spot_size) +
                        evaluate_spot_analytical(vTexCoord, img, SourceSize, delta, lines_even, max_spot_size, min_spot_size))
    else:
        phase = 1 if odd_field_first == 0 else 0
        if (frame_count + phase) % 2 == 0:
            lines = get_lines_odd(vTexCoord.y, SourceSize.y)
            output = evaluate_spot_analytical(vTexCoord, img, SourceSize, delta, lines, max_spot_size, min_spot_size)
        else:
            lines = get_lines_even(vTexCoord.y, SourceSize.y)
            output = evaluate_spot_analytical(vTexCoord, img, SourceSize, delta, lines, max_spot_size, min_spot_size)
    return output


@ti.func
def spot_analytical(sample: tm.vec3, x: float, x0_: float, x1_: float, y_: float,
                    max_spot_size: float, min_spot_size: float):
    w = tm.mix(max_spot_size * min_spot_size, max_spot_size, tm.sqrt(sample))
    x0 = tm.clamp((x - x0_) / w, -1.0, 1.0)
    x1 = tm.clamp((x - x1_) / w, -1.0, 1.0)
    y = tm.clamp(abs(y_) / w, 0.0, 1.0)
    return max_spot_size / (2.0 * np.pi) * sample * 1.0 / w * (0.5 * tm.cos(np.pi * y) + 0.5) * \
        (np.pi * x0 + tm.sin(np.pi * x0) - np.pi * x1 - tm.sin(np.pi * x1))


@ti.func
def spot_analytical2(sample: tm.vec3, x: float, x0_: float, x1_: float, y_: float,
                     max_spot_size: float, min_spot_size: float):
    w = tm.mix(max_spot_size * min_spot_size, max_spot_size, tm.sqrt(sample))
    x0 = tm.clamp((x - x0_) / w, -1.0, 1.0)
    x1 = tm.clamp((x - x1_) / w, -1.0, 1.0)
    y = tm.clamp(abs(y_) / w, 0.0, 1.0)
    return max_spot_size * sample * 1.0 / w * ((y * y) * (2.0 * y - 3.0) + 1.0) * (
        ((x0 * x0) * (tm.sign(x0) * 0.5 * (x0 * x0) - x0) + x0) -
        ((x1 * x1) * (tm.sign(x1) * 0.5 * (x1 * x1) - x1) + x1)
    )


def spot_fast_fragment(image_in, output_dim, max_spot_size: float, min_spot_size: float,
                       overscan_horizontal: float, overscan_vertical: float,
                       interlace_mode: int = 0, odd_field_first: int = 1, frame_count: int = 0):
    (in_height, in_width, in_planes) = image_in.shape
    (out_height, out_width, out_planes) = output_dim
    aspect_ratio = float(out_width) / float(out_height)
    field_in = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_in.from_numpy(image_in)
    field_intermediate = ti.Vector.field(n=3, dtype=float, shape=(in_height, out_width))
    field_out = ti.Vector.field(n=3, dtype=float, shape=(out_height, out_width))

    taichi_spot_fast_horizontal_fragment(field_in, field_intermediate, max_spot_size, min_spot_size,
                                         overscan_horizontal, overscan_vertical, aspect_ratio, interlace_mode)
    imwrite("intermediate.png", linear_to_srgb(field_intermediate.to_numpy()))
    taichi_spot_fast_vertical_fragment(field_intermediate, field_out, max_spot_size, min_spot_size,
                                       overscan_horizontal, overscan_vertical,
                                       interlace_mode, odd_field_first, frame_count)
    return field_out.to_numpy()


@ti.kernel
def _kernel_spot_fast_horizontal_fragment(field_in: ti.template(), field_out: ti.template(),
                                          max_spot_size: float, min_spot_size: float,
                                          overscan_horizontal: float, overscan_vertical: float,
                                          aspect_ratio: float, interlace_mode: int):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)

    delta = aspect_ratio * SourceSize.y * SourceSize.z * (1 - overscan_vertical) / (1 - overscan_horizontal)
    if interlace_mode <= 1 and SourceSize.y > 300.0:
        delta = 0.5 * delta
    elif interlace_mode == 2 and SourceSize.y < 350.0:
        delta = 2.0 * delta

    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = spot_sim_fast_horizontal(vTexCoord, field_in, SourceSize, OutputSize, delta, max_spot_size, min_spot_size)
    return


def taichi_spot_fast_horizontal_fragment(field_in, field_out, max_spot_size: float, min_spot_size: float,
                                         overscan_horizontal: float, overscan_vertical: float,
                                         aspect_ratio: float, interlace_mode: int = 0):
    _kernel_spot_fast_horizontal_fragment(field_in, field_out, max_spot_size, min_spot_size,
                                          overscan_horizontal, overscan_vertical,
                                          aspect_ratio, interlace_mode)


@ti.kernel
def _kernel_spot_fast_vertical_fragment(field_in: ti.template(), field_out: ti.template(),
                                        max_spot_size: float, min_spot_size: float,
                                        overscan_horizontal: float, overscan_vertical: float,
                                        interlace_mode: int, odd_field_first: int, frame_count: int):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)

    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = spot_sim_fast_vertical(vTexCoord, field_in, SourceSize, OutputSize,
                                                 max_spot_size, min_spot_size,
                                                 overscan_horizontal, overscan_vertical,
                                                 interlace_mode, odd_field_first, frame_count)
    return


def taichi_spot_fast_vertical_fragment(field_in, field_out, max_spot_size: float, min_spot_size: float,
                                       overscan_horizontal: float, overscan_vertical: float,
                                       interlace_mode: int = 0, odd_field_first: int = 1, frame_count: int = 0):
    _kernel_spot_fast_vertical_fragment(field_in, field_out, max_spot_size, min_spot_size,
                                        overscan_horizontal, overscan_vertical,
                                        interlace_mode, odd_field_first, frame_count)



@ti.func
def evaluate_spot_fast(vTexCoord: tm.vec2, img, SourceSize: tm.vec4, lines: tm.vec4,
                       max_spot_size: float, min_spot_size: float) -> tm.vec3:
    upper_sample_y = int(lines.x)
    lower_sample_y = int(lines.y)
    upper_distance_y = lines.z
    lower_distance_y = lines.w

    u_coord = int(vTexCoord.x * SourceSize.x)
    upper_sample = texelFetch(img, tm.ivec2(u_coord, upper_sample_y))
    lower_sample = texelFetch(img, tm.ivec2(u_coord, lower_sample_y))
    return spot_fast_vertical(upper_sample, upper_distance_y, max_spot_size, min_spot_size) + \
           spot_fast_vertical(lower_sample, lower_distance_y, max_spot_size, min_spot_size)


@ti.func
def spot_sim_fast_vertical(vTexCoord: tm.vec2, img, SourceSize: tm.vec4, OutputSize: tm.vec4,
                           max_spot_size: float, min_spot_size: float,
                           overscan_horizontal: float, overscan_vertical: float,
                           interlace_mode: int, odd_field_first: int, frame_count: int) -> tm.vec3:
    # Overscan
    vTexCoord = (1.0 - tm.vec2(overscan_horizontal, overscan_vertical)) * (vTexCoord - 0.5) + 0.5

    output = tm.vec3(0.0)
    if interlace_mode == 2 and SourceSize.y < 350.0:
        lines = get_lines_doubled(vTexCoord.y, SourceSize.y)
        output = evaluate_spot_fast(vTexCoord, img, SourceSize, lines, max_spot_size, min_spot_size)
    elif interlace_mode == 3 or SourceSize.y <= 300.0:
        lines = get_lines_closest(vTexCoord.y, SourceSize.y)
        output = evaluate_spot_fast(vTexCoord, img, SourceSize, lines, max_spot_size, min_spot_size)
    elif interlace_mode == 1:
        lines_odd = get_lines_odd(vTexCoord.y, SourceSize.y)
        lines_even = get_lines_even(vTexCoord.y, SourceSize.y)
        output = 0.5 * (evaluate_spot_fast(vTexCoord, img, SourceSize, lines_odd, max_spot_size, min_spot_size) +
                        evaluate_spot_fast(vTexCoord, img, SourceSize, lines_even, max_spot_size, min_spot_size))
    else:
        phase = 1 if odd_field_first == 0 else 0
        if (frame_count + phase) % 2 == 0:
            lines = get_lines_odd(vTexCoord.y, SourceSize.y)
            output = evaluate_spot_fast(vTexCoord, img, SourceSize, lines, max_spot_size, min_spot_size)
        else:
            lines = get_lines_even(vTexCoord.y, SourceSize.y)
            output = evaluate_spot_fast(vTexCoord, img, SourceSize, lines, max_spot_size, min_spot_size)
    return output


@ti.func
def spot_fast_horizontal(sample: tm.vec3, x: float, x0_: float, x1_: float,
                         max_spot_size: float, min_spot_size: float):
    w = tm.mix(max_spot_size * min_spot_size, max_spot_size, tm.sqrt(sample))
    x0 = tm.clamp((x - x0_) / w, -1.0, 1.0)
    x1 = tm.clamp((x - x1_) / w, -1.0, 1.0)
    return 1 / (2.0 * np.pi) * sample * (np.pi * x0 + tm.sin(np.pi * x0) - np.pi * x1 - tm.sin(np.pi * x1))


@ti.func
def spot_fast_vertical(sample: tm.vec3, y_: float, max_spot_size: float, min_spot_size: float):
    w = tm.mix(max_spot_size * min_spot_size, max_spot_size, tm.sqrt(sample))
    y = tm.clamp(abs(y_) / w, 0.0, 1.0)
    return sample * max_spot_size / w * (0.5 * tm.cos(np.pi * y) + 0.5)


@ti.func
def spot_sim_f16(vTexCoord: tm.vec2, img, SourceSize: tm.vec4, OutputSize: tm.vec4,
                 max_spot_size: float, min_spot_size: float,
                 overscan_horizontal: float, overscan_vertical: float) -> tm.vec3:
    # Overscan
    vTexCoord = (1.0 - tm.vec2(overscan_horizontal, overscan_vertical)) * (vTexCoord - 0.5) + 0.5

    # Distance units (including for delta) are *scanlines heights*. This means
    # we need to adjust x distances by the aspect ratio. Overscan needs to be
    # taken into account because it can change the aspect ratio.
    upper_sample_y = int(tm.round(vTexCoord.y * SourceSize.y))
    lower_sample_y = upper_sample_y - 1
    delta = OutputSize.x * OutputSize.w * SourceSize.y * SourceSize.z * (1 - overscan_vertical) / (1 - overscan_horizontal)
    upper_distance_y = ti.f16((upper_sample_y + 0.5) - vTexCoord.y * SourceSize.y)
    lower_distance_y = ti.f16((lower_sample_y + 0.5) - vTexCoord.y * SourceSize.y)

    output = tm.vec3(0.0)
    for sample_x in range(int(tm.round(vTexCoord.x * SourceSize.x - (max_spot_size / delta))),
                           int(tm.round(vTexCoord.x * SourceSize.x + (max_spot_size / delta)))):
        upper_sample = f16vec3(texelFetch(img, tm.ivec2(sample_x, upper_sample_y)))
        lower_sample = f16vec3(texelFetch(img, tm.ivec2(sample_x, lower_sample_y)))
        distance_x = ti.f16(delta * ((sample_x + 0.5) - vTexCoord.x * SourceSize.x))
        output += spot3_float16(upper_sample, distance_x, upper_distance_y, max_spot_size, min_spot_size)
        output += spot3_float16(lower_sample, distance_x, lower_distance_y, max_spot_size, min_spot_size)
    return delta * output


@ti.func
def spot3_float16(sample: f16vec3, distance_x: f16vec3, distance_y: f16vec3,
                  max_spot_size: float, min_spot_size: float) -> f16vec3:
    width_rcp = ti.f16(1.0) / tm.mix(max_spot_size * min_spot_size, max_spot_size, tm.sqrt(sample))
    x = tm.clamp(abs(distance_x) * width_rcp, ti.f16(0.0), ti.f16(1.0))
    y = tm.clamp(abs(distance_y) * width_rcp, ti.f16(0.0), ti.f16(1.0))
    return sample * max_spot_size * width_rcp**2 * \
            ((x * x) * (ti.f16(2.0) * x - ti.f16(3.0)) + ti.f16(1.0)) * \
            ((y * y) * (ti.f16(2.0) * y - ti.f16(3.0)) + ti.f16(1.0))


def spot_fast2_fragment(image_in, output_dim, max_spot_size: float, min_spot_size: float,
                        overscan_horizontal: float, overscan_vertical: float,
                        interlace_mode: int = 0, odd_field_first: int = 1, frame_count: int = 0):
    (in_height, in_width, in_planes) = image_in.shape
    (out_height, out_width, out_planes) = output_dim
    aspect_ratio = float(out_width) / float(out_height)
    field_in = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_in.from_numpy(image_in)
    energy_intermediate = ti.Vector.field(n=3, dtype=float, shape=(in_height, out_width))
    width_intermediate = ti.Vector.field(n=3, dtype=float, shape=(in_height, out_width))
    field_out = ti.Vector.field(n=3, dtype=float, shape=(out_height, out_width))

    taichi_spot_fast2_horizontal_fragment(field_in, energy_intermediate, width_intermediate,
                                          max_spot_size, min_spot_size,
                                          overscan_horizontal, overscan_vertical, aspect_ratio,
                                          interlace_mode)
    imwrite("intermediate.png", linear_to_srgb(energy_intermediate.to_numpy()))
    taichi_spot_fast2_vertical_fragment(energy_intermediate, width_intermediate, field_out,
                                        max_spot_size, min_spot_size,
                                        overscan_horizontal, overscan_vertical,
                                        interlace_mode, odd_field_first, frame_count)
    return field_out.to_numpy()


@ti.kernel
def _kernel_spot_fast2_horizontal_fragment(field_in: ti.template(), energy_out: ti.template(), width_out: ti.template(),
                                           max_spot_size: float, min_spot_size: float,
                                           overscan_horizontal: float, overscan_vertical: float,
                                           aspect_ratio: float, interlace_mode: int):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = energy_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)

    delta = aspect_ratio * SourceSize.y * SourceSize.z * (1 - overscan_vertical) / (1 - overscan_horizontal)
    if interlace_mode <= 1 and SourceSize.y > 300.0:
        delta = 0.5 * delta
    elif interlace_mode == 2 and SourceSize.y < 350.0:
        delta = 2.0 * delta

    for y, x in energy_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        energy_out[y, x], width_out[y, x] = spot_sim_fast2_horizontal(vTexCoord, field_in, SourceSize, OutputSize, delta,
                                                                      max_spot_size, min_spot_size)
    return


def taichi_spot_fast2_horizontal_fragment(field_in, energy_out, width_out,
                                          max_spot_size: float, min_spot_size: float,
                                          overscan_horizontal: float, overscan_vertical: float,
                                          aspect_ratio: float, interlace_mode: int = 0):
    _kernel_spot_fast2_horizontal_fragment(field_in, energy_out, width_out,
                                           max_spot_size, min_spot_size,
                                           overscan_horizontal, overscan_vertical,
                                           aspect_ratio, interlace_mode)


@ti.kernel
def _kernel_spot_fast2_vertical_fragment(energy_in: ti.template(), width_in: ti.template(), field_out: ti.template(),
                                         max_spot_size: float, min_spot_size: float,
                                         overscan_horizontal: float, overscan_vertical: float,
                                         interlace_mode: int, odd_field_first: int, frame_count: int):
    (in_height, in_width) = energy_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)

    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = spot_sim_fast2_vertical(vTexCoord, energy_in, width_in, SourceSize, OutputSize,
                                                  max_spot_size, min_spot_size,
                                                  overscan_horizontal, overscan_vertical,
                                                  interlace_mode, odd_field_first, frame_count)
    return


def taichi_spot_fast2_vertical_fragment(energy_in, width_in, field_out,
                                        max_spot_size: float, min_spot_size: float,
                                        overscan_horizontal: float, overscan_vertical: float,
                                        interlace_mode: int = 0, odd_field_first: int = 1, frame_count: int = 0):
    _kernel_spot_fast2_vertical_fragment(energy_in, width_in, field_out,
                                         max_spot_size, min_spot_size,
                                         overscan_horizontal, overscan_vertical,
                                         interlace_mode, odd_field_first, frame_count)



@ti.func
def spot_sim_fast2_horizontal(vTexCoord: tm.vec2, img, SourceSize: tm.vec4, OutputSize: tm.vec4, delta: float,
                              max_spot_size: float, min_spot_size: float) -> tm.vec3:
    sample_y = int(vTexCoord.y * SourceSize.y)
    energy = tm.vec3(0.0)
    moment = tm.vec3(0.0)
    x = vTexCoord.x * SourceSize.x * delta
    for sample_x in range(int(tm.floor(vTexCoord.x * SourceSize.x - (max_spot_size / delta))),
                           int(tm.floor(vTexCoord.x * SourceSize.x + (max_spot_size / delta))) + 1):
        sample = texelFetch(img, tm.ivec2(sample_x, sample_y))
        x0 = delta * sample_x
        x1 = x0 + delta
        piece_width = tm.mix(max_spot_size * min_spot_size, max_spot_size, tm.sqrt(sample))
        piece_energy = spot_fast2_horizontal(sample, piece_width, x, x0, x1)
        energy += piece_energy
        moment += piece_width * piece_energy
    width = min(ti.select(energy > 0.0, moment / energy, min_spot_size * max_spot_size), max_spot_size)
    return (energy, width)


@ti.func
def spot_sim_fast3_horizontal(vTexCoord: tm.vec2, img, SourceSize: tm.vec4, OutputSize: tm.vec4, delta: float,
                              max_spot_size: float, min_spot_size: float) -> tm.vec3:
    sample_y = int(vTexCoord.y * SourceSize.y)
    energy = tm.vec3(0.0)
    moment = tm.vec3(0.0)
    x = vTexCoord.x * SourceSize.x * delta
    for sample_x in range(int(tm.floor(vTexCoord.x * SourceSize.x - (max_spot_size / delta))),
                           int(tm.floor(vTexCoord.x * SourceSize.x + (max_spot_size / delta))) + 1):
        sample = texelFetch(img, tm.ivec2(sample_x, sample_y))
        x0 = delta * sample_x
        x1 = x0 + delta
        piece_width = tm.mix(max_spot_size * min_spot_size, max_spot_size, tm.sqrt(sample))
        piece_energy = spot_fast2_horizontal(sample, piece_width, x, x0, x1)
        energy += piece_energy
        moment += piece_width * piece_width * piece_energy
    width = min(ti.select(energy > 0.0, tm.sqrt(moment / energy), min_spot_size * max_spot_size), max_spot_size)
    return (energy, width)


@ti.func
def spot_sim_fast4_horizontal(vTexCoord: tm.vec2, img, SourceSize: tm.vec4, OutputSize: tm.vec4, delta: float,
                              max_spot_size: float, min_spot_size: float) -> tm.vec3:
    sample_y = int(vTexCoord.y * SourceSize.y)
    energy = tm.vec3(0.0)
    moment = tm.vec3(0.0)
    x = vTexCoord.x * SourceSize.x * delta
    for sample_x in range(int(tm.floor(vTexCoord.x * SourceSize.x - (max_spot_size / delta))),
                           int(tm.floor(vTexCoord.x * SourceSize.x + (max_spot_size / delta))) + 1):
        sample = texelFetch(img, tm.ivec2(sample_x, sample_y))
        x0 = delta * sample_x
        x1 = x0 + delta
        piece_width = tm.mix(max_spot_size * min_spot_size, max_spot_size, tm.sqrt(sample))
        piece_energy = spot_fast2_horizontal(sample, piece_width, x, x0, x1)
        energy += piece_energy
        moment += piece_energy / piece_width
    width = min(ti.select(energy > 0.0, energy / moment, min_spot_size * max_spot_size), max_spot_size)
    return (energy, width)


@ti.func
def evaluate_spot_fast2(vTexCoord: tm.vec2, img, widths, SourceSize: tm.vec4, lines: tm.vec4,
                        max_spot_size: float, min_spot_size: float) -> tm.vec3:
    upper_sample_y = int(lines.x)
    lower_sample_y = int(lines.y)
    upper_distance_y = lines.z
    lower_distance_y = lines.w

    u_coord = int(vTexCoord.x * SourceSize.x)
    upper_sample = texelFetch(img, tm.ivec2(u_coord, upper_sample_y))
    lower_sample = texelFetch(img, tm.ivec2(u_coord, lower_sample_y))
    upper_width = texelFetch(widths, tm.ivec2(u_coord, upper_sample_y))
    lower_width = texelFetch(widths, tm.ivec2(u_coord, lower_sample_y))

    min_width = min_spot_size * max_spot_size
    upper_width = max(upper_width, min_width)
    lower_width = max(lower_width, min_width)

    return spot_fast2_vertical(upper_sample, upper_width, upper_distance_y, max_spot_size) + \
           spot_fast2_vertical(lower_sample, lower_width, lower_distance_y, max_spot_size)


@ti.func
def spot_sim_fast2_vertical(vTexCoord: tm.vec2, img, widths, SourceSize: tm.vec4, OutputSize: tm.vec4,
                            max_spot_size: float, min_spot_size: float,
                            overscan_horizontal: float, overscan_vertical: float,
                            interlace_mode: int, odd_field_first: int, frame_count: int) -> tm.vec3:
    # Overscan
    vTexCoord = (1.0 - tm.vec2(overscan_horizontal, overscan_vertical)) * (vTexCoord - 0.5) + 0.5

    output = tm.vec3(0.0)
    if interlace_mode == 2 and SourceSize.y < 350.0:
        lines = get_lines_doubled(vTexCoord.y, SourceSize.y)
        output = evaluate_spot_fast2(vTexCoord, img, widths, SourceSize, lines, max_spot_size, min_spot_size)
    elif interlace_mode == 3 or SourceSize.y <= 300.0:
        lines = get_lines_closest(vTexCoord.y, SourceSize.y)
        output = evaluate_spot_fast2(vTexCoord, img, widths, SourceSize, lines, max_spot_size, min_spot_size)
    elif interlace_mode == 1:
        lines_odd = get_lines_odd(vTexCoord.y, SourceSize.y)
        lines_even = get_lines_even(vTexCoord.y, SourceSize.y)
        output = 0.5 * (evaluate_spot_fast2(vTexCoord, img, widths, SourceSize, lines_odd, max_spot_size, min_spot_size) +
                        evaluate_spot_fast2(vTexCoord, img, widths, SourceSize, lines_even, max_spot_size, min_spot_size))
    else:
        phase = 1 if odd_field_first == 0 else 0
        if (frame_count + phase) % 2 == 0:
            lines = get_lines_odd(vTexCoord.y, SourceSize.y)
            output = evaluate_spot_fast2(vTexCoord, img, widths, SourceSize, lines, max_spot_size, min_spot_size)
        else:
            lines = get_lines_even(vTexCoord.y, SourceSize.y)
            output = evaluate_spot_fast2(vTexCoord, img, widths, SourceSize, lines, max_spot_size, min_spot_size)
    return output


@ti.func
def spot_fast2_horizontal(sample: tm.vec3, width: float, x: float, x0_: float, x1_: float):
    x0 = tm.clamp((x - x0_) / width, -1.0, 1.0)
    x1 = tm.clamp((x - x1_) / width, -1.0, 1.0)
    return 1 / (2.0 * np.pi) * sample * (np.pi * x0 + tm.sin(np.pi * x0) - np.pi * x1 - tm.sin(np.pi * x1))


@ti.func
def spot_fast2_vertical(sample: tm.vec3, width: float, y_: float, max_spot_size: float):
    y = tm.clamp(abs(y_) / width, 0.0, 1.0)
    return sample * max_spot_size / width * (0.5 * tm.cos(np.pi * y) + 0.5)


def simulate_spot(image_in, output_dim, method: str,
                  max_spot_size: float, min_spot_size: float,
                  overscan_horizontal: float, overscan_vertical: float,
                  interlacing_type: str | int | float = 'bob',
                  odd_field_first: bool | int | float = True,
                  frame_count: int = 0):
    interlace_mode = parse_interlacing_mode(interlacing_type)
    odd_phase = 1 if odd_field_first else 0
    if method == 'sampled':
        return spot_fragment(image_in, output_dim, max_spot_size, min_spot_size, overscan_horizontal, overscan_vertical,
                             interlace_mode, odd_phase, frame_count)
    elif method == 'analytical':
        return spot_analytical_fragment(image_in, output_dim, max_spot_size, min_spot_size, overscan_horizontal, overscan_vertical,
                                        interlace_mode, odd_phase, frame_count)
    elif method in ('fast', 'fast2'):
        return spot_fast2_fragment(image_in, output_dim, max_spot_size, min_spot_size, overscan_horizontal, overscan_vertical,
                                   interlace_mode, odd_phase, frame_count)
    elif method == 'fast1':
        return spot_fast_fragment(image_in, output_dim, max_spot_size, min_spot_size, overscan_horizontal, overscan_vertical,
                                  interlace_mode, odd_phase, frame_count)
    else:
        raise ValueError(f"Unknown scanline simulation method: {method}. Supported methods: 'sampled', 'analytical', 'fast'.")

