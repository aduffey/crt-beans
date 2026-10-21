import numpy as np
import taichi as ti
import taichi.math as tm
from common import texelFetch


def box_blur(image_in, radius: int):
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


def gaussian_blur(image_in, sigma: float):
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


def apply_diffusion(img, diffusion_type: str, blur_sigma: float, blur_amount: float):
    if diffusion_type in (None, 'none', 'disabled', False) or blur_amount <= 0.0:
        return img

    sigma = blur_sigma * img.shape[0]
    if diffusion_type == 'gaussian':
        blurred = gaussian_blur(img, sigma)
    elif diffusion_type == 'box':
        box_radius = int(np.round((np.sqrt(3 * sigma * sigma + 1) - 1) / 2))
        blurred = box_blur(img, box_radius)
    else:
        raise ValueError(f"Unknown diffusion type: {diffusion_type}. Supported types: 'gaussian', 'box', 'none'.")

    return img + (blurred - img) * blur_amount
