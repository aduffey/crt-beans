import argparse
import math

import numpy as np
import skimage
import taichi as ti
import taichi.math as tm
from imageio.v3 import imread, imwrite
from skimage.metrics import structural_similarity as ssim

ti.init()


COEFF = 0.35


def srgb_to_linear(img):
    """sRGB uint8 to simple gamma float"""
    out = img.astype(np.float32) / 255
    out = np.where(out <= 0.04045, out / 12.92, np.power((out + 0.055) / 1.055, 2.4))
    return out


def linear_to_srgb(img):
    """Linear float to sRGB uint8"""
    out = np.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * (np.power(np.clip(img, 0.0, 1.0), (1.0 / 2.4))) - 0.055,
    )
    if (
        (0 > out).any()
        or (out >= 1).any()
        or np.isnan(out).any()
        or not np.isfinite(out).any()
    ):
        print(out)
    out = np.around(out * 255).astype(np.uint8)
    return out


@ti.func
def texelFetch(Source, vTexCoords: tm.ivec2) -> tm.vec3:
    """Fetch pixel from img at (x, y), or 0 if outside range"""
    y_size, x_size = Source.shape
    val = tm.vec3(0.0)
    if not (
        vTexCoords.x < 0
        or vTexCoords.x >= x_size
        or vTexCoords.y < 0
        or vTexCoords.y >= y_size
    ):
        val = Source[vTexCoords.y, vTexCoords.x]
    return val


@ti.func
def texture(Source, vTexCoords: tm.vec2) -> tm.vec3:
    """Sample from Source at (x, y) using bilinear interpolation.

    Outside of the Source is considered to be 0.
    """
    y_size, x_size = Source.shape
    lookup_coords = vTexCoords.xy * tm.vec2(x_size, y_size)
    coords = tm.round(lookup_coords, int)
    v11 = texelFetch(Source, coords)
    v01 = texelFetch(Source, coords - tm.ivec2(1, 0))
    v10 = texelFetch(Source, coords - tm.ivec2(0, 1))
    v00 = texelFetch(Source, coords - tm.ivec2(1, 1))
    col1 = tm.mix(v10, v11, lookup_coords.y - coords.y + 0.5)
    col0 = tm.mix(v00, v01, lookup_coords.y - coords.y + 0.5)
    return tm.mix(col0, col1, lookup_coords.x - coords.x + 0.5)


def blur(image_in, coeff):
    (in_height, in_width, in_planes) = image_in.shape
    iterations = math.floor(math.log2(max(in_height, in_width)))

    field_in = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_in.from_numpy(image_in)
    fields = [field_in]

    for i in range(iterations):
        previous_height, previous_width = fields[i].shape
        fields.append(
            ti.Vector.field(
                n=3, dtype=float, shape=(previous_height // 2, previous_width // 2)
            )
        )
        cubic_fragment(fields[i], fields[i + 1])
    print(f'{len(fields)} fields, {iterations} iterations')

    for i in range(iterations):
        upsample_and_merge(fields[iterations - i], fields[iterations - i - 1], coeff)

    return fields[0].to_numpy()


@ti.kernel
def cubic_fragment(field_in: ti.template(), field_out: ti.template()):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = cubic_downscale(vTexCoord, field_in, SourceSize, OutputSize)
    return


@ti.func
def cubic_downscale(vTexCoord: tm.vec2, Source, SourceSize: tm.vec4, OutputSize: tm.vec4) -> tm.vec3:
    pos = vTexCoord * SourceSize.xy
    weight_sum = 0.0
    value = tm.vec3(0.0)
    center = int(tm.round(pos))  # TODO
    for x in range(center.x - 2, center.x + 2):
        for y in range(center.y - 2, center.y + 2):
            distance_x = abs(pos.x - x - 0.5)
            distance_y = abs(pos.y - y - 0.5)
            weight = (
                distance_x * distance_x * ((2.0 / 8.0) * distance_x - (3.0 / 4.0)) + 1.0
            ) * (
                distance_y * distance_y * ((2.0 / 8.0) * distance_y - (3.0 / 4.0)) + 1.0
            )
            weight_sum += weight
            value += weight * texelFetch(Source, tm.ivec2(x, y))
    #         if 50 < pos.x < 50.2 and 50 < pos.y < 50.2:
    #             print(pos, distance_x, distance_y, weight)
    # if 50 < pos.x < 50.2 and 50 < pos.y < 50.2:
    #         print(pos, vTexCoord, SourceSize, value, weight_sum)
    return value / weight_sum


@ti.kernel
def upsample_and_merge(field_in: ti.template(), field_out: ti.template(), coeff: float):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = tm.mix(
            bilinear(vTexCoord, field_in, SourceSize, OutputSize),
            field_out[y, x],
            coeff,
        )
    return


@ti.func
def bilinear(vTexCoord: tm.vec2, Source, SourceSize: tm.vec4, OutputSize: tm.vec4) -> tm.vec3:
    pos = vTexCoord * SourceSize.xy
    return texture(Source, vTexCoord)


def main():
    parser = argparse.ArgumentParser(description='Generate a CRT-simulated image')
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    # Read image
    img_original = imread(args.input)
    image_height, image_width, planes = img_original.shape

    img_linear = srgb_to_linear(img_original)
    img_standard = blur(img_linear, COEFF)
    img_standard = np.clip(img_standard, 0.0, 1.0)
    img_standard = linear_to_srgb(img_standard)
    imwrite(args.output, img_standard)


if __name__ == '__main__':
    main()
