import argparse
import math
from imageio.v3 import imread, imwrite
import numpy as np
import taichi as ti
import taichi.math as tm
import skimage
from skimage.metrics import structural_similarity as ssim


ti.init()


def srgb_to_linear(img):
    """sRGB uint8 to simple gamma float"""
    out = img.astype(np.float32) / 255
    out = np.where(out <= 0.04045, out / 12.92, np.power((out + 0.055) / 1.055, 2.4))
    return out


def linear_to_srgb(img):
    """Linear float to sRGB uint8"""
    out = np.where(img <= 0.0031308, img * 12.92, 1.055 * (np.power(np.clip(img, 0.0, 1.0), (1.0 / 2.4))) - 0.055)
    if (0 > out).any() or (out >= 1).any() or np.isnan(out).any() or not np.isfinite(out).any():
        print(out)
    out = np.around(out * 255).astype(np.uint8)
    return out


def psnr(img1, img2, max_value=255):
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


@ti.func
def texelFetch(Source, vTexCoords: tm.ivec2) -> tm.vec3:
    """Fetch pixel from img at (x, y), or 0 if outside range"""
    y_size, x_size = Source.shape
    val = tm.vec3(0.0)
    if not (vTexCoords.x < 0 or vTexCoords.x >= x_size or vTexCoords.y < 0 or vTexCoords.y >= y_size):
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


def box_blur(image_in, sigma):
    """Do several box blurs on the image, approximating a gaussian blur.

    This is a very fast blur for large images. The speed is not
    dependent on the sigma."""
    radius = int(np.round((np.sqrt(3 * sigma * sigma + 1) - 1) / 2))
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


def chained_gaussian_blur(image_in, sigma):
    (in_height, in_width, in_planes) = image_in.shape
    field_previous = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_previous.from_numpy(image_in)
    for i in range(7):
        previous_height, previous_width = field_previous.shape
        field_next = ti.Vector.field(n=3, dtype=float, shape=(previous_height // 2, previous_width // 2))
        cg_blur_fragment(field_previous, field_next, sigma)
        field_previous = field_next
        img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
        imwrite("{}.png".format(i), img_srgb)  # DEBUG
    field_next = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    cg_blur_fragment(field_previous, field_next, sigma)

    return field_next.to_numpy()


@ti.kernel
def cg_blur_fragment(field_in: ti.template(), field_out: ti.template(), sigma: float):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = cg_blur(vTexCoord, field_in, SourceSize, OutputSize, sigma)
    return


@ti.func
def cg_blur(vTexCoord: tm.vec2, Source, SourceSize: tm.vec4, OutputSize: tm.vec4, sigma: float) -> tm.vec3:
    pos = vTexCoord * SourceSize.xy
    weight_sum = 0.0
    value = tm.vec3(0.0)
    center = int(tm.round(pos))
    for x in range(center.x - 2, center.x + 2):
        for y in range(center.y - 2, center.y + 2):
            distance_x = pos.x - x - 0.5
            distance_y = pos.y - y - 0.5
            weight = tm.exp(-(distance_x * distance_x) / (2 * sigma * sigma)) * tm.exp(-(distance_y * distance_y) / (2 * sigma * sigma))
            weight_sum += weight
            value += weight * texelFetch(Source, tm.ivec2(x, y))
    #         if 50 < pos.x < 50.2 and 50 < pos.y < 50.2:
    #             print(pos, distance_x, distance_y, weight)
    # if 50 < pos.x < 50.2 and 50 < pos.y < 50.2:
    #         print(pos, vTexCoord, SourceSize, value, weight_sum)
    return value / weight_sum


@ti.kernel
def cg_blur_corrected_fragment(field_in: ti.template(), field_out: ti.template(), sigma: float):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = cg_blur_corrected(vTexCoord, field_in, SourceSize, OutputSize, sigma)
    return


@ti.func
def cg_blur_corrected(vTexCoord: tm.vec2, Source, SourceSize: tm.vec4, OutputSize: tm.vec4, sigma: float) -> tm.vec3:
    pos = vTexCoord * SourceSize.xy
    weight_sum = 0.0
    value = tm.vec3(0.0)
    center = int(tm.round(pos))
    for x in range(center.x - 2, center.x + 2):
        for y in range(center.y - 2, center.y + 2):
            distance_x = pos.x - x - 0.5
            distance_y = pos.y - y - 0.5
            weight = small_gaussian(distance_x, sigma) * small_gaussian(distance_y, sigma)
            weight_sum += weight
            value += weight * texelFetch(Source, tm.ivec2(x, y))
    return value / weight_sum


@ti.func
def small_gaussian(x: float, sigma: float):
    return erf((x + 0.5) / sigma * tm.sqrt(0.5)) - erf((x - 0.5) / sigma * tm.sqrt(0.5))


@ti.func
def erf(x: float):
    # Janky erf approximation with a cubic polynomial
    x = tm.clamp(x, -2.0, 2.0)
    val = abs(x) * (1.1283791670955126 + abs(x) * (-0.37837916709551256 + abs(x) * 0.03209479177387814))  # TODO
    return -val if x < 0 else val  # TODO sign faster?


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


def cubic_blur(image_in, sigma):
    (in_height, in_width, in_planes) = image_in.shape
    field_previous = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_previous.from_numpy(image_in)
    passes = int(math.log2(sigma)) + 1
    print('passes: {}'.format(passes))
    for i in range(passes):
        previous_height, previous_width = field_previous.shape
        field_next = ti.Vector.field(n=3, dtype=float, shape=(previous_height // 2, previous_width // 2))
        cubic_fragment(field_previous, field_next)
        field_previous = field_next
        img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
        imwrite("{}-cubic.png".format(i), img_srgb)  # DEBUG
    previous_height, previous_width = field_previous.shape
    field_next = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    small_sigma = (sigma / (in_width / previous_width) + sigma / (in_height / previous_height)) / 2
    print('small_sigma: {}'.format(small_sigma))
    cg_blur_corrected_fragment(field_previous, field_next, small_sigma)

    return field_next.to_numpy()


def cubic_blur2(image_in, sigma):
    (in_height, in_width, in_planes) = image_in.shape
    field_previous = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_previous.from_numpy(image_in)
    passes = int(math.log2(sigma))
    print('passes: {}'.format(passes))
    for i in range(passes):
        previous_height, previous_width = field_previous.shape
        field_next = ti.Vector.field(n=3, dtype=float, shape=(previous_height // 2, previous_width // 2))
        cubic_fragment(field_previous, field_next)
        field_previous = field_next
        img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
        imwrite("{}-cubic2.png".format(i), img_srgb)  # DEBUG
    previous_height, previous_width = field_previous.shape
    field_next = ti.Vector.field(n=3, dtype=float, shape=(in_width, previous_height))
    print('sigma_x = {}'.format(sigma / (in_width / previous_width)))
    gaussian_fragment(field_previous, field_next, sigma / (in_width / previous_width))
    img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
    imwrite("6pass1-cubic2.png".format(i), img_srgb)  # DEBUG
    field_previous = field_next
    field_next = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    print('sigma_y = {}'.format(sigma / (in_height / previous_height)))
    gaussian_fragment(field_previous, field_next, sigma / (in_height / previous_height))
    img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
    imwrite("6pass2-cubic2.png".format(i), img_srgb)  # DEBUG

    return field_next.to_numpy()


def cubic_blur3(image_in, sigma):
    (in_height, in_width, in_planes) = image_in.shape
    field_previous = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_previous.from_numpy(image_in)
    for i in range(4):
        previous_height, previous_width = field_previous.shape
        field_next = ti.Vector.field(n=3, dtype=float, shape=(previous_height // 2, previous_width // 2))
        lanczos1_fragment(field_previous, field_next)
        field_previous = field_next
        img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
        imwrite("{}-cubic3.png".format(i), img_srgb)  # DEBUG

    previous_height, previous_width = field_previous.shape
    field_next = ti.Vector.field(n=3, dtype=float, shape=(previous_width, previous_height))
    print('sigma_x = {}'.format(sigma / (in_width / previous_width)))
    gaussian_fragment(field_previous, field_next, sigma / (in_width / previous_width))
    img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
    imwrite("pass1-cubic3.png".format(i), img_srgb)  # DEBUG

    field_previous = field_next
    field_next = ti.Vector.field(n=3, dtype=float, shape=(previous_height, previous_width))
    print('sigma_y = {}'.format(sigma / (in_height / previous_height)))
    gaussian_fragment(field_previous, field_next, sigma / (in_height / previous_height))
    img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
    imwrite("pass2-cubic3.png".format(i), img_srgb)  # DEBUG

    # field_previous = field_next
    # field_next = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    # cubic_up_fragment(field_previous, field_next)

    # field_previous = field_next
    # field_next = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    # lanczos1_up_fragment(field_previous, field_next)

    field_previous = field_next
    for i in range(3):
        previous_height, previous_width = field_previous.shape
        field_next = ti.Vector.field(n=3, dtype=float, shape=(previous_height * 2, previous_width * 2))
        lanczos1_up_fragment(field_previous, field_next)
        field_previous = field_next
        img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
        imwrite("{}-up-cubic3.png".format(i), img_srgb)  # DEBUG
    field_next = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    lanczos1_up_fragment(field_previous, field_next)

    return field_next.to_numpy()


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
            weight = (distance_x * distance_x * ((2.0 / 8.0) * distance_x - (3.0 / 4.0)) + 1.0) * \
                     (distance_y * distance_y * ((2.0 / 8.0) * distance_y - (3.0 / 4.0)) + 1.0)
            weight_sum += weight
            value += weight * texelFetch(Source, tm.ivec2(x, y))
    #         if 50 < pos.x < 50.2 and 50 < pos.y < 50.2:
    #             print(pos, distance_x, distance_y, weight)
    # if 50 < pos.x < 50.2 and 50 < pos.y < 50.2:
    #         print(pos, vTexCoord, SourceSize, value, weight_sum)
    return value / weight_sum


@ti.kernel
def cubic_up_fragment(field_in: ti.template(), field_out: ti.template()):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = cubic_upscale(vTexCoord, field_in, SourceSize, OutputSize)
    return


@ti.func
def cubic_upscale(vTexCoord: tm.vec2, Source, SourceSize: tm.vec4, OutputSize: tm.vec4) -> tm.vec3:
    pos = vTexCoord * SourceSize.xy
    weight_sum = 0.0
    value = tm.vec3(0.0)
    center = int(tm.round(pos))  # TODO
    for x in range(center.x - 1, center.x + 1):
        for y in range(center.y - 1, center.y + 1):
            distance_x = abs(pos.x - x - 0.5)
            distance_y = abs(pos.y - y - 0.5)
            if distance_x > 1.0 or distance_y > 1.0:
                print(distance_x, distance_y)
            weight = (distance_x * distance_x * (2.0 * distance_x - 3.0) + 1.0) * \
                     (distance_y * distance_y * (2.0 * distance_y - 3.0) + 1.0)
            weight_sum += weight
            value += weight * texelFetch(Source, tm.ivec2(x, y))
            if 50.4 < pos.x < 50.6 and 50.4 < pos.y < 50.6:
                print(pos, x, y, distance_x, distance_y, weight)
    if 50.4 < pos.x < 50.6 and 50.4 < pos.y < 50.6:
            print(pos, vTexCoord, SourceSize, value, weight_sum)
    return value / weight_sum


def bilinear_blur(image_in, sigma):
    (in_height, in_width, in_planes) = image_in.shape
    field_previous = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_previous.from_numpy(image_in)
    passes = int(math.log2(sigma)) + 1
    for i in range(passes):
        previous_height, previous_width = field_previous.shape
        field_next = ti.Vector.field(n=3, dtype=float, shape=(previous_height // 2, previous_width // 2))
        bilinear_fragment(field_previous, field_next)
        field_previous = field_next
        img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
        imwrite("{}-bilinear.png".format(i), img_srgb)  # DEBUG
    previous_height, previous_width = field_previous.shape
    field_next = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    small_sigma = (sigma / (in_width / previous_width) + sigma / (in_height / previous_height)) / 2
    print('small_sigma: {}'.format(small_sigma))
    cg_blur_corrected_fragment(field_previous, field_next, small_sigma)

    return field_next.to_numpy()


def bilinear_blur_uncorrected(image_in, sigma):
    (in_height, in_width, in_planes) = image_in.shape
    field_previous = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_previous.from_numpy(image_in)
    passes = int(math.log2(sigma)) + 1
    for i in range(passes):
        previous_height, previous_width = field_previous.shape
        field_next = ti.Vector.field(n=3, dtype=float, shape=(previous_height // 2, previous_width // 2))
        bilinear_fragment(field_previous, field_next)
        field_previous = field_next
        img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
        imwrite("{}-bilinear-uncorrected.png".format(i), img_srgb)  # DEBUG
    previous_height, previous_width = field_previous.shape
    field_next = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    small_sigma = (sigma / (in_width / previous_width) + sigma / (in_height / previous_height)) / 2
    print('small_sigma: {}'.format(small_sigma))
    cg_blur_fragment(field_previous, field_next, small_sigma)

    return field_next.to_numpy()


@ti.kernel
def bilinear_fragment(field_in: ti.template(), field_out: ti.template()):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = bilinear_downscale(vTexCoord, field_in, SourceSize, OutputSize)
    return


@ti.func
def bilinear_downscale(vTexCoord: tm.vec2, Source, SourceSize: tm.vec4, OutputSize: tm.vec4) -> tm.vec3:
    pos = vTexCoord * SourceSize.xy
    return texture(Source, vTexCoord)



def lanczos1_blur(image_in, sigma):
    (in_height, in_width, in_planes) = image_in.shape
    field_previous = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_previous.from_numpy(image_in)
    for i in range(7):
        previous_height, previous_width = field_previous.shape
        field_next = ti.Vector.field(n=3, dtype=float, shape=(previous_height // 2, previous_width // 2))
        lanczos1_fragment(field_previous, field_next)
        field_previous = field_next
    field_next = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    cg_blur_fragment(field_previous, field_next, sigma)

    return field_next.to_numpy()


@ti.kernel
def lanczos1_fragment(field_in: ti.template(), field_out: ti.template()):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = lanczos1_downscale2(vTexCoord, field_in, SourceSize, OutputSize)
    return


@ti.func
def lanczos1_downscale(vTexCoord: tm.vec2, Source, SourceSize: tm.vec4, OutputSize: tm.vec4) -> tm.vec3:
    pos = vTexCoord * SourceSize.xy
    weight_sum = 0.0
    value = tm.vec3(0.0)
    center = int(tm.round(pos))  # TODO
    for x in range(center.x - 2, center.x + 2):
        for y in range(center.y - 2, center.y + 2):
            distance_x = abs(pos.x - x - 0.5)
            distance_y = abs(pos.y - y - 0.5)
            weight_x = 1 if distance_x == 0.0 else tm.sin(np.pi / 2 * distance_x) * tm.sin(np.pi / 2 * distance_x) / (np.pi * np.pi * distance_x * distance_x / 4)
            weight_y = 1 if distance_y == 0.0 else tm.sin(np.pi / 2 * distance_y) * tm.sin(np.pi / 2 * distance_y) / (np.pi * np.pi * distance_y * distance_y / 4)
            weight = weight_x * weight_y
            weight_sum += weight
            value += weight * texelFetch(Source, tm.ivec2(x, y))
    return value / weight_sum


@ti.func
def lanczos1_downscale2(vTexCoord: tm.vec2, Source, SourceSize: tm.vec4, OutputSize: tm.vec4) -> tm.vec3:
    pos = vTexCoord * SourceSize.xy
    center = tm.floor(pos - 0.5) + 0.5

    # Polynomial approximation based on 1.0-3.20832*x^{2}+3.88631*x^{4}-2.14792*x^{6}+0.469928*x^{8}
    x2 = pos - center + 1.0
    x2 = x2 * x2
    weight0 = tm.clamp((((0.00183565625*x2 - 0.03356125)*x2 + 0.242894375)*x2 - 0.80208)*x2 + 1.0, 0.0, 1.0)
    x2 = pos - center
    x2 = x2 * x2
    weight1 = tm.clamp((((0.00183565625*x2 - 0.03356125)*x2 + 0.242894375)*x2 - 0.80208)*x2 + 1.0, 0.0, 1.0)
    x2 = pos - center - 1.0
    x2 = x2 * x2
    weight2 = tm.clamp((((0.00183565625*x2 - 0.03356125)*x2 + 0.242894375)*x2 - 0.80208)*x2 + 1.0, 0.0, 1.0)
    x2 = pos - center - 2.0
    x2 = x2 * x2
    weight3 = tm.clamp((((0.00183565625*x2 - 0.03356125)*x2 + 0.242894375)*x2 - 0.80208)*x2 + 1.0, 0.0, 1.0)

    weight_sum = weight0 + weight1 + weight2 + weight3
    weight0 /= weight_sum
    weight1 /= weight_sum
    weight2 /= weight_sum
    weight3 /= weight_sum

    scale0 = weight0 + weight1
    scale1 = weight2 + weight3

    coord0 = (center - 1.0 + weight1 / scale0) * SourceSize.zw
    coord1 = (center + 1.0 + weight3 / scale1) * SourceSize.zw

    return (texture(Source, tm.vec2(coord0.x, coord0.y)) * scale0.x +
            texture(Source, tm.vec2(coord1.x, coord0.y)) * scale1.x) * scale0.y + \
           (texture(Source, tm.vec2(coord0.x, coord1.y)) * scale0.x +
            texture(Source, tm.vec2(coord1.x, coord1.y)) * scale1.x) * scale1.y


@ti.kernel
def lanczos1_up_fragment(field_in: ti.template(), field_out: ti.template()):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = lanczos1_upscale(vTexCoord, field_in, SourceSize, OutputSize)
    return


@ti.func
def lanczos1_upscale(vTexCoord: tm.vec2, Source, SourceSize: tm.vec4, OutputSize: tm.vec4) -> tm.vec3:
    pos = vTexCoord * SourceSize.xy
    weight_sum = 0.0
    value = tm.vec3(0.0)
    center = int(tm.round(pos))  # TODO
    for x in range(center.x - 1, center.x + 1):
        for y in range(center.y - 1, center.y + 1):
            distance_x = abs(pos.x - x - 0.5)
            distance_y = abs(pos.y - y - 0.5)
            weight_x = 1 if distance_x == 0.0 else tm.sin(np.pi * distance_x) * tm.sin(np.pi * distance_x) / (np.pi * np.pi * distance_x * distance_x)
            weight_y = 1 if distance_y == 0.0 else tm.sin(np.pi * distance_y) * tm.sin(np.pi * distance_y) / (np.pi * np.pi * distance_y * distance_y)
            weight = weight_x * weight_y
            weight_sum += weight
            value += weight * texelFetch(Source, tm.ivec2(x, y))
    return value / weight_sum


@ti.func
def lanczos1_upscale2(vTexCoord: tm.vec2, Source, SourceSize: tm.vec4, OutputSize: tm.vec4) -> tm.vec3:
    pos = vTexCoord * SourceSize.xy
    weight_sum = 0.0
    value = tm.vec3(0.0)
    center = int(tm.round(pos))  # TODO
    for x in range(center.x - 1, center.x + 1):
        for y in range(center.y - 1, center.y + 1):
            distance_x = abs(pos.x - x - 0.5)
            distance_y = abs(pos.y - y - 0.5)
            x2 = distance_x * distance_x
            weight_x = tm.clamp((((0.469928 * x2 - 2.14792) * x2 + 3.88631) * x2 - 3.20832) * x2 + 1.0, 0.0, 1.0)
            y2 = distance_y * distance_y
            weight_y = tm.clamp((((0.469928 * y2 - 2.14792) * y2 + 3.88631) * y2 - 3.20832) * y2 + 1.0, 0.0, 1.0)
            weight = weight_x * weight_y
            weight_sum += weight
            value += weight * texelFetch(Source, tm.ivec2(x, y))
    return value / weight_sum


def cubic_gaussian_bloom(image_in, passes, sigma):
    (in_height, in_width, in_planes) = image_in.shape
    field_previous = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_previous.from_numpy(image_in)
    print('passes: {}'.format(passes))
    down_passes = []
    for i in range(passes):
        field_next = ti.Vector.field(n=3, dtype=float, shape=(in_height // 2**i, in_width // 2**i))
        cubic_fragment(field_previous, field_next)
        field_previous = field_next
        down_passes.append(field_next)
        img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
        imwrite("{}-cubic-gaussian.png".format(i), img_srgb)  # DEBUG
    for i in range(passes):
        field_next = ti.Vector.field(n=3, dtype=float, shape=(in_height // 2**(passes - i - 1), in_width // 2**(passes - i - 1)))
        cg_blur_fragment(field_previous, field_next, sigma)
        field_previous = field_next
        img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
        imwrite("{}-cubic-gaussian.png".format(i), img_srgb)  # DEBUG

    return field_next.to_numpy()


def cubic_bilinear_blur(image_in, sigma):
    (in_height, in_width, in_planes) = image_in.shape
    field_previous = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_previous.from_numpy(image_in)
    for i in range(3):
        previous_height, previous_width = field_previous.shape
        field_next = ti.Vector.field(n=3, dtype=float, shape=(previous_height // 2, previous_width // 2))
        cubic_fragment(field_previous, field_next)
        field_previous = field_next
        img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
        imwrite("{}-cubic-bilinear.png".format(i), img_srgb)  # DEBUG

    previous_height, previous_width = field_previous.shape
    field_next = ti.Vector.field(n=3, dtype=float, shape=(previous_width, previous_height))
    print('sigma_x = {}'.format(sigma / (in_width / previous_width)))
    gaussian_fragment(field_previous, field_next, sigma / (in_width / previous_width))
    img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
    imwrite("horiz-cubic-bilinear.png".format(i), img_srgb)  # DEBUG

    field_previous = field_next
    field_next = ti.Vector.field(n=3, dtype=float, shape=(previous_height, previous_width))
    print('sigma_y = {}'.format(sigma / (in_height / previous_height)))
    gaussian_fragment(field_previous, field_next, sigma / (in_height / previous_height))
    img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
    imwrite("vert-cubic-bilinear.png".format(i), img_srgb)  # DEBUG

    field_previous = field_next
    for i in range(2):
        previous_height, previous_width = field_previous.shape
        field_next = ti.Vector.field(n=3, dtype=float, shape=(previous_height * 2, previous_width * 2))
        bilinear_fragment(field_previous, field_next)
        field_previous = field_next
        img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
        imwrite("{}-up-cubic-bilinear.png".format(i), img_srgb)  # DEBUG
    field_next = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    bilinear_fragment(field_previous, field_next)

    return field_next.to_numpy()


def lanczos_bilinear_blur(image_in, sigma):
    (in_height, in_width, in_planes) = image_in.shape
    field_previous = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_previous.from_numpy(image_in)
    for i in range(3):
        previous_height, previous_width = field_previous.shape
        field_next = ti.Vector.field(n=3, dtype=float, shape=(previous_height // 2, previous_width // 2))
        lanczos1_fragment(field_previous, field_next)
        field_previous = field_next
        img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
        imwrite("{}-lanzcos-bilinear.png".format(i), img_srgb)  # DEBUG

    previous_height, previous_width = field_previous.shape
    field_next = ti.Vector.field(n=3, dtype=float, shape=(previous_width, previous_height))
    print('sigma_x = {}'.format(sigma / (in_width / previous_width)))
    gaussian_fragment(field_previous, field_next, sigma / (in_width / previous_width))
    img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
    imwrite("horiz-lanczos-bilinear.png".format(i), img_srgb)  # DEBUG

    field_previous = field_next
    field_next = ti.Vector.field(n=3, dtype=float, shape=(previous_height, previous_width))
    print('sigma_y = {}'.format(sigma / (in_height / previous_height)))
    gaussian_fragment(field_previous, field_next, sigma / (in_height / previous_height))
    img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
    imwrite("vert-lanczos-bilinear.png".format(i), img_srgb)  # DEBUG

    field_previous = field_next
    for i in range(2):
        previous_height, previous_width = field_previous.shape
        field_next = ti.Vector.field(n=3, dtype=float, shape=(previous_height * 2, previous_width * 2))
        bilinear_fragment(field_previous, field_next)
        field_previous = field_next
        img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
        imwrite("{}-up-lanczos-bilinear.png".format(i), img_srgb)  # DEBUG
    field_next = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    bilinear_fragment(field_previous, field_next)

    return field_next.to_numpy()


def cubic_gaussian_blur(image_in, sigma):
    (in_height, in_width, in_planes) = image_in.shape
    field_previous = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_previous.from_numpy(image_in)
    for i in range(3):
        previous_height, previous_width = field_previous.shape
        field_next = ti.Vector.field(n=3, dtype=float, shape=(previous_height // 2, previous_width // 2))
        cubic_fragment(field_previous, field_next)
        field_previous = field_next
        img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
        imwrite("{}-cubic-gaussian.png".format(i), img_srgb)  # DEBUG

    previous_height, previous_width = field_previous.shape
    field_next = ti.Vector.field(n=3, dtype=float, shape=(previous_width, previous_height))
    print('sigma_x = {}'.format(sigma / (in_width / previous_width)))
    gaussian_fragment(field_previous, field_next, sigma / (in_width / previous_width))
    img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
    imwrite("horiz-cubic-gaussian.png".format(i), img_srgb)  # DEBUG

    field_previous = field_next
    field_next = ti.Vector.field(n=3, dtype=float, shape=(previous_height, previous_width))
    print('sigma_y = {}'.format(sigma / (in_height / previous_height)))
    gaussian_fragment(field_previous, field_next, sigma / (in_height / previous_height))
    img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
    imwrite("vert-cubic-gaussian.png".format(i), img_srgb)  # DEBUG

    field_previous = field_next
    field_next = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    cg_blur_fragment(field_previous, field_next, 0.5)

    return field_next.to_numpy()


def cubic_gaussianc_blur(image_in, sigma):
    (in_height, in_width, in_planes) = image_in.shape
    field_previous = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_previous.from_numpy(image_in)
    for i in range(3):
        previous_height, previous_width = field_previous.shape
        field_next = ti.Vector.field(n=3, dtype=float, shape=(previous_height // 2, previous_width // 2))
        cubic_fragment(field_previous, field_next)
        field_previous = field_next
        img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
        imwrite("{}-cubic-gaussianc.png".format(i), img_srgb)  # DEBUG

    previous_height, previous_width = field_previous.shape
    field_next = ti.Vector.field(n=3, dtype=float, shape=(previous_width, previous_height))
    print('sigma_x = {}'.format(sigma / (in_width / previous_width)))
    gaussian_fragment(field_previous, field_next, sigma / (in_width / previous_width))
    img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
    imwrite("horiz-cubic-gaussianc.png".format(i), img_srgb)  # DEBUG

    field_previous = field_next
    field_next = ti.Vector.field(n=3, dtype=float, shape=(previous_height, previous_width))
    print('sigma_y = {}'.format(sigma / (in_height / previous_height)))
    gaussian_fragment(field_previous, field_next, sigma / (in_height / previous_height))
    img_srgb = linear_to_srgb(field_next.to_numpy())  # DEBUG
    imwrite("vert-cubic-gaussianc.png".format(i), img_srgb)  # DEBUG

    field_previous = field_next
    field_next = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    cg_blur_corrected_fragment(field_previous, field_next, 0.5)

    return field_next.to_numpy()


def main():
    parser = argparse.ArgumentParser(description='Generate a CRT-simulated image')
    parser.add_argument('input')
    #parser.add_argument('output')
    args = parser.parse_args()

    # Read image
    img_original = imread(args.input)
    image_height, image_width, planes = img_original.shape

    img_linear = srgb_to_linear(img_original)

    sigma = 16

    print('Standard gaussian blur...')
    img_standard = gaussian_blur(img_linear, sigma)
    img_standard = np.clip(img_standard, 0, 1)
    img_standard = linear_to_srgb(img_standard)
    imwrite("standard-out.png", img_standard)

    # skimage implementation as a sanity check
    print('Standard gaussian blur (skimage)...')
    img_sk = skimage.filters.gaussian(img_linear, sigma=sigma, mode='constant', preserve_range=True, channel_axis=-1)
    img_sk = np.clip(img_sk, 0, 1)
    img_sk = linear_to_srgb(img_sk)
    print("PSNR: {}".format(psnr(img_standard, img_sk)))
    print('SSIM: {}'.format(ssim(img_standard, img_sk, channel_axis=-1)))
    imwrite("skimage-out.png", img_sk)

    # print('4-pass box blur...')
    # img_box = box_blur(img_linear, sigma)
    # img_box = np.clip(img_box, 0, 1)
    # img_box = linear_to_srgb(img_box)
    # print("PSNR: {}".format(psnr(img_standard, img_box)))
    # print('SSIM: {}'.format(ssim(img_standard, img_box, channel_axis=-1)))
    # imwrite("box-out.png", img_box)

    # print('Chained gaussian blur...')
    # img_cg = chained_gaussian_blur(img_linear, 1.0)
    # img_cg = np.clip(img_cg, 0, 1)
    # img_cg = linear_to_srgb(img_cg)
    # print("PSNR: {}".format(psnr(img_standard, img_cg)))
    # print('SSIM: {}'.format(ssim(img_standard, img_cg, channel_axis=-1)))
    # imwrite("chained-gaussian-out.png", img_cg)

    # print('Cubic down + gaussian up blur...')
    # img_cubic = cubic_blur(img_linear, sigma)
    # img_cubic = np.clip(img_cubic, 0, 1)
    # img_cubic = linear_to_srgb(img_cubic)
    # print("PSNR: {}".format(psnr(img_standard, img_cubic)))
    # print('SSIM: {}'.format(ssim(img_standard, img_cubic, channel_axis=-1)))
    # imwrite("cubic-out.png", img_cubic)

    # print('Cubic down 2 + gaussian up blur...')
    # img_cubic2 = cubic_blur2(img_linear, sigma)
    # img_cubic2 = np.clip(img_cubic2, 0, 1)
    # img_cubic2 = linear_to_srgb(img_cubic2)
    # print("PSNR: {}".format(psnr(img_standard, img_cubic2)))
    # print('SSIM: {}'.format(ssim(img_standard, img_cubic2, channel_axis=-1)))
    # imwrite("cubic2-out.png", img_cubic2)

    # print('Cubic down 3 + gaussian up blur...')
    # img_cubic3 = cubic_blur3(img_linear, sigma)
    # img_cubic3 = np.clip(img_cubic3, 0, 1)
    # img_cubic3 = linear_to_srgb(img_cubic3)
    # print("PSNR: {}".format(psnr(img_standard, img_cubic3)))
    # print('SSIM: {}'.format(ssim(img_standard, img_cubic3, channel_axis=-1)))
    # imwrite("cubic3-out.png", img_cubic3)

    # print('Bilinear down + gaussian up blur...')
    # img_bilinear = bilinear_blur(img_linear, sigma)
    # img_bilinear = np.clip(img_bilinear, 0, 1)
    # img_bilinear = linear_to_srgb(img_bilinear)
    # print("PSNR: {}".format(psnr(img_standard, img_bilinear)))
    # print('SSIM: {}'.format(ssim(img_standard, img_bilinear, channel_axis=-1)))
    # imwrite("bilinear-out.png", img_bilinear)

    # print('Bilinear down + gaussian uncorrected up blur...')
    # img_bilinear = bilinear_blur_uncorrected(img_linear, sigma)
    # img_bilinear = np.clip(img_bilinear, 0, 1)
    # img_bilinear = linear_to_srgb(img_bilinear)
    # print("PSNR: {}".format(psnr(img_standard, img_bilinear)))
    # print('SSIM: {}'.format(ssim(img_standard, img_bilinear, channel_axis=-1)))
    # imwrite("bilinear-uncorrected-out.png", img_bilinear)

    # print('Lanczos1 down + gaussian up blur...')
    # img_lanczos1 = lanczos1_blur(img_linear, 1.0)
    # img_lanczos1 = np.clip(img_lanczos1, 0, 1)
    # img_lanczos1 = linear_to_srgb(img_lanczos1)
    # print("PSNR: {}".format(psnr(img_standard, img_lanczos1)))
    # print('SSIM: {}'.format(ssim(img_standard, img_lanczos1, channel_axis=-1)))
    # imwrite("lanczos1-out.png", img_lanczos1)

    print('Cubic down + bilinear up blur...')
    img_cubic_bilinear = cubic_bilinear_blur(img_linear, sigma)
    img_cubic_bilinear = np.clip(img_cubic_bilinear, 0, 1)
    img_cubic_bilinear = linear_to_srgb(img_cubic_bilinear)
    print("PSNR: {}".format(psnr(img_standard, img_cubic_bilinear)))
    print('SSIM: {}'.format(ssim(img_standard, img_cubic_bilinear, channel_axis=-1)))
    imwrite("cubic-bilinear-out.png", img_cubic_bilinear)

    print('Cubic down + gaussian up blur...')
    img_cubic_gaussian = cubic_gaussian_blur(img_linear, sigma)
    img_cubic_gaussian = np.clip(img_cubic_gaussian, 0, 1)
    img_cubic_gaussian = linear_to_srgb(img_cubic_gaussian)
    print("PSNR: {}".format(psnr(img_standard, img_cubic_gaussian)))
    print('SSIM: {}'.format(ssim(img_standard, img_cubic_gaussian, channel_axis=-1)))
    imwrite("cubic-gaussian-out.png", img_cubic_gaussian)

    print('Cubic down + corrected gaussian up blur...')
    img_cubic_gaussianc = cubic_gaussianc_blur(img_linear, sigma)
    img_cubic_gaussianc = np.clip(img_cubic_gaussianc, 0, 1)
    img_cubic_gaussianc = linear_to_srgb(img_cubic_gaussianc)
    print("PSNR: {}".format(psnr(img_standard, img_cubic_gaussianc)))
    print('SSIM: {}'.format(ssim(img_standard, img_cubic_gaussianc, channel_axis=-1)))
    imwrite("cubic-gaussianc-out.png", img_cubic_gaussianc)

    # print('Lanczos1 down + bilinear up blur...')
    # img_lanczos_bilinear = lanczos_bilinear_blur(img_linear, sigma)
    # img_lanczos_bilinear = np.clip(img_lanczos_bilinear, 0, 1)
    # img_lanczos_bilinear = linear_to_srgb(img_lanczos_bilinear)
    # print("PSNR: {}".format(psnr(img_standard, img_lanczos_bilinear)))
    # print('SSIM: {}'.format(ssim(img_standard, img_lanczos_bilinear, channel_axis=-1)))
    # imwrite("lanczos-bilinear-out.png", img_lanczos_bilinear)


if __name__ == '__main__':
    main()
