import numpy as np
import taichi as ti
import taichi.math as tm


def srgb_to_gamma(img, out_gamma):
    """sRGB uint8 to simple gamma float"""
    out = img.astype(np.float32) / 255
    out = np.where(out <= 0.04045, out / 12.92, np.power((out + 0.055) / 1.055, 2.4))
    out = np.power(out, 1.0 / out_gamma)
    return out


def gamma_to_gamma(img, in_gamma, out_gamma):
    return np.power(np.power(np.clip(img, 0.0, 1.0), in_gamma), 1.0 / out_gamma)


# Matrices matching slang-shaders/shaders/colorspace/conversions.inc and experiments/yuv_matrix.py
RGB_TO_YUV = np.array([
    [0.29900000, 0.58700000, 0.11400000],
    [-0.14714119, -0.28886916, 0.43601035],
    [0.61497538, -0.51496512, -0.10001026]
], dtype=np.float32)

YUV_TO_RGB = np.array([
    [1.00000000, 0.00000000, 1.13988303],
    [1.00000000, -0.39464234, -0.58062185],
    [1.00000000, 2.03206187, 0.00000000]
], dtype=np.float32)


def gamma_to_yuv(img):
    """Native gamma to YUV float"""
    return np.dot(img, RGB_TO_YUV.T)


def yuv_to_linear(img, in_gamma):
    """YUV float to linear float"""
    rgb = np.dot(img, YUV_TO_RGB.T)
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.power(rgb, in_gamma)


def gamma_to_linear(img, in_gamma):
    """Simple gamma float to linear float"""
    return np.power(np.clip(img, 0.0, 1.0), in_gamma)


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
    v01 = texelFetch(Source, coords - tm.ivec2(1, 0))
    v10 = texelFetch(Source, coords - tm.ivec2(0, 1))
    v00 = texelFetch(Source, coords - tm.ivec2(1, 1))
    row1 = tm.mix(v10, v11, lookup_coords.y - coords.y + 0.5)
    row0 = tm.mix(v01, v00, lookup_coords.y - coords.y + 0.5)
    return tm.mix(row0, row1, lookup_coords.x - coords.x + 0.5)


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
    a = v * (v + 0.0245786) - 0.000090537
    b = v * (0.983729 * v + 0.4329510) + 0.238081
    return np.tensordot(a / b, acesOutput, axes=1)
