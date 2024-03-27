from imageio.v3 import imwrite, imread
import numpy as np
import skimage
import argparse
import multiprocessing as mp
import functools
from PIL import Image, ImageFilter
import taichi as ti
import taichi.math as tm


ti.init()


def srgb_to_gamma(img, out_gamma):
    """sRGB uint8 to simple gamma float"""
    out = img.astype(np.float32) / 255
    out = np.where(out <= 0.04045, out / 12.92, np.power((out + 0.055) / 1.055, 2.4))
    out = np.power(out, 1.0 / out_gamma)
    return out


def srgb_to_yiq(img, out_gamma):
    """sRGB uint8 to YIQ float"""
    out = img.astype(np.float32) / 255
    out = np.where(out <= 0.04045, out / 12.92, np.power((out + 0.055) / 1.055, 2.4))
    out = np.power(out, 1.0 / out_gamma)
    rgb2yiq = np.array([[0.30, 0.59, 0.11],
                        [0.599, -0.2773, -0.3217],
                        [0.213, -0.5251, 0.3121]])
    out = np.dot(out, rgb2yiq.T.copy())
    return out


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
    out = np.where(img <= 0.0031308, img * 12.92, 1.055 * (np.power(img, (1.0 / 2.4))) - 0.055)
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
        field_out[y, x] = filter_sim(vTexCoord, field_in, SourceSize)
    return


@ti.func
def filter_sim(vTexCoord: tm.vec2, Source, SourceSize: tm.vec4) -> tm.vec3:
    max_L = tm.max(tm.max(L.r, L.g), L.b)
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
        filtered += s / (2.0 * L) * (t1 - t0 + (L / np.pi) * (tm.sin((np.pi / L) * (t - t0)) - tm.sin((np.pi / L) * (t - t1))))
    return filtered


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
    aspect_ratio = OutputSize.x / OutputSize.y

    lower_sample_y = 0
    upper_sample_y = 0
    delta = 0.0
    lower_distance_y = 0.0
    upper_distance_y = 0.0
    # Check if we should be deinterlacing.
    # if INTERLACING and SourceSize.y > 300:
    #     if INTERLACING_EVEN:
    #         lower_sample_y = int(tm.ceil(vTexCoord.y * SourceSize.y + 0.5) * 0.5) * 2 - 2
    #         upper_sample_y = int(tm.ceil(vTexCoord.y * SourceSize.y + 0.5) * 0.5) * 2
    #     else:
    #         lower_sample_y = int(tm.floor(vTexCoord.y * SourceSize.y + 0.5) * 0.5) * 2 - 1
    #         upper_sample_y = int(tm.floor(vTexCoord.y * SourceSize.y + 0.5) * 0.5) * 2 + 1
    #     # TODO 0.5 doesn't quite work with an odd number of lines. Does that ever happen?
    #     delta = 0.5 * aspect_ratio * SourceSize.y * SourceSize.z
    #     lower_distance_y = 0.5 * ((lower_sample_y + 0.5) - vTexCoord.y * SourceSize.y)
    #     upper_distance_y = 0.5 * ((upper_sample_y + 0.5) - vTexCoord.y * SourceSize.y)
    # else:
    lower_sample_y = int(tm.round(vTexCoord.y * SourceSize.y)) - 1
    upper_sample_y = int(tm.round(vTexCoord.y * SourceSize.y))
    delta = aspect_ratio * SourceSize.y * SourceSize.z
    lower_distance_y = (lower_sample_y + 0.5) - vTexCoord.y * SourceSize.y
    upper_distance_y = (upper_sample_y + 0.5) - vTexCoord.y * SourceSize.y

    output = tm.vec3(0.0)
    for sample_x in range(int(tm.round(vTexCoord.x * SourceSize.x - (MAX_SPOT_SIZE / delta))),
                          int(tm.round(vTexCoord.x * SourceSize.x + (MAX_SPOT_SIZE / delta)))):
        lower_sample = texelFetch(img, tm.ivec2(sample_x, lower_sample_y))
        upper_sample = texelFetch(img, tm.ivec2(sample_x, upper_sample_y))
        # Find reciprocal of widths to save divisions later.
        lower_width_rcp = 1.0 / (MAX_SPOT_SIZE * ((1.0 - MIN_SPOT_SIZE) * tm.sqrt(lower_sample) + MIN_SPOT_SIZE))
        upper_width_rcp = 1.0 / (MAX_SPOT_SIZE * ((1.0 - MIN_SPOT_SIZE) * tm.sqrt(upper_sample) + MIN_SPOT_SIZE))
        # Distance units are *scanlines heights*, so we have to adjust x
        # distance with the aspect ratio.
        distance_x = delta * ((sample_x + 0.5) - vTexCoord.x * SourceSize.x)
        lower_output = (lower_sample * lower_width_rcp * 0.25) * \
            (1 + tm.cos(np.pi * tm.clamp(distance_x * lower_width_rcp, -1, 1))) * \
            (1 + tm.cos(np.pi * tm.clamp(lower_distance_y * lower_width_rcp, -1, 1)))
        upper_output = (upper_sample * upper_width_rcp * 0.25) * \
            (1 + tm.cos(np.pi * tm.clamp(distance_x * upper_width_rcp, -1, 1))) * \
            (1 + tm.cos(np.pi * tm.clamp(upper_distance_y * upper_width_rcp, -1, 1)))
        output += lower_output + upper_output
    return delta * output


USE_YIQ = False
GAMMA = 2.4
# -6dB cutoff is at 1 / 2L in cycles. We want CUTOFF * 53.33e-6 cycles (CUTOFF bandwidth and NTSC standard active line time of 53.33us).
# CUTOFF = np.array([5.0e6, 0.6e6, 0.6e6])  # Hz
CUTOFF = np.array([6.0e6, 6.0e6, 6.0e6])  # Hz
# L = 1 / (CUTOFF * 53.33e-6 * 2)
Lnp = 1 / (CUTOFF * 53.33e-6 * 2)
L = tm.vec3(Lnp[0], Lnp[1], Lnp[2])
OUTPUT_RESOLUTION = (2160, 2880)  #(8640, 11520) #(1080, 1440, 3))
MAX_SPOT_SIZE= 0.6
MIN_SPOT_SIZE= 0.5
MASK_AMOUNT = 0.0
BLUR_SIGMA = 75
BLUR_AMOUNT = 0.075
SAMPLES = 1400  # 907
INTERLACING = True
INTERLACING_EVEN = False


def main():
    print('L = {}'.format(L))

    parser = argparse.ArgumentParser(description='Generate a CRT-simulated image')
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    # Read image
    img_original = imread(args.input)
    image_height, image_width, planes = img_original.shape

    # To CRT gamma
    if USE_YIQ:
        img_crt_gamma = srgb_to_yiq(img_original, GAMMA)
    else:
        img_crt_gamma = srgb_to_gamma(img_original, GAMMA)

    # Horizontal low pass filter
    print('Low pass filtering...')
    img_filtered = filter_fragment(img_crt_gamma, (image_height, SAMPLES, 3))

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
    # print('Masking...')
    # mask_resized = imread('mask_slot_distort.png').astype(np.float32) / 65535.0  # 255.0
    # mask_resized = mask_resized / np.max(mask_resized)
    # img_masked = img_spot * ((1 - MASK_AMOUNT) + mask_resized[:, :, 0:3] * MASK_AMOUNT)
    img_masked = img_spot

    # mask_tile = imread('mask.png').astype(np.float32) / 255.0
    # mask = np.tile(mask_tile, ((2 * 250 * 3 // 4), 250, 1))
    # imwrite('mask_fullsized.png', linear_to_srgb(mask))
    # # We have to resize each plane individually because pillow doesn't support
    # # multiple-channel, floating point images.
    # mask_red = mask[:, :, 0]
    # mask_green = mask[:, :, 1]
    # mask_blue = mask[:, :, 2]
    # mask_resized = np.zeros((2160, 2880, 3))
    # mask_resized[:, :, 0] = np.array(Image.fromarray(mask_red, mode='F').resize((2880, 2160), resample=Image.Resampling.LANCZOS))
    # mask_resized[:, :, 1] = np.array(Image.fromarray(mask_green, mode='F').resize((2880, 2160), resample=Image.Resampling.LANCZOS))
    # mask_resized[:, :, 2] = np.array(Image.fromarray(mask_blue, mode='F').resize((2880, 2160), resample=Image.Resampling.LANCZOS))
    # mask_resized = mask_resized / np.max(mask_resized)
    # mask_resized = np.minimum(mask_resized, 0)
    # imwrite('mask_resized.png', linear_to_srgb(mask_resized))
    # img_masked = mask_resized * img_spot

    # Diffusion
    # print('Blurring...')
    # blurred = skimage.filters.gaussian(img_masked, sigma=BLUR_SIGMA, mode='constant', preserve_range=True, channel_axis=-1)
    # imwrite('blurred.png', linear_to_srgb(blurred))  # DEBUG
    # img_diffused = img_masked + (blurred - img_masked) * BLUR_AMOUNT
    img_diffused = img_masked

    # To sRGB
    img_final_srgb = linear_to_srgb(img_diffused)

    #DEBUG
    imwrite('overexposed.png', img_final_srgb - np.clip(img_final_srgb, 0, 255))

    imwrite(args.output, img_final_srgb)


if __name__ == '__main__':
    main()
