from imageio.v3 import imwrite, imread
import numpy as np
import skimage
import argparse
import multiprocessing as mp
import functools
from PIL import Image, ImageFilter

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
    return np.power(img, in_gamma)


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


def fragment(fn, image_in, output_dim):
    pool = mp.Pool(mp.cpu_count())
    image_out = np.zeros(output_dim)
    (height, width, planes) = output_dim
    for y, row in enumerate(pool.map(functools.partial(__fragment_helper, fn=fn, image_in=image_in, output_dim=output_dim), range(height))):
        image_out[y] = row
    return image_out


def __fragment_helper(y, fn, image_in, output_dim):
    (height, width, planes) = output_dim
    row_out = np.zeros((width, planes))
    for x in range(width):
        row_out[x] = fn(((x + 0.5) / width, (y + 0.5) / height), image_in, output_dim)
    return row_out


def fetch(img, x, y):
    """Fetch pixel from img at (x, y), or 0 if outside range"""
    y_size, x_size, planes = img.shape
    if x < 0 or x >= x_size or y < 0 or y >= y_size:
        return np.zeros(planes)
    return img[y, x]


def filter_sim(loc, img, out_shape):
    image_height, image_width, planes = img.shape
    pixel_width = 1 / image_width

    filtered = np.zeros(planes)
    pix_y = np.floor(loc[1] * image_height).astype(int)
    t = loc[0]
    for pix_x in range(np.floor(image_width * (loc[0] - np.max(L))).astype(int),
                       np.floor(image_width * (loc[0] + np.max(L))).astype(int) + 1):
        s = fetch(img, pix_x, pix_y)
        t0 = pix_x / image_width
        t1 = t0 + pixel_width
        t0 = np.clip(t0, t - L, t + L)
        t1 = np.clip(t1, t - L, t + L)
        # Integral of s * (1 / L) * (0.5 + 0.5 * cos(PI * (t - t_x) / L)) dt_x over t0 to t1
        f = s / (2 * L) * (t1 - t0 + (L / np.pi) * (np.sin((np.pi / L) * (t - t0)) - np.sin((np.pi / L) * (t - t1))))
        filtered += f
    return filtered


def spot_sim_anisotropic(loc, img, out_shape):
    image_height, image_width, planes = img.shape
    out_height, out_width, out_planes = out_shape
    aspect_ratio = out_width / out_height
    delta = (aspect_ratio * image_height) / image_width

    output = np.zeros(planes)
    lower_pix_loc = (np.floor(loc[0] * image_width).astype(int), np.around(loc[1] * image_height).astype(int) - 1)
    upper_pix_loc = (np.floor(loc[0] * image_width).astype(int), np.around(loc[1] * image_height).astype(int))
    lower_distance_y = image_height * ((lower_pix_loc[1] + 0.5) / image_height - loc[1])  # TODO can simplify this (multiply out image_height)
    upper_distance_y = image_height * ((upper_pix_loc[1] + 0.5) / image_height - loc[1])
    for i in range(np.round(loc[0] * image_width - (1 / delta)).astype(int), np.round(loc[0] * image_width + (1 / delta)).astype(int)):
        lower_s = fetch(img, i, lower_pix_loc[1])
        upper_s = fetch(img, i, upper_pix_loc[1])
        lower_width = MAX_SPOT_SIZE * ((1.0 - MIN_SPOT_SIZE) * np.sqrt(lower_s) + MIN_SPOT_SIZE)
        upper_width = MAX_SPOT_SIZE * ((1.0 - MIN_SPOT_SIZE) * np.sqrt(upper_s) + MIN_SPOT_SIZE)
        # Distance units are *scanlines heights*
        distance_x = aspect_ratio * image_height * ((i + 0.5) / image_width - loc[0])
        lower_output = (lower_s / (4 * lower_width)) * \
            (1 + np.cos(np.pi * np.clip(distance_x / lower_width, -1, 1))) * \
            (1 + np.cos(np.pi * np.clip(lower_distance_y / lower_width, -1, 1)))
        upper_output = (upper_s / (4 * upper_width)) * \
            (1 + np.cos(np.pi * np.clip(distance_x / upper_width, -1, 1))) * \
            (1 + np.cos(np.pi * np.clip(upper_distance_y / upper_width, -1, 1)))
        output += lower_output + upper_output
    return np.clip(delta * output, 0.0, 1.0)


USE_YIQ = False
GAMMA = 2.4
# -6dB cutoff is at 1 / 2L in cycles. We want CUTOFF * 53.33e-6 cycles (CUTOFF bandwidth and NTSC standard active line time of 53.33us).
# CUTOFF = np.array([5.0e6, 0.6e6, 0.6e6])  # Hz
CUTOFF = np.array([6.0e6, 6.0e6, 6.0e6])  # Hz
# L = 1 / (CUTOFF * 53.33e-6 * 2)
L = 1 / (CUTOFF * 53.33e-6 * 2) * (256 / 38)
MAX_SPOT_SIZE= 0.8
MIN_SPOT_SIZE= 0.3
MASK_AMOUNT = 0.0
BLUR_SIGMA = 75
BLUR_AMOUNT = 0.075
SAMPLES = 907


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
    img_filtered = fragment(filter_sim, img_crt_gamma, (int(image_height), SAMPLES, 3))

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
    img_spot = fragment(spot_sim_anisotropic, img_filtered_linear, (2160, 2880, 3))  #(1080, 1440, 3))

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
    print('Blurring...')
    blurred = skimage.filters.gaussian(img_masked, sigma=BLUR_SIGMA, mode='constant', preserve_range=True, channel_axis=-1)
    imwrite('blurred.png', linear_to_srgb(blurred))  # DEBUG
    img_diffused = img_masked + (blurred - img_masked) * BLUR_AMOUNT

    # To sRGB
    img_final_srgb = linear_to_srgb(img_diffused)

    imwrite(args.output, img_final_srgb)


if __name__ == '__main__':
    main()
