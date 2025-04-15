from imageio.v3 import imwrite, imread
import numpy as np


OUTPUT_RESOLUTION = (1440, 1080)  # (1440, 1080) (2880, 2160)


def linear_to_srgb(img):
    """Linear float to sRGB uint8"""
    img_clipped = np.clip(img, 0.0, 1.0)
    out = np.where(img_clipped <= 0.0031308, img_clipped * 12.92, 1.055 * (np.power(img_clipped, (1.0 / 2.4))) - 0.055)
    out = np.around(out * 255).astype(np.uint8)
    return out


def linear_to_gamma(img):
    """Linear float to 2.2 gamma uint8"""
    img_clipped = np.clip(img, 0.0, 1.0)
    out = np.power(img_clipped, (1.0 / 2.2))
    out = np.around(out * 255).astype(np.uint8)
    return out


def luminance(img):
    """Calculate luminance of each pixel and broadcast it to all subpixels"""
    lum = np.dot(img, [0.2126, 0.7152, 0.0722])
    return np.dstack((lum, lum, lum))


def max_pixel(img):
    m = np.max(img, axis=2)
    return np.dstack((m, m, m))


def main():
    # Generate a 0.0 -> 1.0 linear gradient
    grad = np.power(np.linspace(0.0, 1.0, OUTPUT_RESOLUTION[0]), 2)
    grad_2d = np.tile(grad, (OUTPUT_RESOLUTION[1], 1))
    grad_rgb = np.dstack((grad_2d, grad_2d, grad_2d))

    # BGR mask gives 480 triads per screen width at 1440 pixels and reduces brightness by a factor of 3.
    # mask_tile = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    # mask_coverage = 3
    mask_tile = np.array([[1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0]])  # 4k, lower TVL
    mask_coverage = 15 / 6
    mask = np.broadcast_to(mask_tile[np.arange(OUTPUT_RESOLUTION[0]) % mask_tile.shape[0]], (OUTPUT_RESOLUTION[1], OUTPUT_RESOLUTION[0], 3))

    # Linear phase-in. The original image starts phasing in immediately, linearly up until 1.
    weight = grad_rgb
    img_masked_linear = grad_rgb * grad_rgb + mask_coverage * mask * (1.0 - grad_rgb) * grad_rgb

    # Piecewise phase-in. The original image starts phasing in at 1/3 luminance, then linearly up until 1.
    s = mask_coverage / (mask_coverage - 1)
    weight = np.clip(-s * grad_rgb + s, 0.0, 1.0)
    img_masked_piecewise = (1 - weight) * grad_rgb + mask_coverage * mask * weight * grad_rgb

    # Bezier phase-in. Ramps up the mask to higher strength faster than linear but has no discontinuity like piecewise.
    s = mask_coverage / (mask_coverage - 1)
    a = -s + 2
    b = s - 3
    weight = a * np.power(grad_rgb, 3) + b * np.power(grad_rgb, 2) + 1
    img_masked_bezier = (1 - weight) * grad_rgb + mask_coverage * mask * weight * grad_rgb

    # Assemble the output image.
    # * Linear phase-in is on top.
    # * The original is in the middle.
    # * Piecewise phase-in is on the bottom.
    output = np.zeros_like(grad_rgb)
    one_third = int(OUTPUT_RESOLUTION[1] / 3)
    two_third = int(2 * OUTPUT_RESOLUTION[1] / 3)
    output[:one_third] = img_masked_piecewise[:one_third]
    output[one_third:two_third] = grad_rgb[one_third:two_third]
    output[two_third:] = img_masked_bezier[two_third:]
    output[one_third] = np.array([0.0, 1.0, 1.0])
    output[two_third] = np.array([0.0, 1.0, 1.0])

    print("Original: {}".format(np.mean(grad_rgb, axis=(0,1), dtype=np.float64)))
    print("Linear: {}".format(np.mean(img_masked_linear, axis=(0,1), dtype=np.float64)))
    print("Piecewise: {}".format(np.mean(img_masked_piecewise, axis=(0,1), dtype=np.float64)))
    print("Bezier: {}".format(np.mean(img_masked_bezier, axis=(0,1), dtype=np.float64)))

    imwrite('gradients_srgb.png', linear_to_srgb(output))
    imwrite('gradients_gamma.png', linear_to_gamma(output))

    half = int(OUTPUT_RESOLUTION[1] / 2)
    output[:half] = img_masked_piecewise[:half]
    output[half:] = img_masked_bezier[half:]
    output[half] = np.array([0.0, 1.0, 1.0])
    imwrite('gradients_half_srgb.png', linear_to_srgb(output))


if __name__ == '__main__':
    main()
