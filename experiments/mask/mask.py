from imageio.v3 import imwrite, imread
import numpy as np


OUTPUT_RESOLUTION = (1440, 1080)


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


def main():
    # Generate a 0.0 -> 1.0 linear gradient
    grad = np.linspace(0.0, 1.0, OUTPUT_RESOLUTION[0])
    grad_2d = np.tile(grad, (OUTPUT_RESOLUTION[1], 1))
    grad_rgb = np.dstack((grad_2d, grad_2d, grad_2d))

    # BGR mask gives 480 triads per screen width at 1440 pixels and reduces brightness by a factor of 3.
    mask_tile = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    mask = np.broadcast_to(mask_tile[np.arange(OUTPUT_RESOLUTION[0]) % mask_tile.shape[0]], (OUTPUT_RESOLUTION[1], OUTPUT_RESOLUTION[0], 3))

    # Linear phase-in. The original image starts phasing in immediately, linearly up until 1.
    lum = luminance(grad_rgb)
    img_masked_linear = lum * grad_rgb + 3 * mask * (1.0 - lum) * grad_rgb

    # Piecewise phase-in. The original image starts phasing in at 1/3 luminance, then linearly up until 1.
    lum = np.clip(1.5 * lum - 0.5, 0.0, 1.0)
    img_masked_piecewise = lum * grad_rgb + 3 * mask * (1.0 - lum) * grad_rgb

    # Assemble the output image.
    # * Linear phase-in is on top.
    # * The original is in the middle.
    # * Piecewise phase-in is on the bottom.
    output = np.zeros_like(grad_rgb)
    one_third = int(OUTPUT_RESOLUTION[1] / 3)
    two_third = int(2 * OUTPUT_RESOLUTION[1] / 3)
    output[:one_third] = img_masked_linear[:one_third]
    output[one_third:two_third] = grad_rgb[one_third:two_third]
    output[two_third:] = img_masked_piecewise[two_third:]
    output[one_third] = np.array([0.0, 1.0, 1.0])
    output[two_third] = np.array([0.0, 1.0, 1.0])

    imwrite('gradients_srgb.png', linear_to_srgb(output))
    imwrite('gradients_gamma.png', linear_to_gamma(output))


if __name__ == '__main__':
    main()
