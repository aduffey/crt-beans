from imageio.v3 import imwrite
import numpy as np


def cos_sep_spot(x, y):
    return (0.5 + 0.5 * np.cos(np.pi * np.clip(x, -1, 1))) * (0.5 + 0.5 * np.cos(np.pi * np.clip(y, -1, 1)))


def cos_isotropic_spot(x, y):
    return (0.5 + 0.5 * np.cos(np.pi * np.clip(np.sqrt(x**2 + y**2), -1, 1)))


def cubic_sep_spot(x, y):
    x = np.clip(np.abs(x), 0.0, 1.0)
    y = np.clip(np.abs(y), 0.0, 1.0)
    return ((x * x) * (2.0 * x - 3.0) + 1.0) * ((y * y) * (2.0 * y - 3.0) + 1.0)


def vcubic_hcos_spot(x, y):
    x = np.clip(np.abs(x), 0.0, 1.0)
    y = np.clip(np.abs(y), 0.0, 1.0)
    return (0.5 + 0.5 * np.cos(np.pi * x)) * ((y * y) * (2.0 * y - 3.0) + 1.0)


def gauss_spot(x, y):
    c = 1 / (2 * np.sqrt(2 * np.log(2)))
    #return 1 / (np.sqrt(2 * np.pi) * c) * np.exp(-x**2 / (2 * c**2)) * np.exp(-y**2 / (2 * c**2))
    return np.exp(-x**2 / (2 * c**2)) * np.exp(-y**2 / (2 * c**2))


def vcubic_hgauss_spot(x, y):
    c = 1 / (2 * np.sqrt(2 * np.log(2)))
    y = np.clip(np.abs(y), 0.0, 1.0)
    return np.exp(-x**2 / (2 * c**2)) * ((y * y) * (2.0 * y - 3.0) + 1.0)


def linear_to_srgb(img):
    """Linear float to sRGB uint8"""
    out = np.where(img < 0.0031308, img * 12.92, 1.055 * (np.power(img, (1.0 / 2.4))) - 0.055)
    out = (img * 255).astype(np.uint8)
    return out


def main():
    xaxis = np.linspace(-1.5, 1.5, 800)
    yaxis = np.linspace(-1.5, 1.5, 800)

    gauss_result = gauss_spot(xaxis[None,:], yaxis[:,None])
    imwrite("gauss_spot.png", linear_to_srgb(gauss_result))

    cos_sep_result = cos_sep_spot(xaxis[None,:], yaxis[:,None])
    imwrite("cos_sep_spot.png", linear_to_srgb(cos_sep_result))

    cos_isotropic_result = cos_isotropic_spot(xaxis[None,:], yaxis[:,None])
    imwrite("cos_isotropic_spot.png", linear_to_srgb(cos_isotropic_result))

    cubic_sep_result = cubic_sep_spot(xaxis[None,:], yaxis[:,None])
    imwrite("cubic_sep_spot.png", linear_to_srgb(cubic_sep_result))

    vcubic_hcos_result = vcubic_hcos_spot(xaxis[None,:], yaxis[:,None])
    imwrite("vcubic_hcos_spot.png", linear_to_srgb(vcubic_hcos_result))

    vcubic_hgauss_result = vcubic_hgauss_spot(xaxis[None,:], yaxis[:,None])
    imwrite("vcubic_hgauss_spot.png", linear_to_srgb(vcubic_hgauss_result))


if __name__ == '__main__':
    main()
