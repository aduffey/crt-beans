from imageio.v3 import imwrite, imread
import numpy as np
import matplotlib.pyplot as plt


def luminance(img):
    return np.dot(img, [0.2126, 0.7152, 0.0722])


def main():
    values = list(reversed([1.0 / (2**x) for x in range(10)]))
    linear_means = []
    piecewise_means = []
    bezier_means = []
    for val in values:
        img = np.full((2160, 2880, 3), val)
        linear_masked, piecewise_masked, bezier_masked = mask(img)
        linear_means.append(np.mean(linear_masked, axis=(0,1)))
        piecewise_means.append(np.mean(piecewise_masked, axis=(0,1)))
        bezier_means.append(np.mean(bezier_masked, axis=(0,1)))

    fig, ax = plt.subplots()
    ax.plot(values, values)
    ax.plot(values, [r for (r, g, b) in linear_means], 'ro-')
    ax.plot(values, [g for (r, g, b) in linear_means], 'go-')
    ax.plot(values, [b for (r, g, b) in linear_means], 'bo-')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(values, values)
    ax.plot(values, [r for (r, g, b) in piecewise_means], 'ro-')
    ax.plot(values, [g for (r, g, b) in piecewise_means], 'go-')
    ax.plot(values, [b for (r, g, b) in piecewise_means], 'bo-')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(values, values)
    ax.plot(values, [r for (r, g, b) in bezier_means], 'ro-')
    ax.plot(values, [g for (r, g, b) in bezier_means], 'go-')
    ax.plot(values, [b for (r, g, b) in bezier_means], 'bo-')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    plt.show()

def mask(img):
    (height, width, depth) = img.shape
    # BGR mask gives 480 triads per screen width at 1440 pixels and reduces brightness by a factor of 3.
    # mask_tile = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    # mask_coverage = 3
    mask_tile = np.array([[1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0]])  # 4k, lower TVL
    mask_coverage = 15 / 6
    mask = np.broadcast_to(mask_tile[np.arange(width) % mask_tile.shape[0]], (height, width, 3))

    # Linear phase-in. The original image starts phasing in immediately, linearly up until 1.
    weight = img
    img_masked_linear = img * img + mask_coverage * mask * (1.0 - img) * img

    # Piecewise phase-in. The original image starts phasing in at 1/3 luminance, then linearly up until 1.
    s = mask_coverage / (mask_coverage - 1)
    weight = np.clip(-s * img + s, 0.0, 1.0)
    img_masked_piecewise = (1 - weight) * img + mask_coverage * mask * weight * img

    # Bezier phase-in. Ramps up the mask to higher strength faster than linear but has no discontinuity like piecewise.
    s = mask_coverage / (mask_coverage - 1)
    a = -s + 2
    b = s - 3
    weight = a * np.power(img, 3) + b * np.power(img, 2) + 1
    img_masked_bezier = (1 - weight) * img + mask_coverage * mask * weight * img

    return img_masked_linear, img_masked_piecewise, img_masked_bezier

if __name__ == '__main__':
    main()
