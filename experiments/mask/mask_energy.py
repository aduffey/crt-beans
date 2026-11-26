import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import taichi.math as tm
from imageio.v3 import imread, imwrite

ti.init()


def luminance(img):
    return np.dot(img, [0.2126, 0.7152, 0.0722])


def linear_to_srgb(img):
    """Linear float to sRGB uint8"""
    img_clipped = np.clip(img, 0.0, 1.0)
    out = np.where(
        img_clipped <= 0.0031308,
        img_clipped * 12.92,
        1.055 * (np.power(img_clipped, (1.0 / 2.4))) - 0.055,
    )
    out = np.around(out * 255).astype(np.uint8)
    return out


def main():
    values = list(reversed([1.0 / (1.2**x) for x in range(30)]))
    means = []
    for val in values:
        img = np.full((2160, 2880, 3), val)
        masked = np.clip(mask(img), 0, 1)
        means.append(np.mean(masked, axis=(0, 1)))
        # imwrite(f'original_{val}.png', linear_to_srgb(img))
        # imwrite(f'masked_{val}.png', linear_to_srgb(masked))

    fig, ax = plt.subplots()
    ax.plot(values, values)
    ax.plot(values, [r for (r, g, b) in means], "ro-")
    ax.plot(values, [g for (r, g, b) in means], "go-")
    ax.plot(values, [b for (r, g, b) in means], "bo-")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    plt.show()


def bandlimit_mask(image_in, mask_triads):
    (in_height, in_width, in_planes) = image_in.shape
    field_in = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_in.from_numpy(image_in)
    field_out = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    bandlimit_mask_fragment(field_in, field_out, mask_triads)
    return field_out.to_numpy()


@ti.kernel
def bandlimit_mask_fragment(
    field_in: ti.template(), field_out: ti.template(), mask_triads: int
):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = bandlimit_mask_taichi2(
            vTexCoord, field_in, SourceSize, OutputSize, mask_triads
        )
    return


phosphors = ti.Vector.field(n=3, dtype=float, shape=3)
phosphors[0] = tm.vec3(1.0, 0.0, 0.0)
phosphors[1] = tm.vec3(0.0, 1.0, 0.0)
phosphors[2] = tm.vec3(0.0, 0.0, 1.0)


@ti.func
def bandlimit_mask_taichi2(
    vTexCoord: tm.vec2,
    Source,
    SourceSize: tm.vec4,
    OutputSize: tm.vec4,
    mask_triads: int,
):
    # With unrolled loops and common terms collapsed for performance
    w = (mask_triads * 3.0) / OutputSize.x
    x = mask_triads * 3.0 * vTexCoord.x
    mask = tm.vec3(0.0)
    if w < 0.5:
        x1 = tm.clamp((x - tm.round(x)) / w, -1.0, 1.0)
        p0 = phosphors[int(tm.round(x) - 1.0) % 3]
        p1 = phosphors[int(tm.round(x)) % 3]
        mask = np.pi * (p0 + p1) + (p1 - p0) * (np.pi * x1 + tm.sin(np.pi * x1))
        mask /= 2.0 * np.pi
    elif w < 1.0:
        x1 = tm.clamp((x - tm.floor(x)) / w, -1.0, 1.0)
        x2 = tm.clamp((x - (tm.floor(x) + 1.0)) / w, -1.0, 1.0)
        p0 = phosphors[int(tm.floor(x) - 1.0) % 3]
        p1 = phosphors[int(tm.floor(x)) % 3]
        p2 = phosphors[int(tm.floor(x) + 1.0) % 3]
        mask = (
            np.pi * (p0 + p2)
            + (p1 - p0) * (np.pi * x1 + tm.sin(np.pi * x1))
            + (p2 - p1) * (np.pi * x2 + tm.sin(np.pi * x2))
        )
        mask /= 2.0 * np.pi
    else:  # w < 1.5
        x1 = tm.clamp((x - (tm.round(x) - 1.0)) / w, -1.0, 1.0)
        x2 = tm.clamp((x - tm.round(x)) / w, -1.0, 1.0)
        x3 = tm.clamp((x - (tm.round(x) + 1.0)) / w, -1.0, 1.0)
        p0 = phosphors[int(tm.round(x) - 2.0) % 3]
        p1 = phosphors[int(tm.round(x) - 1.0) % 3]
        p2 = phosphors[int(tm.round(x)) % 3]
        p3 = phosphors[int(tm.round(x) + 1.0) % 3]
        mask = (
            np.pi * (p0 + p3)
            + (p1 - p0) * (np.pi * x1 + tm.sin(np.pi * x1))
            + (p2 - p1) * (np.pi * x2 + tm.sin(np.pi * x2))
            + (p3 - p2) * (np.pi * x3 + tm.sin(np.pi * x3))
        )
        mask /= 2.0 * np.pi

    if mask.x < 0 or mask.x > 1 or mask.y < 0 or mask.y > 1 or mask.z < 0 or mask.z > 1:
        print(mask)
    return mask


def mask(img):
    (height, width, depth) = img.shape
    # BGR mask gives 480 triads per screen width at 1440 pixels and reduces brightness by a factor of 3.
    # mask_tile = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    # mask_coverage = 3
    # mask_tile = np.array(
    #     [[1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0]]
    # )  # 4k, lower TVL
    # mask_coverage = 15 / 6
    # mask = np.broadcast_to(
    #     mask_tile[np.arange(width) % mask_tile.shape[0]], (height, width, 3)
    # )

    mask = bandlimit_mask(img, 550)
    mask_coverage = 3.0

    s = 2 / 3
    a = np.minimum((s * img - 1) / (1 - mask_coverage), s * img)
    b = np.maximum((1 - mask_coverage * s * img) / (1 - mask_coverage), 0)
    return mask_coverage * a * mask + b


if __name__ == "__main__":
    main()
