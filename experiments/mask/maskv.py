from imageio.v3 import imwrite, imread
import numpy as np
import taichi as ti
import taichi.math as tm


ti.init()


OUTPUT_RESOLUTION = (2880, 2160)  # (1440, 1080) (2880, 2160)


def linear_to_srgb(img):
    """Linear float to sRGB uint8"""
    img_clipped = np.clip(img, 0.0, 1.0)
    out = np.where(img_clipped <= 0.0031308, img_clipped * 12.92, 1.055 * (np.power(img_clipped, (1.0 / 2.4))) - 0.055)
    # scaled = np.where(np.logical_and(out * 255 < 1, out * 255 > 0), np.array([[[255.0, 0.0, 0.0]]]), out * 255)
    scaled = out * 255
    out = np.around(scaled).astype(np.uint8)
    return out


def linear_to_gamma(img):
    """Linear float to 2.2 gamma uint8"""
    img_clipped = np.clip(img, 0.0, 1.0)
    out = np.power(img_clipped, (1.0 / 2.2))
    out = np.around(out * 255).astype(np.uint8)
    return out


def bandlimit_mask(image_in, mask_triads):
    (in_height, in_width, in_planes) = image_in.shape
    field_in = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_in.from_numpy(image_in)
    field_out = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    bandlimit_mask_fragment(field_in, field_out, mask_triads)
    return field_out.to_numpy()


@ti.kernel
def bandlimit_mask_fragment(field_in: ti.template(), field_out: ti.template(), mask_triads: int):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = bandlimit_mask_taichi2(vTexCoord, field_in, SourceSize, OutputSize, mask_triads)
    return


phosphors = ti.Vector.field(n=3, dtype=float, shape=3)
phosphors[0] = tm.vec3(1.0, 0.0, 0.0)
phosphors[1] = tm.vec3(0.0, 1.0, 0.0)
phosphors[2] = tm.vec3(0.0, 0.0, 1.0)


@ti.func
def bandlimit_mask_taichi2(vTexCoord: tm.vec2, Source, SourceSize: tm.vec4, OutputSize: tm.vec4, mask_triads: int):
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
        mask = np.pi * (p0 + p2) + \
            (p1 - p0) * (np.pi * x1 + tm.sin(np.pi * x1)) + \
            (p2 - p1) * (np.pi * x2 + tm.sin(np.pi * x2))
        mask /= 2.0 * np.pi
    else:  # w < 1.5
        x1 = tm.clamp((x - (tm.round(x) - 1.0)) / w, -1.0, 1.0)
        x2 = tm.clamp((x - tm.round(x)) / w, -1.0, 1.0)
        x3 = tm.clamp((x - (tm.round(x) + 1.0)) / w, -1.0, 1.0)
        p0 = phosphors[int(tm.round(x) - 2.0) % 3]
        p1 = phosphors[int(tm.round(x) - 1.0) % 3]
        p2 = phosphors[int(tm.round(x)) % 3]
        p3 = phosphors[int(tm.round(x) + 1.0) % 3]
        mask = np.pi * (p0 + p3) + \
            (p1 - p0) * (np.pi * x1 + tm.sin(np.pi * x1)) + \
            (p2 - p1) * (np.pi * x2 + tm.sin(np.pi * x2)) + \
            (p3 - p2) * (np.pi * x3 + tm.sin(np.pi * x3))
        mask /= 2.0 * np.pi

    if mask.x < 0 or mask.x > 1 or mask.y < 0 or mask.y > 1 or mask.z < 0 or mask.z > 1:
        print(mask)
    return mask


def additive_mask(image_in, mask_triads):
    (in_height, in_width, in_planes) = image_in.shape
    field_in = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_in.from_numpy(image_in)
    field_out = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    additive_mask_fragment(field_in, field_out, mask_triads)
    return field_out.to_numpy()


@ti.kernel
def additive_mask_fragment(field_in: ti.template(), field_out: ti.template(), mask_triads: int):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    OutputSize = tm.vec4(out_width, out_height, 1 / out_width, 1 / out_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = additive_mask_taichi(vTexCoord, field_in, SourceSize, OutputSize, mask_triads)
    return


@ti.func
def additive_mask_taichi(vTexCoord: tm.vec2, Source, SourceSize: tm.vec4, OutputSize: tm.vec4, mask_triads: int):
    offset = tm.vec3(0.0, -1 / (3 * mask_triads), -2 / (3 * mask_triads))
    half = OutputSize.z * 0.5
    x = vTexCoord.x
    mask = tm.vec3(1.0, 1.0, 1.0)
    mask_coverage = 1.0
    # if mask_triads * 4 < OutputSize.x:
    #     # mask = 1 / 3 + \
    #     #     2 / np.pi * tm.sin(np.pi / 3) * tm.cos(2 * np.pi * MASK_TRIADS * (x + offset)) + \
    #     #     1 / np.pi * tm.sin((2 / 3) * np.pi) * tm.cos(4 * np.pi * MASK_TRIADS * (x + offset))

    #     # mask = 1 / 3 + \
    #     #     0.5513288954217920 * tm.sin(2 * np.pi * MASK_TRIADS * (x + offset)) + \
    #     #     0.2756644477108960 * tm.sin(4 * np.pi * MASK_TRIADS * (x + offset))
    #     # mask = (mask + 0.080163) / 1.24049

    #     # mask = 1.0 / 3.0 * OutputSize.z + 0.5513288954217920 * \
    #     #         (tm.sin(2 * np.pi * MASK_TRIADS * (x + offset + half)) - \
    #     #         tm.sin(2 * np.pi * MASK_TRIADS * (x + offset - half))) / (2 * np.pi * MASK_TRIADS) + \
    #     #         0.2756644477108960 * (tm.sin(4 * np.pi * MASK_TRIADS * (x + offset + half)) - \
    #     #         tm.sin(4 * np.pi * MASK_TRIADS * (x + offset - half))) / (4 * np.pi * MASK_TRIADS)
    #     # mask /= OutputSize.z

    #     # Area under offset
    #     mask = (1.0 / 3.0 + 0.080163) * OutputSize.z
    #     # Area under first harmonic
    #     right = tm.sin(2 * np.pi * mask_triads * (x + offset + half))
    #     left = tm.sin(2 * np.pi * mask_triads * (x + offset - half))
    #     mask += 0.5513288954217920 * (right - left) / (2 * np.pi * mask_triads)
    #     # Area under second harmonic
    #     right = tm.sin(4 * np.pi * mask_triads * (x + offset + half))
    #     left = tm.sin(4 * np.pi * mask_triads * (x + offset - half))
    #     mask += 0.2756644477108960 * (right - left) / (4 * np.pi * mask_triads)
    #     mask /= (1.24049 * OutputSize.z)

    #     # mask_coverage = 3
    #     mask_coverage = 1 / (1 / 3 + 0.080163)
    # elif mask_triads * 2 < OutputSize.x:
    # mask = 0.5 + 0.5 * tm.cos(2 * np.pi * MASK_TRIADS * (x + offset))
    max = half + tm.sin(2 * np.pi * mask_triads * half) / (2 * np. pi * mask_triads)  # XXX
    mask = 0.5 * OutputSize.z + 0.5 * \
            (tm.sin(2 * np.pi * mask_triads * (x + offset + half)) - \
            tm.sin(2 * np.pi * mask_triads * (x + offset - half))) / (2 * np.pi * mask_triads)
    # mask = 1 / 3 + \
    #     0.5513288954217920 * tm.cos(2 * np.pi * MASK_TRIADS * (x + offset))
    # mask = 1.0 / 3.0 * OutputSize.z + 0.5513288954217920 * \
    #         (tm.sin(2 * np.pi * MASK_TRIADS * (x + offset + half)) - \
    #         tm.sin(2 * np.pi * MASK_TRIADS * (x + offset - half))) / (2 * np.pi * MASK_TRIADS)
    mask /= OutputSize.z
    mask_coverage = 2
    return mask


@ti.func
def texelFetch(Source, vTexCoords: tm.ivec2) -> tm.vec3:
    """Fetch pixel from img at (x, y), or 0 if outside range"""
    y_size, x_size = Source.shape
    val = tm.vec3(0.0)
    if not (vTexCoords.x < 0 or vTexCoords.x >= x_size or vTexCoords.y < 0 or vTexCoords.y >= y_size):
        val = Source[vTexCoords.y, vTexCoords.x]
    return val


def main():
    # Generate a 0.0 -> 1.0 linear gradient
    grad = np.power(np.linspace(0.0, 1.0, OUTPUT_RESOLUTION[1]), 2).reshape((-1, 1))
    grad_2d = np.tile(grad, (1, OUTPUT_RESOLUTION[0]))
    # grad_rgb = np.dstack((np.zeros_like(grad_2d), grad_2d, np.zeros_like(grad_2d))) # Green only
    grad_rgb = np.dstack((grad_2d, grad_2d, grad_2d)) # White

    # BGR mask gives 480 triads per screen width at 1440 pixels and reduces brightness by a factor of 3.
    # mask_tile = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    # mask_coverage = 3
    mask_tile = np.array([[1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0]])  # 4k, lower TVL
    mask = np.broadcast_to(mask_tile[np.arange(OUTPUT_RESOLUTION[0]) % mask_tile.shape[0]], (OUTPUT_RESOLUTION[1], OUTPUT_RESOLUTION[0], 3))
    mask_coverage = 15 / 6

    # Mask phase-in
    a = np.clip((grad_rgb - 1) / (1 - mask_coverage), 0, grad_rgb)
    b = np.clip((1 - mask_coverage * grad_rgb) / (1 - mask_coverage), 0, grad_rgb)
    img_masked_subpixel = mask_coverage * a * mask + b
    # print(img_masked_subpixel[1666, 0:24, 1])
    # print(np.logical_and(img_masked_subpixel[1666, 0:24, 1] <= (1.0 + 1e-4), img_masked_subpixel[1666, 0:24, 1] >= 0.0))

    # Color any out-of-range values as red (we shouldn't see any!) XXX
    img_masked_subpixel = np.where(np.logical_and(img_masked_subpixel <= (1.0 + 1e-4), img_masked_subpixel >= 0.0),
        img_masked_subpixel, np.array([[[1.0, 0.0, 0.0]]]))

    mask = bandlimit_mask(grad_rgb, 550)
    mask_coverage = 3
    np.set_printoptions(threshold=1000)
    print(mask[0, 0:24, 1])

    # Mask phase-in
    a = np.clip((grad_rgb - 1) / (1 - mask_coverage), 0, grad_rgb)
    b = np.clip((1 - mask_coverage * grad_rgb) / (1 - mask_coverage), 0, grad_rgb)
    img_masked_dynamic = mask_coverage * a * mask + b

    # Color any out-of-range values as red (we shouldn't see any!) XXX
    img_masked_dynamic = np.where(np.logical_and(img_masked_dynamic <= (1.0 + 1e-4), img_masked_dynamic >= 0.0),
        img_masked_dynamic, np.array([[[255.0, 0.0, 0.0]]]))

    # Assemble the output image.
    # * Subpixel mask is on the left.
    # * No mask is in the middle.
    # * Dynamic mask is on the right.
    output = np.zeros_like(grad_rgb)
    one_third = int(OUTPUT_RESOLUTION[0] / 3)
    two_third = int(2 * OUTPUT_RESOLUTION[0] / 3)
    output[:,:one_third] = img_masked_subpixel[:,:one_third]
    output[:,one_third:two_third] = grad_rgb[:,one_third:two_third]
    output[:,two_third:] = img_masked_dynamic[:,two_third:]

    imwrite('gradients_vert_srgb.png', linear_to_srgb(output))
    imwrite('gradients_vert_gamma.png', linear_to_gamma(output))

    img_mean_subpixel = np.tile(np.mean(img_masked_subpixel, 1).reshape(-1, 1, 3), (1, OUTPUT_RESOLUTION[0], 1))
    img_mean_dynamic = np.tile(np.mean(img_masked_dynamic, 1).reshape(-1, 1, 3), (1, OUTPUT_RESOLUTION[0], 1))

    output[:,:one_third] = img_mean_subpixel[:,:one_third]
    output[:,one_third:two_third] = grad_rgb[:,one_third:two_third]
    output[:,two_third:] = img_mean_dynamic[:,two_third:]

    imwrite('gradients_mean_vert_srgb.png', linear_to_srgb(output))
    imwrite('gradients_mean_vert_gamma.png', linear_to_gamma(output))


if __name__ == '__main__':
    main()
