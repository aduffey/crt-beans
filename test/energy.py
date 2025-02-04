import sys

import numpy as np
import sim_taichi
from imageio.v3 import imwrite, imread


def linear_to_srgb(img):
    """Linear float to sRGB uint8"""
    img_clipped = np.clip(img, 0.0, 1.0)
    out = np.where(img_clipped <= 0.0031308, img_clipped * 12.92, 1.055 * (np.power(img_clipped, (1.0 / 2.4))) - 0.055)
    out = np.around(out * 255).astype(np.uint8)
    return out

def gamma_to_linear(img):
    """Simple gamma uint8 to linear float"""
    img_float = img.astype(np.float32) / 255
    return np.power(np.clip(img_float, 0.0, 1.0), 2.4)

filename = sys.argv[1]
img_original = imread(sys.argv[1])
img_original_linear = gamma_to_linear(img_original)

print('Analytic ===')
print('Original: {}'.format(np.mean(img_original_linear, axis=(0,1), dtype=np.float64)))
img_out = sim_taichi.simulate_fast(img_original)
print('Analytic: {}'.format(np.mean(img_out, axis=(0,1), dtype=np.float64)))
img_out = sim_taichi.simulate(img_original)
print('Sampled: {}'.format(np.mean(img_out, axis=(0,1), dtype=np.float64)))
