import sys

import numpy as np
from config import SimulationConfig
import sim_taichi
from imageio.v3 import imread


def gamma_to_linear(img):
    """Simple gamma uint8 to linear float"""
    return np.power(np.clip(img, 0.0, 1.0), 2.4)


filename = sys.argv[1]
img_original = imread(sys.argv[1]).astype(np.float32) / 255
img_original_linear = gamma_to_linear(img_original)

print('Original: {}'.format(np.mean(img_original_linear, axis=(0,1), dtype=np.float64)))

cfg_analytical = SimulationConfig(scanline_method='analytical', enable_analog=False, enable_masking=False, enable_diffusion=False)
img_out = sim_taichi.simulate(img_original, config=cfg_analytical)
print('Analytic: {}'.format(np.mean(img_out, axis=(0,1), dtype=np.float64)))

cfg_fast = SimulationConfig(scanline_method='fast', enable_analog=False, enable_masking=False, enable_diffusion=False)
img_out = sim_taichi.simulate(img_original, config=cfg_fast)
print('Fast: {}'.format(np.mean(img_out, axis=(0,1), dtype=np.float64)))

cfg_sampled = SimulationConfig(scanline_method='sampled', enable_analog=False, enable_masking=False, enable_diffusion=False)
img_out = sim_taichi.simulate(img_original, config=cfg_sampled)
print('Sampled: {}'.format(np.mean(img_out, axis=(0,1), dtype=np.float64)))
