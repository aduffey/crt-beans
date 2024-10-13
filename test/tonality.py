import numpy as np
import sim_taichi
from imageio.v3 import imwrite, imread
import matplotlib.pyplot as plt


def linear_to_srgb(img):
    """Linear float to sRGB uint8"""
    img_clipped = np.clip(img, 0.0, 1.0)
    out = np.where(img_clipped <= 0.0031308, img_clipped * 12.92, 1.055 * (np.power(img_clipped, (1.0 / 2.4))) - 0.055)
    out = np.around(out * 255).astype(np.uint8)
    return out

values = list(reversed([1.0 / (2**x) for x in range(10)]))
simulated_means = []
for val in values:
    img = np.around(np.full((240, 320, 3), val**(1/2.4) * 255)).astype(np.uint8)
    out = sim_taichi.simulate(img)
    out = np.clip(out, 0.0, 1.0)
    # print(val**(1/2.4))  # DEBUG
    # imwrite(f'debug-in-{val}.png', linear_to_srgb(img))  # DEBUG
    # imwrite(f'debug-{val}.png', linear_to_srgb(out))  # DEBUG
    # print(out)  # DEBUG
    simulated_means.append(np.mean(out))

fig, ax = plt.subplots()
ax.plot(values, values)
ax.plot(values, simulated_means, 'o-')
ax.set_xscale('log', base=2)
ax.set_yscale('log', base=2)
plt.show()
