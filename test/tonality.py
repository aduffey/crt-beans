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
    img = np.full((240, 256, 3), val**(1/2.4)).astype(np.float32)
    out = sim_taichi.simulate_analytical(img)
    # out = np.clip(out, 0.0, 1.0)
    # print(val**(1/2.4))  # DEBUG
    # imwrite(f'debug-in-{val}.png', linear_to_srgb(img))  # DEBUG
    # imwrite(f'debug-{val}.png', linear_to_srgb(out))  # DEBUG
    # print(out)  # DEBUG
    print(np.mean(out, dtype=np.float64))
    print(np.mean(out, axis=(0,1), dtype=np.float64))
    simulated_means.append(np.mean(out, axis=(0,1)))

fig, ax = plt.subplots()
ax.plot(values, values)
ax.plot(values, [r for (r, g, b) in simulated_means], 'ro-')
ax.plot(values, [g for (r, g, b) in simulated_means], 'go-')
ax.plot(values, [b for (r, g, b) in simulated_means], 'bo-')
ax.set_xscale('log', base=2)
ax.set_yscale('log', base=2)
plt.show()
