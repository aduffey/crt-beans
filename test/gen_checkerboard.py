from imageio.v3 import imwrite, imread
import numpy as np


tile = np.array([[[255, 255, 255], [0, 0, 0]],
                 [[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
imwrite('checkerboard_pixel.png', np.tile(tile, (120, 160, 1)))
tile = np.array([[[255, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
                 [[0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 255, 255]]], dtype=np.uint8)
imwrite('checkerboard_double.png', np.tile(tile, (120, 160, 1)))
