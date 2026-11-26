import numpy as np
from imageio.v3 import imwrite

def generate_test_pattern(width=320, height=240):
    img = np.zeros((height, width, 3), dtype=np.float32)

    # 1. Color bars on top third (lines 0 to 60)
    colors = [
        [1.0, 1.0, 1.0], # White
        [1.0, 1.0, 0.0], # Yellow
        [0.0, 1.0, 1.0], # Cyan
        [0.0, 1.0, 0.0], # Green
        [1.0, 0.0, 1.0], # Magenta
        [1.0, 0.0, 0.0], # Red
        [0.0, 0.0, 1.0], # Blue
        [0.0, 0.0, 0.0], # Black
    ]
    bar_w = width // len(colors)
    for i, col in enumerate(colors):
        x0 = i * bar_w
        x1 = (i + 1) * bar_w if i < len(colors) - 1 else width
        img[0:60, x0:x1] = col

    # 2. Horizontal color stripes (lines 60 to 110)
    # Testing hanging dots at sharp horizontal color boundaries
    stripes = [
        [1.0, 0.0, 0.0], # Red (10 lines)
        [0.0, 1.0, 0.0], # Green (10 lines)
        [0.0, 0.0, 1.0], # Blue (10 lines)
        [1.0, 1.0, 0.0], # Yellow (10 lines)
        [0.0, 0.0, 0.0], # Black (10 lines)
    ]
    for i, col in enumerate(stripes):
        y0 = 60 + i * 10
        y1 = 60 + (i + 1) * 10
        img[y0:y1, 0:160] = col

    # Single-line horizontal stripes (lines 60 to 110 on right side: x=160 to 320)
    # Testing 1-line isolated detail fallback to notch filter
    img[60:110, 160:320] = [0.1, 0.1, 0.1]
    img[65, 170:310] = [1.0, 0.0, 0.0] # 1-line Red
    img[75, 170:310] = [0.0, 1.0, 0.0] # 1-line Green
    img[85, 170:310] = [0.0, 0.0, 1.0] # 1-line Blue
    img[95, 170:310] = [1.0, 1.0, 1.0] # 1-line White

    # 3. Black/White vertical frequency sweeps (lines 110 to 150)
    # 1px alternating vertical lines (highest frequency luma)
    for x in range(0, 80):
        if x % 2 == 0:
            img[110:150, x] = [1.0, 1.0, 1.0]
    # 2px alternating vertical lines
    for x in range(80, 160):
        if (x // 2) % 2 == 0:
            img[110:150, x] = [1.0, 1.0, 1.0]
    # 3px alternating vertical lines
    for x in range(160, 240):
        if (x // 3) % 2 == 0:
            img[110:150, x] = [1.0, 1.0, 1.0]
    # 4px alternating vertical lines
    for x in range(240, 320):
        if (x // 4) % 2 == 0:
            img[110:150, x] = [1.0, 1.0, 1.0]

    # 4. Diagonal lines (lines 150 to 195)
    # Diagonal colored edge and diagonal black/white edges
    for y in range(150, 195):
        for x in range(0, 160):
            # Diagonal color transition (Red above diagonal, Cyan below)
            if (x - 20) > (y - 150) * 3:
                img[y, x] = [1.0, 0.0, 0.0]
            else:
                img[y, x] = [0.0, 1.0, 1.0]
        for x in range(160, 320):
            # Diagonal B/W lines
            if ((x + y) // 3) % 2 == 0:
                img[y, x] = [1.0, 1.0, 1.0]
            else:
                img[y, x] = [0.0, 0.0, 0.0]

    # 5. Checkerboard / Dither patterns (lines 195 to 240)
    # 1x1 checkerboard (standard dithering)
    for y in range(195, 240):
        for x in range(0, 100):
            if (x + y) % 2 == 0:
                img[y, x] = [1.0, 1.0, 1.0]
            else:
                img[y, x] = [0.0, 0.0, 0.0]
        # Colored dither (Blue and Yellow checkerboard)
        for x in range(100, 200):
            if (x + y) % 2 == 0:
                img[y, x] = [1.0, 0.8, 0.0] # Yellow
            else:
                img[y, x] = [0.0, 0.2, 0.9] # Blue
        # Red and Green checkerboard
        for x in range(200, 320):
            if (x + y) % 2 == 0:
                img[y, x] = [0.9, 0.1, 0.1]
            else:
                img[y, x] = [0.1, 0.9, 0.1]

    return img

if __name__ == '__main__':
    pattern = generate_test_pattern()
    imwrite('test-images/test_pattern.png', (pattern * 255).astype(np.uint8))
    print('Generated test-images/test_pattern.png')
