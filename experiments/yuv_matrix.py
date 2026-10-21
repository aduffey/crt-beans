import numpy as np

# -------------------------------------------------------------------------
# Step 1: Define canonical BT.601 / SMPTE 170M constants in 64-bit float
# -------------------------------------------------------------------------
# Luma weighting coefficients: Y' = Wr * R' + Wg * G' + Wb * B'
Wr = np.float64(0.299)
Wg = np.float64(0.587)
Wb = np.float64(0.114)

# Excursion scaling factors for NTSC/PAL composite encoding:
# U = W_scale_B * (B' - Y')
# V = W_scale_R * (R' - Y')
W_scale_B = 0.492111
W_scale_R = 0.877283

print(W_scale_B, W_scale_R)

# -------------------------------------------------------------------------
# Step 2: Assemble the RGB -> YUV transformation matrix (float64)
# -------------------------------------------------------------------------
# Row 0 (Y):  [ Wr,                         Wg,                         Wb ]
# Row 1 (U):  [ -Wr * W_scale_B,            -Wg * W_scale_B,            (1 - Wb) * W_scale_B ]
# Row 2 (V):  [ (1 - Wr) * W_scale_R,       -Wg * W_scale_R,            -Wb * W_scale_R ]

M_rgb_to_yuv_f64 = np.array([
    [Wr, Wg, Wb],
    [-Wr * W_scale_B, -Wg * W_scale_B, (np.float64(1.0) - Wb) * W_scale_B],
    [(np.float64(1.0) - Wr) * W_scale_R, -Wg * W_scale_R, -Wb * W_scale_R]
], dtype=np.float64)

# -------------------------------------------------------------------------
# Step 3: Derive the Inverse Matrix (YUV -> RGB) in float64
# -------------------------------------------------------------------------
M_yuv_to_rgb_f64 = np.linalg.inv(M_rgb_to_yuv_f64)

# -------------------------------------------------------------------------
# Output formatting
# -------------------------------------------------------------------------
np.set_printoptions(floatmode='unique', suppress=False)

print("=== RGB to YUV Matrix (float64 derivation) ===")
print(M_rgb_to_yuv_f64)

print("\n=== YUV to RGB Inverse Matrix (float64 inversion) ===")
print(M_yuv_to_rgb_f64)

# GLSL mat3 column-major ready format
print("\n=== GLSL Column-Major mat3 Definitions (float32) ===")
def to_glsl_mat3(name, mat):
    cols = [f"vec3({mat[0, i]:.8f}, {mat[1, i]:.8f}, {mat[2, i]:.8f})" for i in range(3)]
    return f"const mat3 {name} = mat3(\n    " + ",\n    ".join(cols) + "\n);"

print(to_glsl_mat3("RGB_TO_YUV", M_rgb_to_yuv_f64))
print()
print(to_glsl_mat3("YUV_TO_RGB", M_yuv_to_rgb_f64))
