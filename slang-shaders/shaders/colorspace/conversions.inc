const float CRT_GAMMA = 2.4;

// These matrices were produced by the yuv_matrix.py script.
// They include the scaling factors already, so the resulting values will be
// ready for modulation.
const mat3 RGB_TO_YUV = mat3(
    vec3(0.29900000, -0.14714119, 0.61497538),
    vec3(0.58700000, -0.28886916, -0.51496512),
    vec3(0.11400000, 0.43601035, -0.10001026)
);

const mat3 YUV_TO_RGB = mat3(
    vec3(1.00000000, 1.00000000, 1.00000000),
    vec3(0.00000000, -0.39464234, 2.03206187),
    vec3(1.13988303, -0.58062185, 0.00000000)
);

vec3 linear_to_srgb(vec3 rgb) {
    // Floating point inaccuracies could result in negative values, so we'll
    // clamp here to avoid NaNs.
    rgb = clamp(rgb, 0.0, 1.0);
    bvec3 cutoff = lessThan(rgb, vec3(0.0031308));
    vec3 lower = rgb * vec3(12.92);
    vec3 higher = vec3(1.055) * pow(rgb, vec3(1.0 / 2.4)) - vec3(0.055);

    return mix(higher, lower, cutoff);
}

vec3 srgb_to_linear(vec3 rgb) {
    // Floating point inaccuracies could result in negative values, so we'll
    // clamp here to avoid NaNs.
    rgb = clamp(rgb, 0.0, 1.0);
    bvec3 cutoff = lessThan(rgb, vec3(0.04045));
    vec3 lower = rgb / 12.92;
    vec3 higher = pow((rgb + 0.055) / 1.055, vec3(2.4));

    return mix(higher, lower, cutoff);
}

vec3 linear_to_gamma(vec3 rgb, float gamma) {
    // Floating point inaccuracies could result in negative values, so we'll
    // clamp here to avoid NaNs.
    rgb = clamp(rgb, 0.0, 1.0);
    return pow(rgb, vec3(1.0 / gamma));
}

vec3 gamma_to_linear(vec3 rgb, float gamma) {
    // Floating point inaccuracies could result in negative values, so we'll
    // clamp here to avoid NaNs.
    rgb = clamp(rgb, 0.0, 1.0);
    return pow(rgb, vec3(gamma));
}

vec3 gamma_to_yuv(vec3 rgb) {
    return RGB_TO_YUV * rgb;
}

vec3 yuv_to_linear(vec3 rgb) {
    return gamma_to_linear(YUV_TO_RGB * rgb, CRT_GAMMA);
}
