// SPDX-License-Identifier: MIT

// The trap spans three carrier periods on either side of the source pixel.
// Its triangle-windowed carrier basis also matches the luma response slope,
// preventing the full-depth notch from crossing through zero at the carrier.
const float LUMA_TRAP_OMEGA = 6.0 * PI;

// sin(x)/x and (sin(x) - x*cos(x))/x^2, with stable limits at zero.
vec2 luma_trap_sinc_j1(float x) {
    float x2 = x * x;
    if (x2 < 0.25) {
        float sinc = 1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0
                   + x2 * (-1.0 / 5040.0 + x2 / 362880.0)));
        float j1 = x * (1.0 / 3.0 + x2 * (-1.0 / 30.0
                 + x2 * (1.0 / 840.0 + x2 * (-1.0 / 45360.0
                 + x2 / 3991680.0))));
        return vec2(sinc, j1);
    }
    float s = sin(x);
    return vec2(s / x, (s - x * cos(x)) / x2);
}

// l is the luma Hann half-width divided by the trap half-width.
// This parameter-only work is evaluated in the vertex stage. The returned
// primitive coefficients include the depth and low_pass()'s 2*PI scale.
vec3 luma_trap_coefficients(float l, float depth_db) {
    vec3 z = LUMA_TRAP_OMEGA * l + vec3(0.0, -PI, PI);
    vec2 f0 = luma_trap_sinc_j1(z.x);
    vec2 fm = luma_trap_sinc_j1(z.y);
    vec2 fp = luma_trap_sinc_j1(z.z);
    float amplitude = f0.x + 0.5 * (fm.x + fp.x);
    float slope = -l * (f0.y + 0.5 * (fm.y + fp.y));
    float correction = slope / (-1.0 / 12.0
                       - 1.0 / (4.0 * LUMA_TRAP_OMEGA * LUMA_TRAP_OMEGA));
    vec3 coefficients = vec3(
        2.0 * (amplitude + correction / LUMA_TRAP_OMEGA) / LUMA_TRAP_OMEGA,
        correction / (LUMA_TRAP_OMEGA * LUMA_TRAP_OMEGA),
        correction / LUMA_TRAP_OMEGA
    );
    float strength = 1.0 - pow(10.0, -clamp(depth_db, 0.0, 32.0) / 20.0);
    return (2.0 * PI * strength) * coefficients;
}

// Integral of the direct trap correction. x is the source-boundary offset
// divided by the trap half-width; no absolute carrier phase is involved.
float luma_trap_primitive(float x, vec3 coefficients) {
    float u = 1.0 - abs(x);
    if (u <= 0.0)
        return 0.0;

    float s = sin(LUMA_TRAP_OMEGA * x);
    float c = cos(LUMA_TRAP_OMEGA * x);
    return (coefficients.x * u - coefficients.y) * s
         - coefficients.z * x * u * c
         + coefficients.x * sign(x) / LUMA_TRAP_OMEGA * (1.0 - c);
}
