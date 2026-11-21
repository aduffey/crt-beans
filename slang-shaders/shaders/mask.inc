// SPDX-License-Identifier: MIT

#ifndef PI
#define PI 3.141592653589793
#endif

// This functions generates a static, subpixel-aware mask.
//
// Note this approach limits phosphor triad counts to integer divisors of the
// screen width in pixels.
//
// Parameters:
//     coord: The current uv coordinates on the screen
//     output_size: <x_size, y_size, 1 / x_size, 1 / y_size>
//         (equivalent to the Retroarch-provided OutputSize uniform)
//     subpixel_pattern: The subpixel arrangement of the monitor.
//         0.0: RGB
//         1.0: BGR
//     pixels_per_triad: The number of monitor pixels used to simulate one
//         phosphor triad. 2.0, 3.0, 4.0, and 5.0 are currently supported.
//         Anything else will result in a mask of all 1's (i.e. no mask).
//
// Returns:
//     A vec4 with the rgb components containing the rgb values to apply the
//     mask on the current pixel, and the a component containing the mask
//     coverage (the inverse of the proportion of active subpixels).
vec4 subpixel_aperture_grille(vec2 coord, vec4 output_size, float subpixel_pattern, float pixels_per_triad) {
    // We need the coordinate in viewport pixels.
    float x = coord.x * output_size.x;
    vec3 mask = vec3(1.0);
    float mask_coverage = 1.0;
    int triad_pixels = int(round(pixels_per_triad));
    if (subpixel_pattern < 0.5) { // RGB subpixels
        if (triad_pixels == 2) { // <=1080p
            vec3[2] mask_tile = {
                    vec3(1.0, 0.0, 1.0),
                    vec3(0.0, 1.0, 0.0)
                };
            mask = mask_tile[int(mod(x, 2.0))];
            mask_coverage = 2.0;
        } else if (triad_pixels == 3) { // 1080p/1440p
            vec3[3] mask_tile = {
                    vec3(0.0, 0.0, 1.0),
                    vec3(0.0, 1.0, 0.0),
                    vec3(1.0, 0.0, 0.0)
                };
            mask = mask_tile[int(mod(x, 3.0))];
            mask_coverage = 3.0;
        } else if (triad_pixels == 4) { // 1440p/4k
            vec3[4] mask_tile = {
                    vec3(1.0, 0.0, 0.0),
                    vec3(1.0, 1.0, 0.0),
                    vec3(0.0, 1.0, 1.0),
                    vec3(0.0, 0.0, 1.0)
                };
            mask = mask_tile[int(mod(x, 4.0))];
            mask_coverage = 2.0;
        } else if (triad_pixels == 5) { // 4k, lower TVL
            vec3[5] mask_tile = {
                    vec3(1.0, 0.0, 0.0),
                    vec3(1.0, 0.0, 1.0),
                    vec3(0.0, 0.0, 1.0),
                    vec3(0.0, 1.0, 0.0),
                    vec3(0.0, 1.0, 0.0)
                };
            mask = mask_tile[int(mod(x, 5.0))];
            mask_coverage = 15.0 / 6.0;
        }
    } else { // BGR subpixels, same as above but red and blue values transposed
        if (triad_pixels == 2) { // <=1080p
            vec3[2] mask_tile = {
                    vec3(1.0, 0.0, 1.0),
                    vec3(0.0, 1.0, 0.0)
                };
            mask = mask_tile[int(mod(x, 2.0))];
            mask_coverage = 2.0;
        } else if (triad_pixels == 3) { // 1080p/1440p
            vec3[3] mask_tile = {
                    vec3(1.0, 0.0, 0.0),
                    vec3(0.0, 1.0, 0.0),
                    vec3(0.0, 0.0, 1.0)
                };
            mask = mask_tile[int(mod(x, 3.0))];
            mask_coverage = 3.0;
        } else if (triad_pixels == 4) { // 1440p/4k
            vec3[4] mask_tile = {
                    vec3(0.0, 0.0, 1.0),
                    vec3(0.0, 1.0, 1.0),
                    vec3(1.0, 1.0, 0.0),
                    vec3(1.0, 0.0, 0.0)
                };
            mask = mask_tile[int(mod(x, 4.0))];
            mask_coverage = 2.0;
        } else if (triad_pixels == 5) { // 4k, lower TVL
            vec3[5] mask_tile = {
                    vec3(0.0, 0.0, 1.0),
                    vec3(1.0, 0.0, 1.0),
                    vec3(1.0, 0.0, 0.0),
                    vec3(0.0, 1.0, 0.0),
                    vec3(0.0, 1.0, 0.0)
                };
            mask = mask_tile[int(mod(x, 5.0))];
            mask_coverage = 15.0 / 6.0;
        }
    }

    return vec4(mask, mask_coverage);
}

// This function generates an aperture grille mask for the current coordinates.
//
// It works by scaling and low pass filtering a simple RGB aperture grille and
// sampling at the output pixel center. The low pass kernel is a raised cosine:
//
//     (0.5 + 0.5 * cos(pi * x / radius)) / radius
//
// The output is the convolution of the mask and the low pass kernel, if the
// mask is considered to be a piecewise-constant function of R, G, and B.
// Equivalently, the mask could be considered to be a series of dirac delta
// functions and the result is the convolution of the mask, a box function (to
// give the mask width), and the low pass kernel.
//
// For a more in-depth description of a similar thing, see:
//
//     https://themaister.net/blog/2018/08/25/pseudo-bandlimited-pixel-art-filtering-in-3d-a-mathematical-derivation/
//
// Parameters:
//     coord: The current uv coordinates on the screen
//     output_size: <x_size, y_size, 1 / x_size, 1 / y_size>
//         (equivalent to the Retroarch-provided OutputSize uniform)
//     mask_triads: The number of phosphor triads to generate across the screen
//     smoothness: Scales the kernel width from 1x (0) to 2x (1).
//
// Returns:
//     A vec4 with the rgb components containing the rgb values to apply the
//     mask on the current pixel, and the a component containing the mask
//     coverage (the inverse of the integral of the mask). In this case, the
//     mask coverage is always 3.0.
vec4 dynamic_aperture_grille(vec2 coord, vec4 output_size, float mask_triads, float smoothness) {
    // Loops are unrolled and common terms collapsed for performance.
    const vec3[3] phosphors = {
        vec3(1.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        vec3(0.0, 0.0, 1.0)
    };
    // All units are in terms of phosphors (i.e. 1 == phosphor width).
    // Set the kernel radius to one to two output pixels.
    float kernel_radius = 3.0 * mask_triads * output_size.z * (1.0 + smoothness);
    float x = 3.0 * mask_triads * coord.x;
    vec3 mask = vec3(0.0);
    if (kernel_radius < 0.5) {
        float x1 = clamp((x - round(x)) / kernel_radius, -1.0, 1.0);
        // mod(x, 3.0) isn't exact in floating point, so we'll add a small
        // value. This might not make a noticeable difference.
        vec3 p0 = phosphors[int(mod(round(x) - 1.0, 3.0) + 1e-3)];
        vec3 p1 = p0.brg;
        mask = p0 + p1 + (p1 - p0) * (x1 + sin(PI * x1) / PI);
        mask *= 0.5;
    }
    else if (kernel_radius < 1.0) {
        float x1 = clamp((x - (floor(x) + 0.0)) / kernel_radius, -1.0, 1.0);
        float x2 = clamp((x - (floor(x) + 1.0)) / kernel_radius, -1.0, 1.0);
        vec3 p0 = phosphors[int(mod(floor(x) - 1.0, 3.0) + 1e-3)];
        vec3 p1 = p0.brg;
        vec3 p2 = p1.brg;
        mask = p0 + p2 +
                (p1 - p0) * (x1 + sin(PI * x1) / PI) +
                (p2 - p1) * (x2 + sin(PI * x2) / PI);
        mask *= 0.5;
    }
    else if (kernel_radius < 1.5) {
        float x1 = clamp((x - (round(x) - 1.0)) / kernel_radius, -1.0, 1.0);
        float x2 = clamp((x - (round(x) + 0.0)) / kernel_radius, -1.0, 1.0);
        float x3 = clamp((x - (round(x) + 1.0)) / kernel_radius, -1.0, 1.0);
        vec3 p0 = phosphors[int(mod(round(x) - 2.0, 3.0) + 1e-3)];
        vec3 p1 = p0.brg;
        vec3 p2 = p1.brg;
        vec3 p3 = p2.brg;
        mask = p0 + p3 +
                (p1 - p0) * (x1 + sin(PI * x1) / PI) +
                (p2 - p1) * (x2 + sin(PI * x2) / PI) +
                (p3 - p2) * (x3 + sin(PI * x3) / PI);
        mask *= 0.5;
    } else {
        // Wider kernels can span more than three phosphor edges. Integrate
        // every transition inside the support of the raised cosine.
        float start = floor(x - kernel_radius);
        vec3 p0 = phosphors[int(mod(start, 3.0) + 1e-3)];
        mask = p0;
        for (float edge = start + 1.0; edge < x + kernel_radius; edge += 1.0) {
            vec3 p1 = p0.brg;
            float u = (x - edge) / kernel_radius;
            mask += (p1 - p0) * (0.5 + 0.5 * (u + sin(PI * u) / PI));
            p0 = p1;
        }
    }

    return vec4(mask, 3.0);
}

// Harmonic aperture grille that phases in the first and second harmonics via
// smoothstep based on pixels per triad, blends per-channel by blooming bright
// phosphors (shedding the second harmonic before the first), and applies a
// DC-preserving Gaussian smoothness filter.
vec3 apply_dynamic2_mask(
    vec3 pixel_value,
    vec2 coord,
    vec4 output_size,
    float mask_triads,
    float smoothness,
    float mitigation
) {
    float p = output_size.x / mask_triads;
    float t1 = smoothstep(2.0, 3.0, p);
    float t2 = smoothstep(4.0, 5.0, p);
    float dc0 = 1.0 - 0.5 * t1 - 0.125 * t2;
    float dc_base = 0.5 - 0.125 * t2;
    float dc_knee = mix(dc0, dc_base, mitigation);
    float cap_denom = mix(t1, 1.0, mitigation);
    float cap_scale = (cap_denom > 0.0) ? (t1 / cap_denom) : 0.0;

    // Align R, G, B phosphor centers at 1/6, 3/6, 5/6 of each triad to match
    // dynamic_aperture_grille and pixel centers when p is an integer.
    float phase = 2.0 * PI * mask_triads * coord.x - PI / 3.0;
    float cs = cos(phase);
    float sn = sin(phase);
    vec3 c1 = vec3(cs, -0.5 * cs + 0.86602540378 * sn, -0.5 * cs - 0.86602540378 * sn);
    vec3 c2 = 2.0 * c1 * c1 - 1.0;

    vec3 v = pixel_value * mix(dc0, 1.0, mitigation);

    // Scale linearly up to dc_knee (sliding from dc0 at mitigation=0 for pure
    // linear masking to dc_base at mitigation=1 for uniform post-bloom t1
    // attenuation), then bloom (shed 2nd harmonic first, then 1st) while
    // keeping peaks <= 1 and spatial mean == v.
    vec3 v_scaled = v / dc_knee;
    vec3 a1 = max(min(0.5 * t1 * min(v_scaled, vec3(1.0)), cap_scale * (1.0 - v)), vec3(0.0));
    vec3 a2 = max(min(0.125 * t2 * v_scaled, 0.5 - v), vec3(0.0));

    // Apply DC-preserving Gaussian smoothness around mean v: H(n) = (1 - s)^(n^2).
    float u = 1.0 - smoothness;
    float u2 = u * u;
    return v + (a1 * u) * c1 + (a2 * (u2 * u2)) * c2;
}

// Static 2-5 pixel repeats: WB, WWB, WWBB, WWWBB, independent of subpixels.
// Split each repeat evenly, assigning an odd repeat's extra pixel to white.
vec4 monochrome_mask(vec2 coord, vec4 output_size, float pixels_per_repeat) {
    int period = int(round(pixels_per_repeat));
    if (period < 2) {
        return vec4(1.0);
    }
    int white_pixels = (period + 1) / 2;
    // Sample pixel centers so floating-point mod cannot misclassify wrap edges.
    float x = floor(coord.x * output_size.x) + 0.5;
    float value = mod(x, float(period)) < float(white_pixels) ? 1.0 : 0.0;
    return vec4(vec3(value), float(period) / float(white_pixels));
}

// Blend the mask with the current pixel value and return the result.
//
// The blending is done by keeping the mask as strong as possible and fading it
// out only as necessary to maintain the overall brightness of the image.
//
// Mathematically, the blending is represented by:
//
//     f(v) = c * a * m(x) + b
//
// where:
//
//     * m(x) is the mask function (for example, a pulse wave with a duty
//       cycle of 1/3 for an aperture grille)
//     * c is the mask coverage (the inverse of the integral of m(x) over x
//       from 0 to 1)
//     * v is the pixel value
//
// We can solve for a and b in terms of c and v given the following
// constraints:
//
//     * 0 <= m(x) <= 1
//     * 0 <= f(v) <= 1
//     * The integral of f(v) over x from 0 to 1 = v
//
// (The third constraint is responsible for maintaining brightness.)
//
// Parameters:
//     mask: The mask to be applied to this pixel
//     pixel_value: The rgb value of this pixel (before masking)
//     mask_coverage: The inverse of the integral of the mask
//     mitigation: The level of mask brightness loss mitigation to apply
//         (ranging from 0.0 to 1.0)
//
// Returns:
//     A vec3 with the rgb value of this pixel after applying the mask.
vec3 blend_mask(vec3 mask, vec3 pixel_value, float mask_coverage, float mitigation) {
    // Avoid divisions by zero and return early if we don't need to blend.
    if (mask_coverage <= 1.0 + 1.0e-5) {
        return pixel_value;
    }

    // Adjust the pixel value based on the mask strength. We want the strength
    // to range from 1 / mask_coverage (which will result in a pure mask with
    // no brightness loss mitigation) to 1.0 (which will result in full
    // brightness loss mitigation).
    vec3 v = pixel_value * mix(1.0 / mask_coverage, 1.0, mitigation);

    vec3 a = min((v - 1.0) / (1.0 - mask_coverage), v);
    vec3 b = max((1.0 - mask_coverage * v) / (1.0 - mask_coverage), vec3(0.0));
    return mask_coverage * a * mask + b;
}
