// SPDX-License-Identifier: MIT

#include "mask.inc"

// Antiderivative of k(u) = 1 - 3*u^2 + 2*|u|^3 on [-1, 1].
vec3 vertical_cubic_primitive(vec3 u) {
    return u * (1.0 + u * u * (0.5 * abs(u) - 1.0));
}

// Integrate a scanline's vertical profile over [distance_y - half_pixel_height,
// distance_y + half_pixel_height]. Division by pixel_height is deferred to the
// caller after summing all contributing scanlines.
vec3 evaluate_scanline(int x, int y, float distance_y, float half_pixel_height) {
    uvec3 packed_sample = texelFetch(Source, ivec2(x, y), 0).rgb;
    vec2 r = unpackHalf2x16(packed_sample.r);
    vec2 g = unpackHalf2x16(packed_sample.g);
    vec2 b = unpackHalf2x16(packed_sample.b);
    vec3 sample_value = vec3(r.x, g.x, b.x);
    vec3 width_rcp = vec3(r.y, g.y, b.y);

    vec3 u0 = clamp((distance_y - half_pixel_height) * width_rcp, -1.0, 1.0);
    vec3 u1 = clamp((distance_y + half_pixel_height) * width_rcp, -1.0, 1.0);
    return sample_value * (vertical_cubic_primitive(u1) - vertical_cubic_primitive(u0));
}

vec3 integrate_progressive(int x, float y, float half_pixel_height, float radius, int source_height) {
    int k_start = max(int(floor(y - 0.5 - radius)) + 1, 0);
    int k_stop = min(int(floor(y - 0.5 + radius)), source_height - 1);
    vec3 sum = vec3(0.0);
    for (int k = k_start; k <= k_stop; ++k) {
        float d = (float(k) + 0.5) - y;
        sum += evaluate_scanline(x, k, d, half_pixel_height);
    }
    return sum;
}

vec3 integrate_field(int x, float p, float offset, int row_offset, float half_pixel_height, float radius, int source_height) {
    int k_start = max(int(floor(p - offset - radius)) + 1, 0);
    int k_stop = min(int(floor(p - offset + radius)), (source_height - 1 - row_offset) / 2);
    vec3 sum = vec3(0.0);
    for (int k = k_start; k <= k_stop; ++k) {
        float d = (float(k) + offset) - p;
        sum += evaluate_scanline(x, 2 * k + row_offset, d, half_pixel_height);
    }
    return sum;
}

vec3 integrate_doubled(int x, float y2, float half_pixel_height, float radius, int source_height) {
    int k_start = max(int(floor(y2 - 0.5 - radius)) + 1, 0);
    int k_stop = min(int(floor(y2 - 0.5 + radius)), 2 * source_height - 1);
    vec3 sum = vec3(0.0);
    for (int k = k_start; k <= k_stop; ++k) {
        float d = (float(k) + 0.5) - y2;
        sum += evaluate_scanline(x, k >> 1, d, half_pixel_height);
    }
    return sum;
}

vec3 render_scanlines(vec2 vTexCoord, vec2 viewportCoord) {
    int x = clamp(int(floor(vTexCoord.x * params.SourceSize.x)), 0, int(params.SourceSize.x) - 1);
    int source_height = int(params.SourceSize.y);
    float y = vTexCoord.y * params.SourceSize.y;
    float src_pixel_height = (1.0 - params.OverscanVertical) * params.SourceSize.y * params.OutputSize.w;

    vec3 value = vec3(0.0);
    if (params.InterlacingType == 2.0 && params.SourceSize.y < 350.0) {
        // VGA line doubling
        float pixel_height = 2.0 * src_pixel_height;
        float half_h = 0.5 * pixel_height;
        float radius = params.MaxSpotSize + half_h;
        value = integrate_doubled(x, 2.0 * y, half_h, radius, source_height) / pixel_height;
    } else if (params.InterlacingType >= 2.0 || params.SourceSize.y <= 300.0) {
        // no interlacing
        float pixel_height = src_pixel_height;
        float half_h = 0.5 * pixel_height;
        float radius = params.MaxSpotSize + half_h;
        value = integrate_progressive(x, y, half_h, radius, source_height) / pixel_height;
    } else if (params.InterlacingType == 1.0) {
        // weave
        // Simulate even and odd fields, then average them together
        float pixel_height = 0.5 * src_pixel_height;
        float half_h = 0.5 * pixel_height;
        float radius = params.MaxSpotSize + half_h;
        float p = 0.5 * y;
        value = (integrate_field(x, p, 0.75, 1, half_h, radius, source_height) +
                 integrate_field(x, p, 0.25, 0, half_h, radius, source_height)) / src_pixel_height;
    } else {
        // bob
        float pixel_height = 0.5 * src_pixel_height;
        float half_h = 0.5 * pixel_height;
        float radius = params.MaxSpotSize + half_h;
        float p = 0.5 * y;
        uint phase = (params.OddFieldFirst > 0.5) ? 1u : 0u;
        if ((params.FrameCount + phase) % 2u == 0u) {
            // odd field
            value = integrate_field(x, p, 0.25, 0, half_h, radius, source_height) / pixel_height;
        } else {
            // even field
            value = integrate_field(x, p, 0.75, 1, half_h, radius, source_height) / pixel_height;
        }
    }

    if (params.MaskType == 1.0) {
        vec4 mask = subpixel_aperture_grille(viewportCoord, params.OutputSize, params.SubpixelPattern, params.SubpixelMaskPattern);
        value = blend_mask(mask.rgb, value, mask.a, params.MaskMitigation);
    } else if (params.MaskType == 2.0) {
        vec4 mask = dynamic_aperture_grille(viewportCoord, params.OutputSize, params.DynamicMaskTriads, params.MaskSmoothness);
        value = blend_mask(mask.rgb, value, mask.a, params.MaskMitigation);
    } else if (params.MaskType == 3.0) {
        value = apply_dynamic2_mask(value, viewportCoord, params.OutputSize, params.DynamicMaskTriads, params.MaskSmoothness, params.MaskMitigation);
    } else if (params.MaskType == 4.0) {
        vec4 mask = monochrome_mask(viewportCoord, params.OutputSize, params.SubpixelMaskPattern);
        value = blend_mask(mask.rgb, value, mask.a, params.MaskMitigation);
    } // Otherwise, no mask

    return value;
}
