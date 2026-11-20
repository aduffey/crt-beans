// SPDX-License-Identifier: MIT

#include "mask.inc"
#include "line_handling.inc"

vec3 evaluate_scanline(int x, int y, float distance_y, int source_height) {
    if (y < 0 || y >= source_height) {
        return vec3(0.0);
    }

    uvec3 packed_sample = texelFetch(Source, ivec2(x, y), 0).rgb;
    vec2 r = unpackHalf2x16(packed_sample.r);
    vec2 g = unpackHalf2x16(packed_sample.g);
    vec2 b = unpackHalf2x16(packed_sample.b);
    vec3 sample_value = vec3(r.x, g.x, b.x);
    vec3 width_rcp = vec3(r.y, g.y, b.y);

    vec3 dy = clamp(abs(distance_y) * width_rcp, 0.0, 1.0);
    return sample_value * ((dy * dy) * (2.0 * dy - 3.0) + 1.0);
}

vec3 scanlines(lines_t lines, int x, int source_height) {
    return evaluate_scanline(x, lines.upper_sample_y, lines.upper_distance_y, source_height) +
           evaluate_scanline(x, lines.lower_sample_y, lines.lower_distance_y, source_height);
}

vec3 render_scanlines(vec2 vTexCoord, vec2 viewportCoord) {
    int x = clamp(int(floor(vTexCoord.x * params.SourceSize.x)), 0, int(params.SourceSize.x) - 1);
    int source_height = int(params.SourceSize.y);

    vec3 value = vec3(0.0);
    if (params.InterlacingType == 2.0 && params.SourceSize.y < 350.0) {
        // VGA line doubling
        lines_t lines = get_doubled_lines(vTexCoord.y, params.SourceSize);
        value = scanlines(lines, x, source_height);
    } else if (params.InterlacingType >= 2.0 || params.SourceSize.y <= 300.0) {
        // no interlacing
        lines_t lines = get_closest_lines(vTexCoord.y, params.SourceSize);
        value = scanlines(lines, x, source_height);
    } else if (params.InterlacingType == 1.0) {
        // weave
        // Simulate even and odd fields, then average them together
        lines_t lines_even = get_even_lines(vTexCoord.y, params.SourceSize);
        lines_t lines_odd = get_odd_lines(vTexCoord.y, params.SourceSize);
        value = 0.5 * (scanlines(lines_even, x, source_height) + scanlines(lines_odd, x, source_height));
    } else {
        // bob
        uint phase = (params.OddFieldFirst > 0.5) ? 1u : 0u;
        if ((params.FrameCount + phase) % 2u == 0u) {
            // odd field
            lines_t lines = get_odd_lines(vTexCoord.y, params.SourceSize);
            value = scanlines(lines, x, source_height);
        } else {
            // even field
            lines_t lines = get_even_lines(vTexCoord.y, params.SourceSize);
            value = scanlines(lines, x, source_height);
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
