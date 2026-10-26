// SPDX-License-Identifier: MIT

#include "mask.inc"
#include "line_handling.inc"

vec3 scanlines(lines_t lines, float u) {
    uvec3 upper_packed = textureLod(Source, vec2(u, lines.upper_sample_y), 0.0).rgb;
    vec2 upper_r = unpackHalf2x16(upper_packed.r);
    vec2 upper_g = unpackHalf2x16(upper_packed.g);
    vec2 upper_b = unpackHalf2x16(upper_packed.b);
    vec3 upper_sample = vec3(upper_r.x, upper_g.x, upper_b.x);
    vec3 upper_width_rcp = vec3(upper_r.y, upper_g.y, upper_b.y);

    vec3 upper_y = clamp(abs(lines.upper_distance_y) * upper_width_rcp, 0.0, 1.0);
    vec3 upper_value = upper_sample * upper_width_rcp * ((upper_y * upper_y) * (2.0 * upper_y - 3.0) + 1.0);

    uvec3 lower_packed = textureLod(Source, vec2(u, lines.lower_sample_y), 0.0).rgb;
    vec2 lower_r = unpackHalf2x16(lower_packed.r);
    vec2 lower_g = unpackHalf2x16(lower_packed.g);
    vec2 lower_b = unpackHalf2x16(lower_packed.b);
    vec3 lower_sample = vec3(lower_r.x, lower_g.x, lower_b.x);
    vec3 lower_width_rcp = vec3(lower_r.y, lower_g.y, lower_b.y);

    vec3 lower_y = clamp(abs(lines.lower_distance_y) * lower_width_rcp, 0.0, 1.0);
    vec3 lower_value = lower_sample * lower_width_rcp * ((lower_y * lower_y) * (2.0 * lower_y - 3.0) + 1.0);

    return params.MaxSpotSize * (upper_value + lower_value);
}

vec3 render_scanlines(vec2 vTexCoord, vec2 viewportCoord) {
    vec3 value = vec3(0.0);
    if (params.InterlacingType == 2.0 && params.SourceSize.y < 350) {
        // VGA line doubling
        lines_t lines = get_doubled_lines(vTexCoord.y, params.SourceSize);
        value = scanlines(lines, vTexCoord.x);
    } else if (params.InterlacingType >= 2.0 || params.SourceSize.y <= 300.0) {
        // no interlacing
        lines_t lines = get_closest_lines(vTexCoord.y, params.SourceSize);
        value = scanlines(lines, vTexCoord.x);
    } else if (params.InterlacingType == 1.0) {
        // weave
        // Simulate even and odd fields, then average them together
        lines_t lines_even = get_even_lines(vTexCoord.y, params.SourceSize);
        lines_t lines_odd = get_odd_lines(vTexCoord.y, params.SourceSize);
        value = 0.5 * (scanlines(lines_even, vTexCoord.x) + scanlines(lines_odd, vTexCoord.x));
    } else {
        // bob
        uint phase = (params.OddFieldFirst > 0.5) ? 1u : 0u;
        if ((params.FrameCount + phase) % 2u == 0u) {
            // odd field
            lines_t lines = get_odd_lines(vTexCoord.y, params.SourceSize);
            value = scanlines(lines, vTexCoord.x);
        } else {
            // even field
            lines_t lines = get_even_lines(vTexCoord.y, params.SourceSize);
            value = scanlines(lines, vTexCoord.x);
        }
    }

    if (params.MaskType == 1.0) {
        vec4 mask = subpixel_aperture_grille(viewportCoord, params.OutputSize, params.SubpixelPattern, params.SubpixelMaskPattern);
        value = blend_mask(mask.rgb, value, mask.a, params.MaskMitigation);
    } else if (params.MaskType == 2.0) {
        vec4 mask = dynamic_aperture_grille(viewportCoord, params.OutputSize, params.DynamicMaskTriads, params.MaskSmoothness);
        value = blend_mask(mask.rgb, value, mask.a, params.MaskMitigation);
    } else if (params.MaskType == 3.0) {
        vec4 mask = dynamic2_aperture_grille(viewportCoord, params.OutputSize, params.DynamicMaskTriads, params.MaskSmoothness);
        value = blend_mask(mask.rgb, value, mask.a, params.MaskMitigation);
    } else if (params.MaskType == 4.0) {
        vec4 mask = monochrome_mask(viewportCoord, params.OutputSize, params.MaskSmoothness);
        value = blend_mask(mask.rgb, value, mask.a, params.MaskMitigation);
    } // Otherwise, no mask

    return value;
}
