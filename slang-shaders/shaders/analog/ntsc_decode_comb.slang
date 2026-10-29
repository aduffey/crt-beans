#version 450

// SPDX-License-Identifier: MIT

layout(set = 0, binding = 0, std140) uniform UBO {
    mat4 MVP;
};

layout(push_constant) uniform Push {
    vec4 SourceSize;
    vec4 OriginalSize;
    float FilterType;
    float System;
    uint FrameCount;
    float InterlacingType;
    float OddFieldFirst;
} params;

#pragma format R16G16B16A16_SFLOAT

#include "parameters_ntsc.inc"
#include "chroma_carrier.inc"

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
layout(location = 1) flat out vec4 vWeightsUpper;
layout(location = 2) flat out vec4 vWeightsLower;

vec4 carrier_exact_4tap(float offset_x, float source_w, float wc) {
    float delta_texels = offset_x * source_w;
    float d = delta_texels - floor(delta_texels);
    float u = d - 0.5;
    float c1 = cos(0.5 * wc), c3 = cos(1.5 * wc), cu = cos(u * wc);
    float s1 = sin(0.5 * wc), s3 = sin(1.5 * wc), su = sin(u * wc);

    float e3 = 0.5 * (cu - c1) / (c3 - c1);
    float e1 = 0.5 - e3;
    float o3 = (0.5 * su - u * s1) / (s3 - 3.0 * s1);
    float o1 = u - 3.0 * o3;

    return vec4(e3 - o3, e1 - o1, e1 + o1, e3 + o3);
}

void main() {
    gl_Position = MVP * Position;
    vTexCoord = TexCoord;

    chroma_info ci = get_chroma_info(params.System, params.OriginalSize, params.FrameCount);
    float offset = (227.5 - ci.periods_per_line) / ci.periods_per_active_line;
    float wc = 2.0 * PI * (ci.periods_per_active_line * params.SourceSize.z);

    if (params.FilterType > 0.5 && offset != 0.0) {
        vWeightsUpper = carrier_exact_4tap(-offset, params.SourceSize.x, wc);
        vWeightsLower = carrier_exact_4tap( offset, params.SourceSize.x, wc);
    } else {
        vWeightsUpper = vec4(0.0);
        vWeightsLower = vec4(0.0);
    }
}

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 1) flat in vec4 vWeightsUpper;
layout(location = 2) flat in vec4 vWeightsLower;
layout(location = 0) out vec4 FragColor;
layout(binding = 1) uniform sampler2D Source;

#include "../colorspace/conversions.inc"

vec3 interpolate4(vec2 center_coord, vec4 w) {
    vec3 result = w.x * textureLodOffset(Source, center_coord, 0.0, ivec2(-1, 0)).gba;
    result += w.y * textureLodOffset(Source, center_coord, 0.0, ivec2( 0, 0)).gba;
    result += w.z * textureLodOffset(Source, center_coord, 0.0, ivec2( 1, 0)).gba;
    result += w.w * textureLodOffset(Source, center_coord, 0.0, ivec2( 2, 0)).gba;
    return result;
}

void main() {
    ivec2 texel_loc = ivec2(floor(vTexCoord * params.SourceSize.xy));
    chroma_info ci = get_chroma_info(params.System, params.OriginalSize, params.FrameCount);
    bool interlaced = is_interlaced(params.InterlacingType, params.SourceSize);
    int line_step = interlaced ? 2 : 1;
    float uv_step_y = float(line_step) * params.SourceSize.w;

    vec4 middle = texelFetch(Source, texel_loc, 0);
    float composite = middle.r;
    vec2 middle_chroma = middle.gb;
    float middle_luma = middle.a;
    vec2 chroma = middle_chroma;

    if (params.FilterType > 0.5) {
        // If we're comb filtering
        float offset = (227.5 - ci.periods_per_line) / ci.periods_per_active_line;
        vec2 upper_chroma = vec2(0.0);
        vec2 lower_chroma = vec2(0.0);
        float upper_luma = 0.0;
        float lower_luma = 0.0;
        if (offset == 0.0) {
            // Aligned scanlines need no fractional-sample interpolation.
            if (texel_loc.y >= line_step) {
                vec3 upper = texelFetch(Source, texel_loc + ivec2(0, -line_step), 0).gba;
                upper_chroma = upper.xy;
                upper_luma = upper.z;
            }
            if (texel_loc.y + line_step < int(params.SourceSize.y)) {
                vec3 lower = texelFetch(Source, texel_loc + ivec2(0, line_step), 0).gba;
                lower_chroma = lower.xy;
                lower_luma = lower.z;
            }
        } else {
            float shift_u = floor(-offset * params.SourceSize.x) * params.SourceSize.z;
            float shift_l = floor( offset * params.SourceSize.x) * params.SourceSize.z;
            vec3 upper = interpolate4(vec2(vTexCoord.x + shift_u, vTexCoord.y - uv_step_y), vWeightsUpper);
            vec3 lower = interpolate4(vec2(vTexCoord.x + shift_l, vTexCoord.y + uv_step_y), vWeightsLower);
            upper_chroma = upper.xy;
            upper_luma = upper.z;
            lower_chroma = lower.xy;
            lower_luma = lower.z;
        }

        if (params.FilterType < 1.5) {
            // Unconditional 3-line comb with boundary handling
            if (texel_loc.y < line_step) {
                chroma = 0.5 * middle_chroma + 0.5 * lower_chroma;
            } else if (texel_loc.y >= int(params.SourceSize.y) - line_step) {
                chroma = 0.5 * middle_chroma + 0.5 * upper_chroma;
            } else {
                chroma = 0.5 * middle_chroma + 0.25 * upper_chroma + 0.25 * lower_chroma;
            }
        } else {
            // Adaptive directional comb with luma-guided cross-color guard
            vec2 uv_scale = vec2(1.0 / 0.436, 1.0 / 0.615);
            vec2 m_scaled = middle_chroma * uv_scale;
            vec2 u_scaled = upper_chroma * uv_scale;
            vec2 d_scaled = lower_chroma * uv_scale;

            float diff_u = length(m_scaled - u_scaled);
            float diff_d = length(m_scaled - d_scaled);
            float luma_diff_u = abs(middle_luma - upper_luma);
            float luma_diff_d = abs(middle_luma - lower_luma);

            float norm_m = length(m_scaled);
            float norm_u = length(u_scaled);
            float norm_d = length(d_scaled);
            float r_anti_u = length(m_scaled + u_scaled) / (norm_m + norm_u + 1.0e-5);
            float r_anti_d = length(m_scaled + d_scaled) / (norm_m + norm_d + 1.0e-5);

            if (texel_loc.y < line_step) {
                diff_u = 999.0;
                luma_diff_u = 999.0;
                r_anti_u = 999.0;
            }
            if (texel_loc.y >= int(params.SourceSize.y) - line_step) {
                diff_d = 999.0;
                luma_diff_d = 999.0;
                r_anti_d = 999.0;
            }

            // First the directional parameter. Evaluate if we should use the
            // top comb, bottom comb, or 3 line comb and blend between them.
            float diff_min = min(diff_u, diff_d);
            float diff_sum = diff_u + diff_d + 1.0e-5;
            float k_dir = clamp(((diff_u - diff_d) / diff_sum) * 3.0, -1.0, 1.0);

            int x_left = max(texel_loc.x - 1, 0);
            int x_right = min(texel_loc.x + 1, int(params.SourceSize.x) - 1);
            vec2 c_left = texelFetch(Source, ivec2(x_left, texel_loc.y), 0).gb * uv_scale;
            vec2 c_right = texelFetch(Source, ivec2(x_right, texel_loc.y), 0).gb * uv_scale;
            float gh = 0.5 * length(c_right - c_left);
            float r_h = (gh / diff_sum) * (params.SourceSize.x / ci.periods_per_active_line);
            float s_g = smoothstep(0.02, 0.05, r_h);

            float luma_diff_max = max(luma_diff_u, luma_diff_d);
            float r_sym = length(u_scaled - d_scaled) / diff_sum;
            if (texel_loc.y < line_step || texel_loc.y >= int(params.SourceSize.y) - line_step) {
                r_sym = 999.0;
            }

            float cross_guard_3line = (1.0 - smoothstep(0.02, 0.08, luma_diff_max)) *
                                      (1.0 - smoothstep(0.50, 0.70, r_sym)) * s_g;
            float guard_u = (1.0 - smoothstep(0.02, 0.08, luma_diff_u)) *
                            (1.0 - smoothstep(0.50, 0.70, r_anti_u)) * s_g;
            float guard_d = (1.0 - smoothstep(0.02, 0.08, luma_diff_d)) *
                            (1.0 - smoothstep(0.50, 0.70, r_anti_d)) * s_g;
            float cross_guard = max(cross_guard_3line, max(guard_u, guard_d));

            float k_cross = ((guard_d - guard_u) / (guard_u + guard_d + 1.0e-5)) * (1.0 - cross_guard_3line);
            float k_eff = k_dir * (1.0 - cross_guard) + k_cross * cross_guard;

            vec2 combed_dir = 0.5 * middle_chroma + 0.25 * ((1.0 - k_eff) * upper_chroma + (1.0 + k_eff) * lower_chroma);
            if (texel_loc.y < line_step) {
                combed_dir = 0.5 * (middle_chroma + lower_chroma);
            } else if (texel_loc.y >= int(params.SourceSize.y) - line_step) {
                combed_dir = 0.5 * (middle_chroma + upper_chroma);
            }

            // Disable the comb filter when there is a large difference on both
            // vertical neighbors and the cross-color guard is not active.
            float w_base = 1.0 - smoothstep(0.06, 0.24, diff_min);
            float w_comb = max(w_base, cross_guard);

            chroma = mix(middle_chroma, combed_dir, w_comb);
        }
    }

    // Remodulate chroma and subtract from composite to recover luma
    float carrier = chroma_carrier(vTexCoord.x, float(texel_loc.y), params.FrameCount, ci, interlaced, params.OddFieldFirst);
    float luma = composite - chroma.r * sin(carrier) - chroma.g * cos(carrier);

    FragColor.rgb = yuv_to_linear(vec3(luma, chroma.r, chroma.g));
}
