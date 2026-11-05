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

// Displayed-chroma guards fitted across console screenshots. Chroma extrema
// also detect cross-color, so their transition is broader than low-pass Y's.
const float NTSC_COMB_LUMA_LOW = 0.0;
const float NTSC_COMB_LUMA_HIGH = 0.2;
const float NTSC_COMB_CHROMA_LOW = 0.0;
const float NTSC_COMB_CHROMA_HIGH = 1.5;
// Independent luma-subtraction guards, matching the Python baseband adaptation.
const float NTSC_COMB_PROTECT_LOW = 0.04;
const float NTSC_COMB_PROTECT_HIGH = 0.12;
const float NTSC_COMB_LUMA_FLOOR = 0.05;
const float NTSC_COMB_CHROMA_RATIO = 1.5;

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
    vec2 chroma_for_luma = middle_chroma;

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
                chroma_for_luma = chroma;
            } else if (texel_loc.y >= int(params.SourceSize.y) - line_step) {
                chroma = 0.5 * middle_chroma + 0.5 * upper_chroma;
                chroma_for_luma = chroma;
            } else {
                chroma = 0.5 * middle_chroma + 0.25 * upper_chroma + 0.25 * lower_chroma;
                chroma_for_luma = chroma;
            }
        } else {
            // 5-candidate median comb with Toshiba isolated-feature guard
            if (texel_loc.y < line_step) {
                chroma = 0.5 * (middle_chroma + lower_chroma);
                chroma_for_luma = chroma;
            } else if (texel_loc.y >= int(params.SourceSize.y) - line_step) {
                chroma = 0.5 * (middle_chroma + upper_chroma);
                chroma_for_luma = chroma;
            } else {
                // 5-candidate median over {middle, upper, lower, (middle+upper)/2, (middle+lower)/2}
                // Because the two averaged candidates are midpoints bounded between middle and
                // their respective adjacent lines, the 5-element median simplifies algebraically
                // to clamping middle_chroma within bounds determined by upper and lower chroma.
                vec2 u_min = min(upper_chroma, lower_chroma);
                vec2 u_max = max(upper_chroma, lower_chroma);
                vec2 cand_min = 0.5 * (middle_chroma + u_min);
                vec2 cand_max = 0.5 * (middle_chroma + u_max);
                vec2 lo = min(u_min, cand_max);
                vec2 hi = max(u_max, cand_min);
                vec2 c_med5 = clamp(middle_chroma, lo, hi);

                // Toshiba US5548342A isolated-feature detector (luma)
                float luma_delta_u = middle_luma - upper_luma;
                float luma_delta_d = middle_luma - lower_luma;
                float iso_luma = max(0.0, min(abs(luma_delta_u), abs(luma_delta_d)) * (sign(luma_delta_u) * sign(luma_delta_d)));
                float w_iso_luma = smoothstep(NTSC_COMB_LUMA_LOW, NTSC_COMB_LUMA_HIGH, iso_luma);

                // Baseband approximation of the signed carrier-band extrema
                // detector in Toshiba JPH0865706A / JP3351630B2.
                vec2 chroma_delta_u = middle_chroma - upper_chroma;
                vec2 chroma_delta_d = middle_chroma - lower_chroma;
                const vec2 uv_scale = vec2(1.0 / 0.436, 1.0 / 0.615);
                chroma_delta_u *= uv_scale;
                chroma_delta_d *= uv_scale;
                vec2 iso_chroma = max(vec2(0.0), min(abs(chroma_delta_u), abs(chroma_delta_d)) * (sign(chroma_delta_u) * sign(chroma_delta_d)));
                float w_iso_chroma = smoothstep(NTSC_COMB_CHROMA_LOW, NTSC_COMB_CHROMA_HIGH, max(iso_chroma.r, iso_chroma.g));

                float w_iso = max(w_iso_luma, w_iso_chroma);
                chroma = mix(c_med5, middle_chroma, w_iso);

                // Protect carrier cancellation independently of displayed UV,
                // following Toshiba's separate C/CC paths (US5548342A).
                // Relative Y and chroma-amplitude excess are our baseband
                // adaptation: dark isolated lines need stronger protection.
                float y_sub_scale = max(max(abs(middle_luma), abs(upper_luma)),
                    max(abs(lower_luma), NTSC_COMB_LUMA_FLOOR));
                float y_sub_iso_y = iso_luma / y_sub_scale;
                float y_sub_norm_m = length(middle_chroma * uv_scale);
                float y_sub_norm_u = length(upper_chroma * uv_scale);
                float y_sub_norm_d = length(lower_chroma * uv_scale);
                float y_sub_iso_c = max(0.0,
                    y_sub_norm_m - NTSC_COMB_CHROMA_RATIO * max(y_sub_norm_u, y_sub_norm_d));
                float y_sub_weight = max(smoothstep(NTSC_COMB_PROTECT_LOW, NTSC_COMB_PROTECT_HIGH, y_sub_iso_y),
                    smoothstep(NTSC_COMB_PROTECT_LOW, NTSC_COMB_PROTECT_HIGH, y_sub_iso_c));
                chroma_for_luma = mix(c_med5, middle_chroma, y_sub_weight);
            }
        }
    }

    // Cancel the carrier with protected chroma; display UV keeps its combing.
    float carrier = chroma_carrier(vTexCoord.x, float(texel_loc.y), params.FrameCount, ci, interlaced, params.OddFieldFirst);
    float luma = composite - chroma_for_luma.r * sin(carrier) - chroma_for_luma.g * cos(carrier);

    FragColor.rgb = yuv_to_linear(vec3(luma, chroma.r, chroma.g));
}
