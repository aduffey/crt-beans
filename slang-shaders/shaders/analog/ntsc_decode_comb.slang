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
layout(set = 0, binding = 1) uniform sampler2D Source;

#include "../colorspace/conversions.inc"

vec2 interpolate4(vec2 center_coord, vec4 w) {
    vec2 result = w.x * textureLodOffset(Source, center_coord, 0.0, ivec2(-1, 0)).gb;
    result += w.y * textureLodOffset(Source, center_coord, 0.0, ivec2( 0, 0)).gb;
    result += w.z * textureLodOffset(Source, center_coord, 0.0, ivec2( 1, 0)).gb;
    result += w.w * textureLodOffset(Source, center_coord, 0.0, ivec2( 2, 0)).gb;
    return result;
}

// Raw, unshifted (U, V, low-frequency Y). Do not issue out-of-range texelFetch.
vec3 comb_sample(ivec2 loc) {
    if (loc.y < 0 || loc.y >= int(params.SourceSize.y)) {
        return vec3(0.0);
    }
    return texelFetch(Source, loc, 0).gba;
}

vec2 adaptive_phase_comb(
    vec3 middle, vec3 upper, vec3 lower, vec3 upper2, vec3 lower2,
    vec3 upper_period, vec3 lower_period, vec2 left, vec2 right,
    vec2 baseline, int period, float density, bool valid_upper, bool valid_lower
) {
    // Keep thresholds and the normalized correction cap in sync with filtering.py.
    const vec2 uv_scale = vec2(1.0 / 0.436, 1.0 / 0.615);
    vec2 m = middle.xy, u = upper.xy, d = lower.xy;
    vec2 delta_left = (m - left) * uv_scale;
    vec2 delta_right = (right - m) * uv_scale;
    float energy = density * sqrt(0.5 * (dot(delta_left, delta_left) + dot(delta_right, delta_right)));
    float horizontal = smoothstep(0.02, 0.08, energy);

    float periodic_u = valid_upper ? 1.0 - smoothstep(0.04, 0.16, length((m - upper_period.xy) * uv_scale)) : 0.0;
    float periodic_d = valid_lower ? 1.0 - smoothstep(0.04, 0.16, length((m - lower_period.xy) * uv_scale)) : 0.0;
    float luma_u = abs(middle.z - upper.z);
    float luma_d = abs(middle.z - lower.z);
    float center_luma = max(luma_u, luma_d);
    vec2 center = 0.5 * m + 0.25 * (u + d);
    vec2 candidate_u = 0.5 * (m + u);
    vec2 candidate_d = 0.5 * (m + d);
    if (period == 3) {
        // The modeled NES/SNES carrier advances 120 degrees per field line.
        center = (m + u + d) / 3.0;
        candidate_u = (m + u + upper2.xy) / 3.0;
        candidate_d = (m + d + lower2.xy) / 3.0;
        luma_u = max(luma_u, abs(middle.z - upper2.z));
        luma_d = max(luma_d, abs(middle.z - lower2.z));
    }
    vec3 confidence = horizontal * vec3(
        max(periodic_u, periodic_d) * (1.0 - smoothstep(0.02, 0.06, center_luma)),
        periodic_u * (1.0 - smoothstep(0.02, 0.06, luma_u)),
        periodic_d * (1.0 - smoothstep(0.02, 0.06, luma_d))
    );
    if (!valid_upper || !valid_lower) {
        confidence.x = 0.0;
    }
    vec3 weights = confidence * confidence;
    vec2 selected = (weights.x * center + weights.y * candidate_u + weights.z * candidate_d) /
                    max(weights.x + weights.y + weights.z, 1.0e-12);
    float strength = max(confidence.x, max(confidence.y, confidence.z));
    vec2 correction = strength * (selected - baseline);
    // Limit damage when true chroma and cross-color have the same spatial signature.
    correction *= min(1.0, 0.08 / max(length(correction * uv_scale), 1.0e-12));
    return baseline + correction;
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
    vec2 chroma = middle_chroma;

    if (params.FilterType > 0.5) {
        // If we're comb filtering
        float offset = (227.5 - ci.periods_per_line) / ci.periods_per_active_line;
        vec2 upper_chroma = vec2(0.0);
        vec2 lower_chroma = vec2(0.0);
        vec3 upper = vec3(0.0);
        vec3 lower = vec3(0.0);
        if (offset == 0.0) {
            // Aligned scanlines need no fractional-sample interpolation.
            upper = comb_sample(texel_loc + ivec2(0, -line_step));
            lower = comb_sample(texel_loc + ivec2(0, line_step));
            upper_chroma = upper.xy;
            lower_chroma = lower.xy;
        } else {
            float shift_u = floor(-offset * params.SourceSize.x) * params.SourceSize.z;
            float shift_l = floor( offset * params.SourceSize.x) * params.SourceSize.z;
            upper_chroma = interpolate4(vec2(vTexCoord.x + shift_u, vTexCoord.y - uv_step_y), vWeightsUpper);
            lower_chroma = interpolate4(vec2(vTexCoord.x + shift_l, vTexCoord.y + uv_step_y), vWeightsLower);
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
            // Adaptive directional comb with edge fallback
            vec2 uv_scale = vec2(1.0 / 0.436, 1.0 / 0.615);
            vec2 m_scaled = middle_chroma * uv_scale;
            vec2 u_scaled = upper_chroma * uv_scale;
            vec2 d_scaled = lower_chroma * uv_scale;

            float diff_u = length(m_scaled - u_scaled);
            float diff_d = length(m_scaled - d_scaled);

            if (texel_loc.y < line_step) {
                diff_u = 999.0;
            }
            if (texel_loc.y >= int(params.SourceSize.y) - line_step) {
                diff_d = 999.0;
            }

            // First the directional parameter. Evaluate if we should use the
            // top comb, bottom comb, or 3 line comb and blend between them.
            float diff_min = min(diff_u, diff_d);
            float diff_sum = diff_u + diff_d + 1.0e-5;
            float k_dir = clamp(((diff_u - diff_d) / diff_sum) * 3.0, -1.0, 1.0);

            vec2 combed_dir = 0.5 * middle_chroma + 0.25 * ((1.0 - k_dir) * upper_chroma + (1.0 + k_dir) * lower_chroma);
            if (texel_loc.y < line_step) {
                combed_dir = 0.5 * (middle_chroma + lower_chroma);
            } else if (texel_loc.y >= int(params.SourceSize.y) - line_step) {
                combed_dir = 0.5 * (middle_chroma + upper_chroma);
            }

            // Now decide if we need to disable the comb filter (when there is
            // a large difference on the top *and* the bottom).
            float w_comb = 1.0 - smoothstep(0.06, 0.24, diff_min);

            chroma = mix(middle_chroma, combed_dir, w_comb);

            // Genesis/SMS have no line-to-line phase diversity: keep the baseline.
            float line_phase = fract(ci.periods_per_line);
            if (line_phase > 1.0e-4) {
                int period = abs(line_phase - 0.5) < 1.0e-4 ? 2 : 3;
                if (offset != 0.0) {
                    upper = comb_sample(texel_loc + ivec2(0, -line_step));
                    lower = comb_sample(texel_loc + ivec2(0, line_step));
                }
                vec3 upper2 = comb_sample(texel_loc + ivec2(0, -2 * line_step));
                vec3 lower2 = comb_sample(texel_loc + ivec2(0, 2 * line_step));
                vec3 upper_period = upper2;
                vec3 lower_period = lower2;
                if (period == 3) {
                    upper_period = comb_sample(texel_loc + ivec2(0, -3 * line_step));
                    lower_period = comb_sample(texel_loc + ivec2(0, 3 * line_step));
                }
                vec2 left = texelFetch(Source, ivec2(max(texel_loc.x - 1, 0), texel_loc.y), 0).gb;
                vec2 right = texelFetch(Source, ivec2(min(texel_loc.x + 1, int(params.SourceSize.x) - 1), texel_loc.y), 0).gb;
                // Normalize differences to the standard carrier at 900 samples.
                const float standard_periods = (315.0 / 88.0) * 1.0e6 * 52.5e-6;
                float density = (params.SourceSize.x / ci.periods_per_active_line) * (standard_periods / 900.0);
                chroma = adaptive_phase_comb(
                    middle.gba, upper, lower, upper2, lower2, upper_period, lower_period,
                    left, right, chroma, period, density,
                    texel_loc.y >= period * line_step,
                    texel_loc.y + period * line_step < int(params.SourceSize.y)
                );
            }
        }
    }

    // Remodulate chroma and subtract from composite to recover luma
    float carrier = chroma_carrier(vTexCoord.x, float(texel_loc.y), params.FrameCount, ci, interlaced, params.OddFieldFirst);
    float luma = composite - chroma.r * sin(carrier) - chroma.g * cos(carrier);

    FragColor = vec4(yuv_to_linear(vec3(luma, chroma.r, chroma.g)), 1.0);
}
