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

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;

void main() {
    gl_Position = MVP * Position;
    vTexCoord = TexCoord;
}

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;
layout(binding = 1) uniform sampler2D Source;

#include "../colorspace/conversions.inc"
#include "chroma_carrier.inc"

float lanczos3(float x) {
    // We don't need to check the outer bounds, just don't call with x > 3
    if (x == 0.0) {
        return 1.0;
    }
    x = PI * x;
    return (sin(x) * sin(x / 3.0)) / (x * x);
}

vec2 interpolate(vec2 coord, sampler2D Source, vec4 SourceSize) {
    // When we look backwards or forwards 227.5 chroma subcarrier periods for
    // our comb filter, we may end up between samples on the previous or next
    // line. If we just use simple hardware bilinear sampling, we end up with
    // aliasing that looks like the chroma subcarrier contaminating the luma.
    // One option to fix this would be to use a higher sampling rate, but the
    // more efficient option is to use a better interpolation method.
    //
    // We can use a Lanczos kernel with a = 3. We only need to interpolate on
    // the horizontal axis.
    float texel_coord = coord.x * SourceSize.x;
	float texel_floor = floor(texel_coord - 0.5);
	float f = (texel_floor + 0.5) - texel_coord;
    float center = (texel_floor + 0.5) * SourceSize.z;

    vec2 result = vec2(0.0);
    float weightSum = 0.0;

    float weight = lanczos3(f - 2.0);
    result += weight * textureLodOffset(Source, vec2(center, coord.y), 0.0, ivec2(-2, 0)).gb;
    weightSum += weight;

    weight = lanczos3(f - 1.0);
    result += weight * textureLodOffset(Source, vec2(center, coord.y), 0.0, ivec2(-1, 0)).gb;
    weightSum += weight;

    weight = lanczos3(f);
    result += weight * textureLodOffset(Source, vec2(center, coord.y), 0.0, ivec2(0, 0)).gb;
    weightSum += weight;

    weight = lanczos3(f + 1.0);
    result += weight * textureLodOffset(Source, vec2(center, coord.y), 0.0, ivec2(1, 0)).gb;
    weightSum += weight;

    weight = lanczos3(f + 2.0);
    result += weight * textureLodOffset(Source, vec2(center, coord.y), 0.0, ivec2(2, 0)).gb;
    weightSum += weight;

    weight = lanczos3(f + 3.0);
    result += weight * textureLodOffset(Source, vec2(center, coord.y), 0.0, ivec2(3, 0)).gb;
    weightSum += weight;

    return result / weightSum;
}

/*
void main() {
    ivec2 texel_loc = ivec2(floor(vTexCoord * params.SourceSize.xy));
    chroma_info ci = get_chroma_info(params.System, params.OriginalSize, params.FrameCount);
    vec3 middle = texelFetch(Source, texel_loc, 0).rgb;
    float composite = middle.r;
    vec2 middle_chroma = middle.gb;
    vec2 chroma = vec2(0.0);

    if (params.FilterType < 0.5) {
        // Notch filter
        // Just pass the demodulated, low-passed chroma through.
        chroma = middle_chroma;
    } else if (params.FilterType < 1.5) {
        // 3 line comb filter
        // First we need to find the positions on the previous and next lines
        // that are 227.5 chroma periods before and after the current position.
        // In standard NTSC, these would be directly above and below the
        // current position, but many consoles had non-standard line lengths.
        // We'll use the values here for our comb filter.
        float offset = (227.5 - ci.periods_per_line) / ci.periods_per_active_line;
        vec2 upper_chroma = interpolate(vTexCoord + vec2(offset, params.SourceSize.w), Source, params.SourceSize);
        vec2 lower_chroma = interpolate(vTexCoord - vec2(offset, params.SourceSize.w), Source, params.SourceSize);
        chroma = 0.5 * middle_chroma + 0.25 * upper_chroma + 0.25 * lower_chroma;
    } else {
        // Adaptive comb filter
        float offset = (227.5 - ci.periods_per_line) / ci.periods_per_active_line;
        vec2 upper_chroma = interpolate(vTexCoord + vec2(offset, params.SourceSize.w), Source, params.SourceSize);
        vec2 lower_chroma = interpolate(vTexCoord - vec2(offset, params.SourceSize.w), Source, params.SourceSize);

        float upper_difference = distance(middle_chroma, upper_chroma);
        float lower_difference = distance(middle_chroma, lower_chroma);

        // First the directional parameter. Evaluate if we should use the top
        // comb, bottom comb, or 3 line comb and blend between them.
        float k_dir = (upper_difference - lower_difference) / (upper_difference + lower_difference + 1.0e-5); // TODO revisit epsilon
        // k_dir = sign(k_dir) * smoothstep(0.05, 0.8, abs(k_dir));  // TODO revisit constants 0.2 and 1.0
        k_dir = 2.0 * smoothstep(-0.8, 0.8, k_dir) - 1.0;
        // When k_dir == 0, this simplifies to the 3 line comb.
        vec2 combed_chroma = ((1.0 - k_dir) * (middle_chroma + upper_chroma) + (1.0 + k_dir) * (middle_chroma + lower_chroma)) / 4.0;

        // Now decide if we need to disable the comb filter (when there is a
        // large difference on the top *and* the bottom).
        float k_comb = 1.0 - smoothstep(0.04, 0.18, min(upper_difference, lower_difference)); // TODO revisit constants
        chroma = mix(middle_chroma, combed_chroma, k_comb);
    }

    // We'll remodulate the chroma and subtract it from the composite signal to
    // get the luma.
    float carrier = chroma_carrier(vTexCoord.x, texel_loc.y, params.FrameCount, ci);
    float luma = composite - chroma.r * sin(carrier) - chroma.g * cos(carrier);

    FragColor.rgb = yuv_to_linear(vec3(luma, chroma.r, chroma.g));
}
*/

void main() {
    ivec2 texel_loc = ivec2(floor(vTexCoord * params.SourceSize.xy));
    chroma_info ci = get_chroma_info(params.System, params.OriginalSize, params.FrameCount);
    bool interlaced = is_interlaced(params.InterlacingType, params.SourceSize);
    int line_step = interlaced ? 2 : 1;
    float uv_step_y = float(line_step) * params.SourceSize.w;

    vec3 middle = texelFetch(Source, texel_loc, 0).rgb;
    float composite = middle.r;
    vec2 middle_chroma = middle.gb;
    vec2 chroma = middle_chroma;

    if (params.FilterType > 0.5) {
        // If we're comb filtering
        float offset = (227.5 - ci.periods_per_line) / ci.periods_per_active_line;
        vec2 upper_chroma = vec2(0.0);
        vec2 lower_chroma = vec2(0.0);
        if (offset == 0.0) {
            // Aligned scanlines need no fractional-sample interpolation.
            if (texel_loc.y >= line_step) {
                upper_chroma = texelFetch(Source, texel_loc + ivec2(0, -line_step), 0).gb;
            }
            if (texel_loc.y + line_step < int(params.SourceSize.y)) {
                lower_chroma = texelFetch(Source, texel_loc + ivec2(0, line_step), 0).gb;
            }
        } else {
            upper_chroma = interpolate(vTexCoord + vec2(-offset, -uv_step_y), Source, params.SourceSize);
            lower_chroma = interpolate(vTexCoord + vec2( offset,  uv_step_y), Source, params.SourceSize);
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
        }
    }

    // Remodulate chroma and subtract from composite to recover luma
    float carrier = chroma_carrier(vTexCoord.x, float(texel_loc.y), params.FrameCount, ci, interlaced, params.OddFieldFirst);
    float luma = composite - chroma.r * sin(carrier) - chroma.g * cos(carrier);

    FragColor.rgb = yuv_to_linear(vec3(luma, chroma.r, chroma.g));
}
