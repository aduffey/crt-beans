#version 450

// SPDX-License-Identifier: MIT

layout(set = 0, binding = 0, std140) uniform UBO {
    mat4 MVP;
};

layout(push_constant) uniform Push {
    vec4 SourceSize;
    vec4 OriginalSize;
    float LumaBandwidth;
    float ChromaBandwidth;
    float LumaTrapDepth;
    float System;
    uint FrameCount;
    float InterlacingType;
    float OddFieldFirst;
} params;

#pragma format R16G16B16A16_SFLOAT

#include "parameters_ntsc.inc"
#include "chroma_carrier.inc"
#include "luma_trap.inc"

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
layout(location = 1) flat out vec3 vLumaTrapCoefficients;

void main() {
    gl_Position = MVP * Position;
    vTexCoord = TexCoord;
    vLumaTrapCoefficients = vec3(0.0);
    if (params.LumaTrapDepth > 0.0) {
        float l = CHROMA_SUBCARRIER / (6.0 * params.LumaBandwidth * 1e6);
        vLumaTrapCoefficients = luma_trap_coefficients(l, params.LumaTrapDepth);
    }
}

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 1) flat in vec3 vLumaTrapCoefficients;
layout(location = 0) out vec4 FragColor;
layout(binding = 1) uniform sampler2D Source;

#include "filter.inc"

vec3 low_pass_luma_trap(
    vec3 L, float trap_radius, vec2 pos, vec4 source_size, sampler2D Source
) {
    float radius = max(trap_radius, max(L.r, max(L.g, L.b)));
    vec3 L_rcp = 1.0 / L;
    float trap_radius_rcp = 1.0 / trap_radius;
    int pix_y = int(floor(pos.y * source_size.y));
    int start = max(int(floor(source_size.x * (pos.x - radius))), 0);
    int end = min(int(floor(source_size.x * (pos.x + radius))), int(source_size.x) - 1);

    float offset = pos.x - float(start) * source_size.z;
    vec3 u = clamp(offset * L_rcp, -1.0, 1.0);
    vec3 left = PI * u + sin(PI * u);
    left.r -= luma_trap_primitive(offset * trap_radius_rcp, vLumaTrapCoefficients);

    vec3 filtered = vec3(0.0);
    for (int pix_x = start; pix_x <= end; pix_x++) {
        vec3 s = texelFetch(Source, ivec2(pix_x, pix_y), 0).rgb;
        offset = pos.x - float(pix_x + 1) * source_size.z;
        u = clamp(offset * L_rcp, -1.0, 1.0);
        vec3 right = PI * u + sin(PI * u);
        right.r -= luma_trap_primitive(offset * trap_radius_rcp, vLumaTrapCoefficients);
        filtered += s * (left - right);
        left = right;
    }
    return filtered / (2.0 * PI);
}

void main() {
    // Filter
    // Systems may have different active line times, so we'll use that to
    // adjust the kernel size for the filter.
    chroma_info ci = get_chroma_info(params.System, params.OriginalSize, params.FrameCount);
    float active_line_micros = ci.periods_per_active_line / CHROMA_SUBCARRIER * 1e6;  // microseconds
    vec3 L = 1.0 / (vec3(params.LumaBandwidth, params.ChromaBandwidth, params.ChromaBandwidth) * active_line_micros * 2.0);

    vec3 filtered;
    if (params.LumaTrapDepth > 0.0) {
        float trap_radius = 3.0 / ci.periods_per_active_line;
        filtered = low_pass_luma_trap(L, trap_radius, vTexCoord, params.SourceSize, Source);
    } else {
        filtered = low_pass(L, vTexCoord, params.SourceSize, Source);
    }
    // Retain the signed luma from the narrow trap through modulation.

    // Modulate
    // We'll ignore the 7.5 IRE setup and rescaling to 100 IRE because it
    // doesn't really affect the demodulation. It's just inconvenient to deal
    // with.
    ivec2 texel_loc = ivec2(floor(vTexCoord * params.SourceSize.xy));
    bool interlaced = is_interlaced(params.InterlacingType, params.SourceSize);
    float carrier = chroma_carrier(vTexCoord.x, float(texel_loc.y), params.FrameCount, ci, interlaced, params.OddFieldFirst);
    float composite = dot(filtered, vec3(1.0, sin(carrier), cos(carrier)));

    // We'll just demodulate here, too, to save a pass.
    vec2 chroma = composite * vec2(2.0 * sin(carrier), 2.0 * cos(carrier));
    FragColor.rgb = vec3(composite, chroma.r, chroma.g);
}
