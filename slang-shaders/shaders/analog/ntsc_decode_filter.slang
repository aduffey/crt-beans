#version 450

// SPDX-License-Identifier: MIT

layout(set = 0, binding = 0, std140) uniform UBO {
    mat4 MVP;
};

layout(push_constant) uniform Push {
    vec4 SourceSize;
    vec4 OriginalSize;
    float System;
    uint FrameCount;
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
layout(set = 0, binding = 1) uniform sampler2D Source;

#include "chroma_carrier.inc"

float hann_function(float x, float width_rcp) {
    return 0.5 + 0.5 * cos(PI * x * width_rcp);
}

void main() {
    chroma_info ci = get_chroma_info(params.System, params.OriginalSize, params.FrameCount);
    ivec2 texel_loc = ivec2(floor(vTexCoord * params.SourceSize.xy));

    // Low-pass filter the chroma. We use a carrier-locked Hann window as our
    // kernel instead of a windowed sinc. Setting the half-width to 1.5 subcarrier
    // periods places exact spectral nulls at fc (3.58 MHz) and 2*fc (7.16 MHz),
    // eliminating demodulated subcarrier ripple with no ringing and low cost.
    float width = 1.5 * params.SourceSize.x / ci.periods_per_active_line;
    float width_rcp = 1.0 / width;
    int radius = int(floor(width));

    // Center tap (i = 0): hann_function(0, width_rcp) == 1.0
     float weight_sum = 1.0;
     vec3 center = textureLod(Source, vTexCoord, 0.0).rgb;
     vec2 chroma = center.gb;
     float low_luma = center.r;

     // Symmetric side taps. Note that we are assuming the input size is equal
     // to the output size.
     for (int i = 1; i <= radius; i++) {
         float coeff = hann_function(float(i), width_rcp);
         weight_sum += 2.0 * coeff;
         vec2 offset = vec2(float(i) * params.SourceSize.z, 0.0);
         vec3 pair = textureLod(Source, vTexCoord + offset, 0.0).rgb +
                     textureLod(Source, vTexCoord - offset, 0.0).rgb;
         chroma += coeff * pair.gb;
         low_luma += coeff * pair.r;
    }
    chroma = chroma / weight_sum;
    low_luma = low_luma / weight_sum;

    float composite = texelFetch(Source, texel_loc, 0).r;
    // Low-frequency signal-domain luma for adaptive decisions, not output Y.
    FragColor = vec4(composite, chroma.r, chroma.g, low_luma);
}
