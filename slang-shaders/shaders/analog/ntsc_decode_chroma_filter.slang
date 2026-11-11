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
layout(binding = 1) uniform sampler2D Source;

#include "chroma_carrier.inc"
#include "../colorspace/conversions.inc"

float hann_function(float x, float width_rcp) {
    return 0.5 + 0.5 * cos(PI * x * width_rcp);
}

void main() {
    chroma_info ci = get_chroma_info(params.System, params.OriginalSize, params.FrameCount);

    // Low-pass filter the displayed chroma with the same carrier-locked Hann
    // window as the demodulation pass. Setting the half-width to 1.5
    // subcarrier periods cascades with the pre-filter to place the -3 dB point
    // at ~614 kHz. This suppresses dither beats and color fringing with some
    // consoles (particularly those that use dot clocks at 1.5 * the chroma
    // subcarrier, like SNES, which beat at ~895 kHz).
    //
    // This is also more realistic, as many NTSC comb filter ICs had full-
    // pipeline chroma bandwidths of 500-650 kHz. Toshiba ICs (and probably
    // others) also had pre- and post-filters with split luma subtraction
    // paths.
    //
    // We subtract the chroma from the composite signal to recover the luma
    // *before* this second filter so that we don't introduce too much
    // uncanceled chroma into the luma as dot crawl or hanging dots.
    float width = 1.5 * params.SourceSize.x / ci.periods_per_active_line;
    float width_rcp = 1.0 / width;
    int radius = int(floor(width));

    float weight_sum = 1.0;
    vec3 center_sample = textureLod(Source, vTexCoord, 0.0).rgb;
    vec2 chroma = center_sample.gb;
    float luma = center_sample.r;

    for (int i = 1; i <= radius; i++) {
        float coeff = hann_function(float(i), width_rcp);
        weight_sum += 2.0 * coeff;
        vec2 offset = vec2(float(i) * params.SourceSize.z, 0.0);
        vec2 pos_chroma = textureLod(Source, vTexCoord + offset, 0.0).gb;
        vec2 neg_chroma = textureLod(Source, vTexCoord - offset, 0.0).gb;
        chroma += coeff * (pos_chroma + neg_chroma);
    }
    chroma = chroma / weight_sum;

    FragColor.rgb = yuv_to_linear(vec3(luma, chroma.r, chroma.g));
}
