#version 450

// SPDX-License-Identifier: MIT

layout(set = 0, binding = 0, std140) uniform UBO {
    mat4 MVP;
};

layout(push_constant) uniform Push {
    vec4 SourceSize;
    vec4 OriginalSize;
    float System;
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

float lanczos3(float x) {
    if (x < 1.0e-5) {
        return 1.0;
    }
    float pi_x = PI * x;
    return (sin(pi_x) * sin(pi_x * (1.0 / 3.0))) / (pi_x * pi_x * (1.0 / 3.0));
}

void main() {
    chroma_info ci = get_chroma_info(params.System, params.OriginalSize);

    // Low-pass filter the displayed chroma with the same carrier-locked Hann
    // window as the demodulation pass, and resample luma with Lanczos-3 so
    // the output can upsample from 900 to 1800 samples without box/linear
    // high-frequency roll-off. Setting the chroma half-width to 1.5 subcarrier
    // periods cascades with the pre-filter to place the -3 dB point at ~614 kHz.
    // This suppresses dither beats and color fringing with some consoles
    // (particularly those that use dot clocks at 1.5 * the chroma subcarrier,
    // like SNES, which beat at ~895 kHz).
    //
    // This is also more realistic, as many NTSC comb filter ICs had full-
    // pipeline chroma bandwidths of 500-650 kHz. Toshiba ICs (and probably
    // others) also had pre- and post-filters with split luma subtraction
    // paths.
    //
    // We subtract the chroma from the composite signal to recover the luma
    // *before* this second filter so that we don't introduce too much
    // uncanceled chroma into the luma as dot crawl or hanging dots.
    float tex_x = vTexCoord.x * params.SourceSize.x;

    float width = 1.5 * params.SourceSize.x / ci.periods_per_active_line;
    float width_rcp = 1.0 / width;

    int chroma_start = int(floor(tex_x - width + 0.5));
    int chroma_stop  = int(floor(tex_x + width - 0.5));
    int luma_start   = int(floor(tex_x - 3.0 + 0.5));
    int luma_stop    = int(floor(tex_x + 3.0 - 0.5));
    int start_x = min(chroma_start, luma_start);
    int stop_x  = max(chroma_stop, luma_stop);

    vec2 chroma = vec2(0.0);
    float chroma_weight_sum = 0.0;
    float luma = 0.0;
    float luma_weight_sum = 0.0;

    for (int ix = start_x; ix <= stop_x; ix++) {
        float sample_x = (float(ix) + 0.5) * params.SourceSize.z;
        vec3 sample_val = textureLod(Source, vec2(sample_x, vTexCoord.y), 0.0).rgb;
        float abs_dx = abs(tex_x - (float(ix) + 0.5));
        if (abs_dx < width) {
            float c_coeff = hann_function(abs_dx, width_rcp);
            chroma_weight_sum += c_coeff;
            chroma += c_coeff * sample_val.gb;
        }
        if (abs_dx < 3.0) {
            float l_coeff = lanczos3(abs_dx);
            luma_weight_sum += l_coeff;
            luma += l_coeff * sample_val.r;
        }
    }
    if (chroma_weight_sum > 0.0) {
        chroma /= chroma_weight_sum;
    }
    if (luma_weight_sum > 0.0) {
        luma /= luma_weight_sum;
    }

    FragColor.rgb = yuv_to_linear(vec3(luma, chroma.r, chroma.g));
}
