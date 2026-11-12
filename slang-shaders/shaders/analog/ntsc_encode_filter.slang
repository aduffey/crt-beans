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
#include "filter.inc"

void main() {
    // Systems may have different active line times, so we'll use that to
    // adjust the kernel size for the filter.
    chroma_info ci = get_chroma_info(params.System, params.OriginalSize, params.FrameCount);
    float active_line_micros = ci.periods_per_active_line / CHROMA_SUBCARRIER * 1e6;  // microseconds
    vec3 L = 1.0 / (vec3(params.LumaBandwidth, params.ChromaBandwidth, params.ChromaBandwidth) * active_line_micros * 2.0);

    FragColor = vec4(low_pass(L, vTexCoord, params.SourceSize, Source), 1.0);
}
