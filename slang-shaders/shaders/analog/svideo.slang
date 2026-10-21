#version 450

// SPDX-License-Identifier: MIT

layout(set = 0, binding = 0, std140) uniform UBO {
    mat4 MVP;
};

layout(push_constant) uniform Push {
    vec4 SourceSize;
    float LumaBandwidth;
    float ChromaBandwidth;
} params;

#pragma format R16G16B16A16_SFLOAT

#pragma parameter LumaBandwidth "Luma bandwidth (MHz)" 4.0 0.6 6.0 0.2
#pragma parameter ChromaBandwidth "Chroma bandwidth (MHz, each channel)" 1.2 0.6 1.6 0.2

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
#include "filter.inc"

void main() {
    // We could modulate and demodulate chroma as in a real S-Video connection,
    // but there's really no visual difference from just low pass filtering
    // them appropriately.
    vec3 L = 1.0 / (vec3(params.LumaBandwidth, params.ChromaBandwidth, params.ChromaBandwidth) * 52.5 * 2.0);
    vec3 filtered = low_pass(L, vTexCoord, params.SourceSize, Source);
    FragColor.rgb = yuv_to_linear(filtered);
}
