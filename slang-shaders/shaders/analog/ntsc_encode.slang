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

#include "chroma_carrier.inc"
#include "filter.inc"

void main() {
    // Filter
    // Systems may have different active line times, so we'll use that to
    // adjust the kernel size for the filter.
    chroma_info ci = get_chroma_info(params.System, params.OriginalSize, params.FrameCount);
    float active_line_micros = ci.periods_per_active_line / CHROMA_SUBCARRIER * 1e6;  // microseconds
    vec3 L = 1.0 / (vec3(params.LumaBandwidth, params.ChromaBandwidth, params.ChromaBandwidth) * active_line_micros * 2.0);

    vec3 filtered = low_pass(L, vTexCoord, params.SourceSize, Source);

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
