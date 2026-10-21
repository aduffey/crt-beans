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

float hann_function(float x, float width_rcp) {
    return 0.5 + 0.5 * cos(PI * x * width_rcp);
}

void main() {
    chroma_info ci = get_chroma_info(params.System, params.OriginalSize, params.FrameCount);
    ivec2 texel_loc = ivec2(floor(vTexCoord * params.SourceSize.xy));

    // Low-pass filter the chroma. We'll just use a Hann function as our kernel
    // instead of a windowed sinc. This is faster, results in no ringing, and
    // still has a reasonable roll-off. We're simulating old analog filters in
    // cheap electronics, anyway.

    // First find the filter width for 1.3MHz
    float active_line_seconds = ci.periods_per_active_line / CHROMA_SUBCARRIER;
    float width = params.SourceSize.x / (1.3e6 * active_line_seconds * 2.0);

    vec2 chroma = vec2(0.0);
    float width_rcp = 1.0 / width;
    float weight_sum = 0.0;
    for (int i = -int(floor(width)); i <= int(floor(width)); i++) {
        // Note that these indexes (and the hann function not needing a clamp)
        // rely on the size of the input and ouput being the same.
        // TODO take advantage of symmetry?
        // TODO precompute coefficients in vertex shader?
        float coeff = hann_function(i, width_rcp);
        weight_sum += coeff;
        // We use textureLod because texelFetch is undefined for out-of-texture
        // accesses.
        vec2 offset = vec2(i * params.SourceSize.z, 0.0);
        chroma += coeff * textureLod(Source, vTexCoord + offset, 0.0).gb;
    }
    chroma = chroma / weight_sum;

    float composite = texelFetch(Source, texel_loc, 0).r;
    FragColor.rgb = vec3(composite, chroma.r, chroma.g);
}
