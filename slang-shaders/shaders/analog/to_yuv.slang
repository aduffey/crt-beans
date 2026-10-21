#version 450

// SPDX-License-Identifier: MIT

layout(set = 0, binding = 0, std140) uniform UBO {
    mat4 MVP;
};

layout(push_constant) uniform Push {
    vec4 SourceSize;
} params;

#pragma format R16G16B16A16_SFLOAT

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

void main() {
    ivec2 pix_loc = ivec2(int(floor(vTexCoord.x * params.SourceSize.x)), int(floor(vTexCoord.y * params.SourceSize.y)));
    vec3 rgb = texelFetch(Source, pix_loc, 0).rgb;

    // We assume the input is in native CRT gamma.
    FragColor.rgb = gamma_to_yuv(rgb);
}
