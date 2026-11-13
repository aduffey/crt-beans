#version 450

// SPDX-License-Identifier: MIT

layout(set = 0, binding = 0, std140) uniform UBO {
    mat4 MVP;
};

layout(push_constant) uniform Push {
    vec4 SourceSize;
    vec4 OriginalSize;
    float LumaTrapDepth;
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

void main() {
    chroma_info ci = get_chroma_info(params.System, params.OriginalSize, params.FrameCount);
    ivec2 texel_loc = ivec2(floor(vTexCoord * params.SourceSize.xy));
    vec3 filtered = texelFetch(Source, texel_loc, 0).rgb;

    if (params.LumaTrapDepth > 0.0) {
        // Attenuate luma near the chroma subcarrier with a carrier-locked
        // triangle-windowed notch spanning 3 subcarrier periods on each side.
        float width = 3.0 * params.SourceSize.x / ci.periods_per_active_line;
        float width_rcp = 1.0 / width;
        float omega_c = 2.0 * PI * ci.periods_per_active_line * params.SourceSize.z;
        int radius = int(floor(width));
        float trap = 0.0;
        float norm = 0.0;
        for (int i = 1; i <= radius; i++) {
            float c = cos(float(i) * omega_c);
            float w = (1.0 - float(i) * width_rcp) * c;
            vec2 offset = vec2(float(i) * params.SourceSize.z, 0.0);
            float pos_luma = textureLod(Source, vTexCoord + offset, 0.0).r;
            float neg_luma = textureLod(Source, vTexCoord - offset, 0.0).r;
            trap += w * (pos_luma + neg_luma - 2.0 * filtered.r);
            norm += 2.0 * w * (c - 1.0);
        }
        float strength = 1.0 - pow(10.0, -clamp(params.LumaTrapDepth, 0.0, 32.0) / 20.0);
        filtered.r -= strength * trap / norm;
    }

    // Modulate
    // We'll ignore the 7.5 IRE setup and rescaling to 100 IRE because it
    // doesn't really affect the demodulation. It's just inconvenient to deal
    // with.
    bool interlaced = is_interlaced(params.InterlacingType, params.SourceSize);
    float carrier_phase = chroma_carrier(vTexCoord.x, float(texel_loc.y), params.FrameCount, ci, interlaced, params.OddFieldFirst);
    vec2 carriers = vec2(sin(carrier_phase), cos(carrier_phase));
    float composite = dot(filtered, vec3(1.0, carriers));

    // We'll just demodulate here, too, to save a pass.
    vec2 chroma = 2.0 * composite * carriers;
    FragColor.rgb = vec3(composite, chroma.r, chroma.g);
}
