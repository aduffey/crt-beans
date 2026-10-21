#version 450

// SPDX-License-Identifier: MIT

layout(set = 0, binding = 0, std140) uniform UBO {
    mat4 MVP;
};

layout(push_constant) uniform Push {
    float OutputGamma;
    vec4 SourceSize;
    vec4 ScanlinesSize;
    uint FrameCount;
    float GlowSigma;
    float GlowAmount;
} params;

#include "parameters_common.inc"
#include "colorspace/conversions.inc"

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
layout(binding = 2) uniform sampler2D Scanlines;
layout(binding = 3) uniform sampler2D BlueNoiseTex;

// Plastic constant R2 sequence for 2D toroidal spatial jump
const uvec2 R2 = uvec2(3242171952u, 2447432860u);

// Golden ratio R1 sequence for value-domain Cranley-Patterson rotation
const uint GOLDEN_RATIO_CONJ = 2654435769u;

vec3 noise(ivec2 screen_coord) {
    // We don't want to only offset the noise each frame or the eye can track
    // its movement. We will offset and TODO
    //
    // We will use unsigned integer math to avoid a loss of precision as the
    // FrameCount grows.
    //
    // 1. Calculate integer toroidal pixel offset
    // uvec is 32 bits and our texture is 2^6x2^6, so we shift by 32 - 6.
    ivec2 spatial_offset = ivec2((uvec2(params.FrameCount) * R2) >> uvec2(26u));

    // 2. Fetch spatially shifted blue noise (toroidal wrap via bitwise AND)
    // NOTE: We are assuming the blue noise texture size is 64x64.
    ivec2 texel_coord = (screen_coord + spatial_offset) & ivec2(63);
    vec3 raw_noise = texelFetch(BlueNoiseTex, texel_coord, 0).rgb;

    // 3. Apply Cranley-Patterson rotation
    uint temporal_fixed = (params.FrameCount * GOLDEN_RATIO_CONJ) >> 8u;
    float temporal_offset = float(temporal_fixed) * (1.0 / 16777216.0); // Exact divide by 2^24

    return fract(raw_noise + temporal_offset);
}

// Apply a simple 8-bit dither to minimize banding in the glow.
vec3 dither_srgb(vec3 rgb, ivec2 icoords) {
    vec3 noise = noise(icoords);

    const float quantum = 1.0 / 255.0;
    vec3 lower = floor(linear_to_srgb(rgb) / quantum) * quantum;
    vec3 upper = lower + quantum;
    vec3 threshold = mix(srgb_to_linear(lower), srgb_to_linear(upper), noise);
    return mix(lower, upper, lessThan(threshold, rgb));
}

// Apply a simple 8-bit dither to minimize banding in the glow.
vec3 dither_gamma(vec3 rgb, ivec2 icoords) {
    vec3 noise = noise(icoords);

    const float quantum = 1.0 / 255.0;
    vec3 lower = floor(linear_to_gamma(rgb, 2.2) / quantum) * quantum;
    vec3 upper = lower + quantum;
    vec3 threshold = mix(gamma_to_linear(lower, 2.2), gamma_to_linear(upper, 2.2), noise);
    return mix(lower, upper, lessThan(threshold, rgb));
}

void main() {
    vec3 blur_value = textureLod(Source, vTexCoord, 0.0).rgb;
    ivec2 icoords = ivec2(floor(vTexCoord * params.ScanlinesSize.xy));
    vec3 scanlines_value = texelFetch(Scanlines, icoords, 0).rgb;
    vec3 out_value = mix(scanlines_value, blur_value, params.GlowAmount);
    if (params.OutputGamma < 0.5) {
        out_value = dither_srgb(out_value, icoords);
    } else {
        out_value = dither_gamma(out_value, icoords);
    }
    FragColor = vec4(out_value, 1.0);
}
