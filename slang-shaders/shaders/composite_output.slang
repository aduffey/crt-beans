#version 450

layout(set = 0, binding = 0, std140) uniform UBO {
    mat4 MVP;
};

layout(push_constant) uniform Push {
    vec4 SourceSize;
    vec4 ScanlinesSize;
    float GlowSigma;
    float GlowAmount;
} params;

#include "parameters.inc.slang"

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

vec3 linear_to_srgb(vec3 rgb) {
    // Floating point inaccuracies could result in negative values, so we'll
    // clamp here to avoid NaNs.
    rgb = clamp(rgb, 0.0, 1.0);
    bvec3 cutoff = lessThan(rgb, vec3(0.0031308));
    vec3 higher = vec3(1.055) * pow(rgb, vec3(1.0 / 2.4)) - vec3(0.055);
    vec3 lower = rgb * vec3(12.92);

    return mix(higher, lower, cutoff);
}

void main() {
    vec3 blur_value = textureLod(Source, vTexCoord, 0.0).rgb;
    vec3 scanlines_value = texelFetch(Scanlines, ivec2(floor(vTexCoord * params.ScanlinesSize.xy)), 0).rgb;
    FragColor = vec4(linear_to_srgb(mix(scanlines_value, blur_value, params.GlowAmount)), 1.0);
}
