#version 450

layout(set = 0, binding = 0, std140) uniform UBO {
    mat4 MVP;
};

layout(push_constant) uniform Push {
    vec4 SourceSize;
    float MaxSpotSize;
    float MinSpotSize;
} params;

// 2 * sqrt(2 * log(2)) * sigma = full width at half maximum of a gaussian.
// We're using the FWHM as the spot size parameter, so we need this factor to
// find the sigma.
#define SCALE 2.3548200450309493

#pragma format R16G16B16A16_SFLOAT

#include "parameters_common.inc.slang"
#include "parameters_advanced.inc.slang"

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

void main() {
    ivec2 pix_loc = ivec2(floor(vTexCoord * params.SourceSize.xy));
    vec3 sample_value = texelFetch(Source, pix_loc, 0).rgb;
    vec3 width =  params.MinSpotSize * params.MaxSpotSize - sqrt(sample_value) * (params.MinSpotSize * params.MaxSpotSize - params.MaxSpotSize);
    vec3 sigma_rcp = SCALE / width;
    FragColor.rgb = 0.5 * sigma_rcp * sigma_rcp;
}
