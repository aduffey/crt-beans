#version 450

layout(set = 0, binding = 0, std140) uniform UBO
{
    mat4 MVP;
};

layout(push_constant) uniform Push
{
    vec4 SourceSize;
    float YIQ;
} params;

#pragma format R16G16B16A16_SFLOAT
#pragma parameter YIQ "Composite" 0.0 0.0 1.0 1.0

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
void main()
{
   gl_Position = MVP * Position;
   vTexCoord = TexCoord;
}

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;
layout(binding = 1) uniform sampler2D Source;

#define CRT_GAMMA 2.4
#define GAMMA_YIQ_MAT mat3x3(0.30, 0.599, 0.213, 0.59, -0.2773, -0.5251, 0.11, -0.3217, 0.3121)

vec3 srgb_to_gamma(vec3 srgb) {
    bvec3 cutoff = lessThan(srgb, vec3(0.04045));
    vec3 higher = pow((srgb + vec3(0.055)) / vec3(1.055), vec3(2.4));
    vec3 lower = srgb / vec3(12.92);

    return pow(mix(higher, lower, cutoff), vec3(1.0 / CRT_GAMMA));
}

vec3 srgb_to_yiq(vec3 srgb) {
    return GAMMA_YIQ_MAT * srgb_to_gamma(srgb);
}

void main()
{
    ivec2 pix_loc = ivec2(int(floor(vTexCoord.x * params.SourceSize.x)), int(floor(vTexCoord.y * params.SourceSize.y)));
    vec3 rgb = texelFetch(Source, pix_loc, 0).rgb;
    if (params.YIQ > 0.5) {
        FragColor = vec4(srgb_to_yiq(rgb), 1.0);
    } else {
        FragColor = vec4(srgb_to_gamma(rgb), 1.0);
    }
}
