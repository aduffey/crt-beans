#version 450

layout(set = 0, binding = 0, std140) uniform UBO {
    mat4 MVP;
};

layout(push_constant) uniform Push {
    vec4 SourceSize;
    float Cutoff;
    float ICutoff;
    float QCutoff;
    float YIQ;
} params;

#pragma format R16G16B16A16_SFLOAT

#include "parameters_common.inc.slang"

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

#define PI 3.1415926535897932384626433832795
#define CRT_GAMMA 2.4
// Inverse of GAMMA_YIQ_MAT from transform.slang
#define YIQ_GAMMA_MAT mat3x3(1.0, 1.0, 1.0, 0.946882217090069, -0.274787646298978, -1.108545034642032, 0.623556581986143, -0.63569107918738, 1.709006928406467)

vec3 gamma_to_linear(vec3 rgb) {
    // Floating point inaccuracies or out-of-gamut colors from the YIQ
    // to RGB transform could result in negative values, so we'll clamp
    // here to avoid NaNs.
    return pow(clamp(rgb, 0.0, 1.0), vec3(CRT_GAMMA));
}

vec3 yiq_to_linear(vec3 rgb) {
    return gamma_to_linear(YIQ_GAMMA_MAT * rgb);
}

void main() {
    vec3 L = params.YIQ > 0.5 ?
        1.0 / (vec3(params.Cutoff, params.ICutoff, params.QCutoff) * 53.33 * 2.0) :
        1.0 / (vec3(params.Cutoff) * 53.33 * 2.0);
    float L_max = max(L.r, max(L.g, L.b));
    vec3 L_rcp = 1.0 / L;

    vec3 filtered = vec3(0.0);
    int pix_y = int(floor(vTexCoord.y * params.SourceSize.y));
    float t = vTexCoord.x;
    // Bound the start and end pixel so we don't fetch outside our texture
    // (which is undefined behavior). There's no need to fetch there anyway,
    // it would just be zeros.
    for (int pix_x = max(int(floor(params.SourceSize.x * (vTexCoord.x - L_max))), 0);
             pix_x <= min(int(floor(params.SourceSize.x * (vTexCoord.x + L_max))), int(params.SourceSize.x) - 1);
             pix_x++) {
        vec3 s = texelFetch(Source, ivec2(pix_x, pix_y), 0).rgb;
        vec3 t0 = vec3(float(pix_x) * params.SourceSize.z);
        vec3 t1 = t0 + vec3(params.SourceSize.z);
        t0 = clamp(t0, t - L, t + L);
        t1 = clamp(t1, t - L, t + L);
        // Integral of s * (1 / L) * (0.5 + 0.5 * cos(PI * (t - t_x) / L)) dt_x over t0 to t1
        filtered += 0.5 * s * L_rcp * (t1 - t0 + (L / PI) * (sin(L_rcp * ((PI * t) - PI * t0)) - sin(L_rcp * ((PI * t) - PI * t1))));
    }

    FragColor.rgb = params.YIQ > 0.5 ? yiq_to_linear(filtered) : gamma_to_linear(filtered);
}
