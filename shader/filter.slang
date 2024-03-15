#version 450

layout(set = 0, binding = 0, std140) uniform UBO
{
   mat4 MVP;
};

layout(push_constant) uniform Push
{
   vec4 SourceSize;
   float Cutoff;
   float ICutoff;
   float QCutoff;
   float YIQ;
} params;

#pragma format R16G16B16A16_SFLOAT
#pragma parameter YIQ "Composite" 0.0 0.0 1.0 1.0
#pragma parameter Cutoff "RGB/Y bandwidth (MHz)" 5.0 0.6 6.0 0.2
#pragma parameter ICutoff "I bandwidth (MHz)" 0.6 0.6 6.0 0.2
#pragma parameter QCutoff "Q bandwidth (MHz)" 0.6 0.6 6.0 0.2

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

#define PI 3.1415926535897932384626433832795
#define CRT_GAMMA 2.4
// Inverse of GAMMA_YIQ_MAT from transform.slang
#define YIQ_GAMMA_MAT mat3x3(1.0, 1.0, 1.0, 0.946882217090069, -0.274787646298978, -1.108545034642032, 0.623556581986143, -0.63569107918738, 1.709006928406467)

vec3 gamma_to_linear(vec3 rgb) {
    return pow(rgb, vec3(CRT_GAMMA));
}

vec3 yiq_to_linear(vec3 rgb) {
    // Theoretically, it is possible for a YIQ -> RGB transformation to result
    // in out-of-gamut colors, so we'll clamp here. There may also be floating
    // point inaccuracies.
    return gamma_to_linear(clamp(YIQ_GAMMA_MAT * rgb, 0.0, 1.0));
}

void main()
{
    const vec3 L = params.YIQ > 0.5 ?
        1.0 / (vec3(params.Cutoff, params.ICutoff, params.QCutoff) * 53.33 * 2.0) :
        1.0 / (vec3(params.Cutoff) * 53.33 * 2.0);
    const float max_L = max(L.r, max(L.g, L.b));

    vec3 filtered = vec3(0.0);
    int pix_y = int(floor(vTexCoord.y * params.SourceSize.y));
    float t = vTexCoord.x;
    for (int pix_x = int(floor(params.SourceSize.x * (vTexCoord.x - max_L)));
             pix_x <= int(floor(params.SourceSize.x * (vTexCoord.x + max_L)));
             pix_x++) {
        vec3 s = texelFetch(Source, ivec2(pix_x, pix_y), 0).rgb;
        vec3 t0 = vec3(pix_x * params.SourceSize.z);
        vec3 t1 = vec3(t0 + params.SourceSize.z);
        t0 = clamp(t0, t - L, t + L);
        t1 = clamp(t1, t - L, t + L);
        // Integral of s * (1 / L) * (0.5 + 0.5 * cos(PI * (t - t_x) / L)) dt_x over t0 to t1
        filtered += s / (2.0 * L) * (t1 - t0 + (L / PI) * (sin((PI / L) * (t - t0)) - sin((PI / L) * (t - t1))));
    }

    if (params.YIQ > 0.5) {
        FragColor = vec4(yiq_to_linear(filtered), 1.0);
    } else {
        FragColor = vec4(gamma_to_linear(filtered), 1.0);
    }
}
