#version 450

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

/* This is a 2x downscale using a 4x4 Lanczos-1 kernel and a few tricks for
 * performance.
 *
 * - The Lanczos kernel is estimated with an 8th order, even polynomial. This
 *   avoids having to do expensive sine and division operations.
 * - Bilinear filtering tricks are used to reduce the amount of texture samples
 *   required from 16 to 4, at the expense of a bit of ALU usage.
 *   See: https://vec3.ca/bicubic-filtering-in-fewer-taps
 *
 * Ideally, we would scale the kernel when the input texture is not evenly
 * divisible by 2, but we don't need to be that exact for this.
 */
void main() {
    vec2 pos = vTexCoord * params.SourceSize.xy;
    vec2 center = floor(pos - 0.5) + 0.5;

    // Our polynomial approximation can return small negative values at the
    // extremes, so we'll clamp it.
    vec2 x2 = pos - center + 1.0;
    x2 = x2 * x2;
    vec2 weight0 = clamp((((0.00183565625 * x2 - 0.03356125) * x2 + 0.242894375) * x2 - 0.80208) * x2 + 1.0, 0.0, 1.0);
    x2 = pos - center;
    x2 = x2 * x2;
    vec2 weight1 = clamp((((0.00183565625 * x2 - 0.03356125) * x2 + 0.242894375) * x2 - 0.80208) * x2 + 1.0, 0.0, 1.0);
    x2 = pos - center - 1.0;
    x2 = x2 * x2;
    vec2 weight2 = clamp((((0.00183565625 * x2 - 0.03356125) * x2 + 0.242894375) * x2 - 0.80208) * x2 + 1.0, 0.0, 1.0);
    x2 = pos - center - 2.0;
    x2 = x2 * x2;
    vec2 weight3 = clamp((((0.00183565625 * x2 - 0.03356125) * x2 + 0.242894375) * x2 - 0.80208) * x2 + 1.0, 0.0, 1.0);

    vec2 scale0 = weight0 + weight1;
    vec2 scale1 = weight2 + weight3;

    vec2 coord0 = (center - 1.0 + weight1 / scale0) * params.SourceSize.zw;
    vec2 coord1 = (center + 1.0 + weight3 / scale1) * params.SourceSize.zw;

    vec3 value = (textureLod(Source, vec2(coord0.x, coord0.y), 0.0).rgb * scale0.x +
                  textureLod(Source, vec2(coord1.x, coord0.y), 0.0).rgb * scale1.x) * scale0.y +
                 (textureLod(Source, vec2(coord0.x, coord1.y), 0.0).rgb * scale0.x +
                  textureLod(Source, vec2(coord1.x, coord1.y), 0.0).rgb * scale1.x) * scale1.y;
    vec2 weight_sum = scale0 + scale1;
    value /= (weight_sum.x * weight_sum.y);
    FragColor = vec4(value, 1.0);
}
