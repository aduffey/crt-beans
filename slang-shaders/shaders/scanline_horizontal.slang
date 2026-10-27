#version 450

// SPDX-License-Identifier: MIT

layout(set = 0, binding = 0, std140) uniform UBO {
    mat4 MVP;
};

layout(push_constant) uniform Push {
    vec4 SourceSize;
    vec4 OutputSize;
    vec4 FinalViewportSize;
    float MaxSpotSize;
    float MinSpotSize;
    float OverscanHorizontal;
    float OverscanVertical;
    float InterlacingType;
} params;

#pragma format R32G32B32A32_UINT

#include "parameters_fast.inc"

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
layout(location = 1) flat out float delta;

void main() {
    gl_Position = MVP * Position;

    // Handle horizontal overscan (vertical overscan is handled in the vertical pass).
    vTexCoord = vec2(
        (1.0 - params.OverscanHorizontal) * (TexCoord.x - 0.5) + 0.5,
        TexCoord.y
    );

    // Distance units are *scanlines heights*. delta is the distance between
    // samples on the x-axis, and has to be adjusted by the aspect ratio.
    // Overscan changes the aspect ratio and needs to be taken into account.
    delta = params.FinalViewportSize.x * params.FinalViewportSize.w *
            params.SourceSize.y * params.SourceSize.z *
            (1 - params.OverscanVertical) / (1 - params.OverscanHorizontal);

    // Adjust delta if we're interlacing.
    if (params.InterlacingType <= 1.0 && params.SourceSize.y > 300.0) {
        // bob or weave
        delta = 0.5 * delta;
    } else if (params.InterlacingType == 2.0 && params.SourceSize.y < 350) {
        // VGA line doubling
        delta = 2.0 * delta;
    }
}

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 1) flat in float delta;
layout(location = 0) out uvec4 FragColor;
layout(binding = 1) uniform sampler2D Source;

// Antiderivative of k(u) = 1 - 3*u^2 + 2*|u|^3 on [-1, 1].
// F(1) - F(-1) = 1, so the horizontal spot has unit total energy.
vec3 cubic_primitive(vec3 u) {
    return u * (1.0 + u * u * (0.5 * abs(u) - 1.0));
}

// Convolve s/w * k((x - t)/w) with a constant texel over [x0, x1].
// dx0 = x - x0 and dx1 = x - x1 are the signed distances from the evaluation point.
// All distances are in scanline heights.
vec3 spot_integral(vec3 sample_value, vec3 width, float dx0, float dx1) {
    vec3 w_rcp = 1.0 / width;
    vec3 u0 = clamp(w_rcp * dx0, -1.0, 1.0);
    vec3 u1 = clamp(w_rcp * dx1, -1.0, 1.0);
    return sample_value * (cubic_primitive(u0) - cubic_primitive(u1));
}

void main() {
    vec3 energy = vec3(0.0);
    vec3 moment = vec3(0.0);

    float min_width = params.MinSpotSize * params.MaxSpotSize;
    float width_range = params.MaxSpotSize - min_width;
    float radius = params.MaxSpotSize / delta;

    float tex_x = vTexCoord.x * params.SourceSize.x;
    int start_i = int(floor(tex_x - radius));
    int stop_i  = int(floor(tex_x + radius));

    for (int i = start_i; i <= stop_i; ++i) {
        float sample_x = (float(i) + 0.5) * params.SourceSize.z;
        vec3 sample_value = textureLod(Source, vec2(sample_x, vTexCoord.y), 0.0).rgb;
        float dx0 = (tex_x - float(i)) * delta;
        float dx1 = dx0 - delta;
        vec3 piece_width = min_width + sqrt(sample_value) * width_range;
        vec3 piece_energy = spot_integral(sample_value, piece_width, dx0, dx1);
        energy += piece_energy;
        moment += piece_width * piece_energy;
    }

    vec3 width = min(mix(vec3(min_width), moment / energy, notEqual(energy, vec3(0.0))), vec3(params.MaxSpotSize));
    vec3 width_rcp = 1.0 / width;

    // Pre-multiply energy with MaxSpotSize * width_rcp to save vec3 multiplications
    // per viewport pixel in the vertical scanline pass.
    vec3 scaled_energy = energy * (params.MaxSpotSize * width_rcp);

    // Pack both scaled_energy and width_rcp into one texture. This avoids an extra
    // pass (since we don't have multiple render target support) and allows
    // fewer texture samples in the next pass.
    FragColor.rgb = uvec3(
        packHalf2x16(vec2(scaled_energy.r, width_rcp.r)),
        packHalf2x16(vec2(scaled_energy.g, width_rcp.g)),
        packHalf2x16(vec2(scaled_energy.b, width_rcp.b))
    );
}
