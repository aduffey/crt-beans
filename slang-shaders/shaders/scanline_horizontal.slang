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
// The spot is zero outside radius w, which is also its full width at half
// maximum. All distances are in scanline heights. Clamping the endpoints
// extends the primitive constantly outside the spot's support.
// The vertical spot and MaxSpotSize factor are applied in the vertical pass.
vec3 spot_integral(vec3 sample_value, vec3 width, float x, float x0, float x1) {
    vec3 w_rcp = 1.0 / width;
    vec3 u0 = clamp(w_rcp * (x - x0), -1.0, 1.0);
    vec3 u1 = clamp(w_rcp * (x - x1), -1.0, 1.0);
    return sample_value * (cubic_primitive(u0) - cubic_primitive(u1));
}

void main() {
    vec3 energy = vec3(0.0);
    vec3 moment = vec3(0.0);
    float start_x = params.SourceSize.z * (floor(vTexCoord.x * params.SourceSize.x - (params.MaxSpotSize / delta)) + 0.5);
    float stop_x = params.SourceSize.z * (floor(vTexCoord.x * params.SourceSize.x + (params.MaxSpotSize / delta)) + 1.0);
    float x = vTexCoord.x * params.SourceSize.x * delta;
    for (float sample_x = start_x; sample_x < stop_x; sample_x += params.SourceSize.z) {
        vec3 sample_value = textureLod(Source, vec2(sample_x, vTexCoord.y), 0.0).rgb;
        float x0 = delta * (params.SourceSize.x * sample_x - 0.5);
        float x1 = x0 + delta;
        vec3 piece_width = (params.MinSpotSize * params.MaxSpotSize - sqrt(sample_value) * (params.MinSpotSize * params.MaxSpotSize - params.MaxSpotSize));
        vec3 piece_energy = spot_integral(sample_value, piece_width, x, x0, x1);
        energy += piece_energy;
        moment += piece_width * piece_energy;
    }

    vec3 width = min(mix(vec3(params.MinSpotSize * params.MaxSpotSize), moment / energy, notEqual(energy, vec3(0.0))), vec3(params.MaxSpotSize));
    // We actually only need the reciprocal of the width in the scanline pass.
    // To avoid calculating it multiple times later on, we'll just do it here.
    vec3 width_rcp = 1.0 / width;

    // Pack both energy and width_rcp into one texture. This avoids an extra
    // pass (since we don't have multiple render target support) and allows
    // fewer texture samples in the next pass.
    FragColor.rgb = uvec3(
        packHalf2x16(vec2(energy.r, width_rcp.r)),
        packHalf2x16(vec2(energy.g, width_rcp.g)),
        packHalf2x16(vec2(energy.b, width_rcp.b))
    );
}
