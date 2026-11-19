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

#define FORCE_ZOH
#include "scanline_horizontal.inc"

void main() {
    FragColor.rgb = render_scanline_horizontal(vTexCoord, delta);
}
