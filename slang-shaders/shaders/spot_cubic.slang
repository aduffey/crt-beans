#version 450

layout(set = 0, binding = 0, std140) uniform UBO {
    mat4 MVP;
};

layout(push_constant) uniform Push {
    vec4 SourceSize;
    vec4 OutputSize;
    uint FrameCount;
    float MaxSpotSize;
    float MinSpotSize;
    float OddFieldFirst;
    float OverscanHorizontal;
    float OverscanVertical;
    float MaskType;
    float SubpixelMaskPattern;
    float SubpixelPattern;
    float DynamicMaskTriads;
} params;

#pragma format R16G16B16A16_SFLOAT

#include "parameters_common.inc.slang"
#include "parameters_sampled.inc.slang"
#include "mask.inc.slang"

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
layout(location = 1) flat out float delta;
layout(location = 2) out vec2 viewportCoord;

void main() {
    gl_Position = MVP * Position;

    // Handle overscan.
    vTexCoord = (1.0 - vec2(params.OverscanHorizontal, params.OverscanVertical)) * (TexCoord - 0.5) + 0.5;

    // Distance units are *scanlines heights*. delta is the distance between
    // samples on the x-axis, and has to be adjusted by the aspect ratio.
    // Overscan changes the aspect ratio and needs to be taken into account.
    delta = params.OutputSize.x * params.OutputSize.w *
            params.SourceSize.y * params.SourceSize.z *
            (1 - params.OverscanVertical) / (1 - params.OverscanHorizontal);

    // Adjust delta if we're interlacing.
    delta = params.SourceSize.y > 300.0 ? 0.5 * delta : delta;

    // The viewport coordinates (*not* processed for overscan).
    // We need this for the mask.
    viewportCoord = TexCoord;
}

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 1) flat in float delta;
layout(location = 2) in vec2 viewportCoord;
layout(location = 0) out vec4 FragColor;
layout(binding = 1) uniform sampler2D Source;
layout(binding = 2) uniform sampler2D Filtered;

vec3 spot(vec3 sample_value, vec3 width_rcp, float distance_x, float distance_y) {
    vec3 x = clamp(abs(distance_x) * width_rcp, 0.0, 1.0);
    vec3 y = clamp(abs(distance_y) * width_rcp, 0.0, 1.0);
    return sample_value * width_rcp * width_rcp * ((x * x) * (2.0 * x - 3.0) + 1.0) * ((y * y) * (2.0 * y - 3.0) + 1.0);
}

void main() {
    // Sample locations in Source texel space: [0, SourceSize.y), snapped to the
    // middle of a texel.
    float upper_sample_y, lower_sample_y;
    // Distance units are *scanlines heights*.
    float upper_distance_y, lower_distance_y;

    // Check if we should be interlacing. If we are, grab two closest even or
    // odd lines. If we aren't, just grab the two closest lines.
    if (params.SourceSize.y > 300.0) { // Interlacing
        if ((params.FrameCount + (params.OddFieldFirst > 0.5 ? 1u : 0u)) % 2u == 0u) {
            upper_sample_y = ceil(0.5 * vTexCoord.y * params.SourceSize.y + 0.25) * 2.0 - 0.5;
        } else {
            upper_sample_y = ceil(0.5 * vTexCoord.y * params.SourceSize.y - 0.25) * 2.0 + 0.5;
        }
        lower_sample_y = upper_sample_y - 2.0;
        // TODO 0.5 doesn't quite work with an odd number of lines. Does that ever happen?
        upper_distance_y = 0.5 * (upper_sample_y - vTexCoord.y * params.SourceSize.y);
        lower_distance_y = upper_distance_y - 1.0;
    } else { // No interlacing
        upper_sample_y = round(vTexCoord.y * params.SourceSize.y) + 0.5;
        lower_sample_y = upper_sample_y - 1.0;
        upper_distance_y = upper_sample_y - vTexCoord.y * params.SourceSize.y;
        lower_distance_y = upper_distance_y - 1.0;
    }

    // Transform *_sample_y into [0,1] coordinate space. We need them for the
    // texture sampling below.
    upper_sample_y *= params.SourceSize.w;
    lower_sample_y *= params.SourceSize.w;

    vec3 value = vec3(0.0);
    float start_x = params.SourceSize.z * (round(vTexCoord.x * params.SourceSize.x - (params.MaxSpotSize / delta)) + 0.5);
    float stop_x = params.SourceSize.z * round(vTexCoord.x * params.SourceSize.x + (params.MaxSpotSize / delta));
    float distance_x = delta * params.SourceSize.x * (start_x - vTexCoord.x);
    for (float sample_x = start_x; sample_x < stop_x; sample_x += params.SourceSize.z) {
        vec3 upper_sample = textureLod(Filtered, vec2(sample_x, upper_sample_y), 0.0).rgb;
        vec3 upper_width_rcp = textureLod(Source, vec2(sample_x, upper_sample_y), 0.0).rgb;
        vec3 lower_sample = textureLod(Filtered, vec2(sample_x, lower_sample_y), 0.0).rgb;
        vec3 lower_width_rcp = textureLod(Source, vec2(sample_x, lower_sample_y), 0.0).rgb;
        value += spot(upper_sample, upper_width_rcp, distance_x, upper_distance_y);
        value += spot(lower_sample, lower_width_rcp, distance_x, lower_distance_y);
        distance_x += delta;
    }
    value = delta * params.MaxSpotSize * value;

    if (params.MaskType == 1.0) {
        vec4 mask = subpixel_aperture_grille(viewportCoord, params.OutputSize, params.SubpixelPattern, params.SubpixelMaskPattern);
        value = blend_mask(mask.rgb, value, mask.a);
    } else if (params.MaskType == 2.0) {
        vec4 mask = dynamic_aperture_grille(viewportCoord, params.OutputSize, params.DynamicMaskTriads);
        value = blend_mask(mask.rgb, value, mask.a);
    } // Otherwise, no mask

    FragColor.rgb = value;
}
