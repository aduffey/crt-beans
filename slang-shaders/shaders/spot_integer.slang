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
} params;

#pragma format R16G16B16A16_SFLOAT

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
layout(binding = 2) uniform sampler2D Filtered;

vec3 spot(vec3 sample_value, vec3 width_rcp, float distance_x, float distance_y) {
    vec3 x = clamp(abs(distance_x) * width_rcp, 0.0, 1.0);
    vec3 y = clamp(abs(distance_y) * width_rcp, 0.0, 1.0);
    return sample_value * width_rcp * ((x * x) * (2.0 * x - 3.0) + 1.0) * ((y * y) * (2.0 * y - 3.0) + 1.0);
}

void main() {
    // Handle overscan.
    vec2 pos = (1.0 - vec2(params.OverscanHorizontal, params.OverscanVertical)) * (vTexCoord - 0.5) + 0.5;

    // Sample locations in Source texel space: [0, SourceSize.y).
    int lower_sample_y, upper_sample_y;
    // Distance units are *scanlines heights*. delta is the distance between
    // samples on the x-axis, and has to be adjusted by the aspect ratio.
    // Overescan changes the aspect ratio and needs to be taken into account.
    float delta, upper_distance_y, lower_distance_y;
    // Check if we should be interlacing. If we are, grab two closest even or
    // odd lines. If we aren't, just grab the two closest lines.
    if (params.SourceSize.y > 300) {
        if ((params.FrameCount + (params.OddFieldFirst > 0.5 ? 1u : 0u)) % 2u == 0u) {
            upper_sample_y = int(ceil(pos.y * params.SourceSize.y + 0.5) * 0.5) * 2;
        } else {
            upper_sample_y = int(floor(pos.y * params.SourceSize.y + 0.5) * 0.5) * 2 + 1;
        }
        lower_sample_y = upper_sample_y - 2;
        // TODO 0.5 doesn't quite work with an odd number of lines. Does that ever happen?
        delta = 0.5 * params.OutputSize.x * params.OutputSize.w *
                params.SourceSize.y * params.SourceSize.z *
                (1 - params.OverscanVertical) / (1 - params.OverscanHorizontal);
        upper_distance_y = 0.5 * ((upper_sample_y + 0.5) - pos.y * params.SourceSize.y);
        lower_distance_y = upper_distance_y - 1.0;
    } else {
        upper_sample_y = int(round(pos.y * params.SourceSize.y));
        lower_sample_y = upper_sample_y - 1;
        delta = params.OutputSize.x * params.OutputSize.w *
                params.SourceSize.y * params.SourceSize.z *
                (1 - params.OverscanVertical) / (1 - params.OverscanHorizontal);
        upper_distance_y = (upper_sample_y + 0.5) - pos.y * params.SourceSize.y;
        lower_distance_y = upper_distance_y - 1.0;
    }

    vec3 value = vec3(0.0);
    int start_x = max(int(round(pos.x * params.SourceSize.x - (params.MaxSpotSize / delta))), 0);
    int stop_x = min(int(round(pos.x * params.SourceSize.x + (params.MaxSpotSize / delta))), int(params.SourceSize.x));
    float distance_x = delta * ((start_x + 0.5) - pos.x * params.SourceSize.x);
    if (upper_sample_y >= params.SourceSize.y) {
        for (int sample_x = start_x; sample_x < stop_x; sample_x++) {
            vec3 lower_sample = texelFetch(Filtered, ivec2(sample_x, lower_sample_y), 0).rgb;
            vec3 lower_width_rcp = texelFetch(Source, ivec2(sample_x, lower_sample_y), 0).rgb;
            value += spot(lower_sample, lower_width_rcp, distance_x, lower_distance_y);
            distance_x += delta;
        }
    } else if (lower_sample_y < 0) {
        for (int sample_x = start_x; sample_x < stop_x; sample_x++) {
            vec3 upper_sample = texelFetch(Filtered, ivec2(sample_x, upper_sample_y), 0).rgb;
            vec3 upper_width_rcp = texelFetch(Source, ivec2(sample_x, upper_sample_y), 0).rgb;
            value += spot(upper_sample, upper_width_rcp, distance_x, upper_distance_y);
            distance_x += delta;
        }
    } else {
        for (int sample_x = start_x; sample_x < stop_x; sample_x++) {
            vec3 upper_sample = texelFetch(Filtered, ivec2(sample_x, upper_sample_y), 0).rgb;
            vec3 upper_width_rcp = texelFetch(Source, ivec2(sample_x, upper_sample_y), 0).rgb;
            vec3 lower_sample = texelFetch(Filtered, ivec2(sample_x, lower_sample_y), 0).rgb;
            vec3 lower_width_rcp = texelFetch(Source, ivec2(sample_x, lower_sample_y), 0).rgb;
            value += spot(upper_sample, upper_width_rcp, distance_x, upper_distance_y);
            value += spot(lower_sample, lower_width_rcp, distance_x, lower_distance_y);
            distance_x += delta;
        }
    }
    FragColor = vec4(delta * value, 1.0);
}
