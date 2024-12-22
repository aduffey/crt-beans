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

#include "parameters_common.inc.slang"
#include "parameters_advanced.inc.slang"

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
layout(location = 1) out float delta;
layout(location = 2) out float maxSigma;
layout(location = 3) out float scaleFactor;

#define PI 3.1415926535897932384626433832795

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

    // The sigma at the largest spot size. We need this to know when to
    // truncate the gaussian in the fragment shader, as well as find the peak
    // value below.
    // 2 * sqrt(2 * log(2)) * sigma = full width at half maximum of a gaussian.
    maxSigma = params.MaxSpotSize / (2 * sqrt(2 * log(2)));  // TODO constant?

    // We want to rescale the output so that the maximum value is one.
    // This should be the maximum value of three overlapping gaussian scanlines
    // given the maximum spot size. One scanline has a vertical offset of 0 and
    // the other two have a vertical offset of 1 (i.e. we are centered on the
    // middle scanline, which will be the point of maximum value).
    //
    // This is the integral over -infinity to infinity with respect to x of:
    //         1 / (2 * maxSigma**2) * exp(-(x**2 + 0**2) / (2 * maxSigma**2))
    //   + 2 * 1 / (2 * maxSigma**2) * exp(-(x**2 + 1**2) / (2 * maxSigma**2))
    //
    // This is derived using un-truncated gaussians, so it will be a slight
    // overestimate because we need to truncate the gaussians in practice.
    scaleFactor = sqrt(PI / 2.0) / abs(maxSigma) * (1.0 + 2.0 * exp(-1.0 / (2.0 * maxSigma * maxSigma)));
    scaleFactor = 1.0 / scaleFactor;
}

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 1) in float delta;
layout(location = 2) in float maxSigma;
layout(location = 3) in float scaleFactor;
layout(location = 0) out vec4 FragColor;
layout(binding = 1) uniform sampler2D Source;
layout(binding = 2) uniform sampler2D Filtered;

// A representation of a CRT spot in 2 dimensions, with the spot centered at
// (0, 0). sigma_rcp is 1 / (2 * sigma * sigma);
vec3 spot(vec3 sample_value, vec3 sigma_rcp, float distance_x, float distance_y) {
    return sample_value * sigma_rcp * exp(-(distance_x * distance_x + distance_y * distance_y) * sigma_rcp);
}

void main() {
    // Sample locations in Source texel space: [0, SourceSize.y), snapped to the
    // middle of a texel.
    float middle_sample_y, lower_sample_y, upper_sample_y;
    // Distance units are *scanlines heights*.
    float middle_distance_y, lower_distance_y, upper_distance_y;

    // Check if we should be interlacing. If we are, grab two closest even or
    // odd lines. If we aren't, just grab the two closest lines.
    if (params.SourceSize.y > 300.0) {  // Interlacing
        if ((params.FrameCount + (params.OddFieldFirst > 0.5 ? 1u : 0u)) % 2u == 0u) {
            middle_sample_y = floor(0.5 * vTexCoord.y * params.SourceSize.y + 0.25) * 2.0 + 0.5;
        } else {
            middle_sample_y = floor(0.5 * vTexCoord.y * params.SourceSize.y - 0.25) * 2.0 + 1.5;
        }
        lower_sample_y = middle_sample_y - 2.0;
        upper_sample_y =  middle_sample_y + 2.0;
        // TODO 0.5 doesn't quite work with an odd number of lines. Does that ever happen?
        middle_distance_y = 0.5 * (middle_sample_y - vTexCoord.y * params.SourceSize.y);
        lower_distance_y = middle_distance_y - 1.0;
        upper_distance_y = middle_distance_y + 1.0;
    } else {  // No interlacing
        middle_sample_y = floor(vTexCoord.y * params.SourceSize.y) + 0.5;
        lower_sample_y = middle_sample_y - 1.0;
        upper_sample_y =  middle_sample_y + 1.0;
        middle_distance_y = middle_sample_y - vTexCoord.y * params.SourceSize.y;
        lower_distance_y = middle_distance_y - 1.0;
        upper_distance_y = middle_distance_y + 1.0;
    }

    // Transform *_sample_y into [0,1] coordinate space. We need them for the
    // texture sampling below.
    middle_sample_y *= params.SourceSize.w;
    lower_sample_y *= params.SourceSize.w;
    upper_sample_y *= params.SourceSize.w;

    vec3 value = vec3(0.0);
    float start_x = params.SourceSize.z * (round(vTexCoord.x * params.SourceSize.x - (3.0 * maxSigma / delta)) + 0.5);
    float stop_x = params.SourceSize.z * round(vTexCoord.x * params.SourceSize.x + (3.0 * maxSigma / delta));
    float distance_x = delta * params.SourceSize.x * (start_x - vTexCoord.x);
    for (float sample_x = start_x; sample_x < stop_x; sample_x += params.SourceSize.z) {
        vec3 middle_sample = textureLod(Filtered, vec2(sample_x, middle_sample_y), 0.0).rgb;
        vec3 middle_sigma_rcp = textureLod(Source, vec2(sample_x, middle_sample_y), 0.0).rgb;
        vec3 lower_sample = textureLod(Filtered, vec2(sample_x, lower_sample_y), 0.0).rgb;
        vec3 lower_sigma_rcp = textureLod(Source, vec2(sample_x, lower_sample_y), 0.0).rgb;
        vec3 upper_sample = textureLod(Filtered, vec2(sample_x, upper_sample_y), 0.0).rgb;
        vec3 upper_sigma_rcp = textureLod(Source, vec2(sample_x, upper_sample_y), 0.0).rgb;
        value += spot(middle_sample, middle_sigma_rcp, distance_x, middle_distance_y);
        value += spot(lower_sample, lower_sigma_rcp, distance_x, lower_distance_y);
        value += spot(upper_sample, upper_sigma_rcp, distance_x, upper_distance_y);
        distance_x += delta;
    }
    FragColor.rgb = delta * scaleFactor * value;
}
