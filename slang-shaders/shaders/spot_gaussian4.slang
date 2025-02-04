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
    float MaskEnabled;
    float SubpixelMaskPattern;
    float SubpixelPattern;
} params;

#pragma format R16G16B16A16_SFLOAT

#include "parameters_common.inc.slang"
#include "parameters_sampled.inc.slang"

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
layout(location = 1) flat out float delta;
layout(location = 2) flat out float maxSigma;
layout(location = 3) flat out float scaleFactor;
layout(location = 4) out vec2 viewportCoord;

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
    // This should be the maximum value of four overlapping gaussian scanlines
    // given the maximum spot size. One scanline has a vertical offset of 0,
    // two have a vertical offset of 1, and one has a vertical offset of 2.
    // We are centered on the first upper or lower scanline, which will
    // be the point of maximum value. Technically, the peak shifts slightly
    // off of the center of the scanline with higher values of maxSigma, but
    // this is close enough if we keep maxSigma <= 1.2.
    //
    // This is the integral over -infinity to infinity with respect to x of:
    //         1 / (2 * maxSigma**2) * exp(-(x**2 + 0**2) / (2 * maxSigma**2))
    //   + 2 * 1 / (2 * maxSigma**2) * exp(-(x**2 + 1**2) / (2 * maxSigma**2))
    //   +     1 / (2 * maxSigma**2) * exp(-(x**2 + 2**2) / (2 * maxSigma**2))
    //
    // This is derived using un-truncated gaussians, so it will be a slight
    // overestimate because we need to truncate the gaussians in practice.
    scaleFactor = sqrt(PI / 2.0) * abs(maxSigma) / (maxSigma * maxSigma) * exp(-2.0 / (maxSigma * maxSigma)) *
            (2.0 * exp(3.0 / (2.0 * maxSigma * maxSigma)) + exp(2.0 / (maxSigma * maxSigma)) + 1.0);
    scaleFactor = 1.0 / scaleFactor;

    // The pixel values of the viewport coord (*not* processed for overscan).
    // These should get interpolated as floats with no decimal portion (i.e.
    // integers).
    viewportCoord = floor(TexCoord * params.OutputSize.xy);
}

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 1) flat in float delta;
layout(location = 2) flat in float maxSigma;
layout(location = 3) flat in float scaleFactor;
layout(location = 4) in vec2 viewportCoord;
layout(location = 0) out vec4 FragColor;
layout(binding = 1) uniform sampler2D Source;
layout(binding = 2) uniform sampler2D Filtered;

// A representation of a CRT spot in 2 dimensions, with the spot centered at
// (0, 0). sigma_rcp is 1 / (2 * sigma * sigma);
vec3 spot(vec3 sample_value, vec3 sigma_rcp, float distance_x, float distance_y) {
    return sample_value * sigma_rcp * exp(-(distance_x * distance_x + distance_y * distance_y) * sigma_rcp);
}

vec3 mask(vec3 pixel_value) {
    // We want to use viewportCoord here, not vTexCoord, which is processed
    // for overscan.
    vec3 mask = vec3(1.0);
    float mask_coverage = 1.0;
    if (params.SubpixelPattern == 0.0) {  // RGB subpixels
        if (params.SubpixelMaskPattern == 2.0) {  // <=1080p
            vec3[2] mask_tile = { vec3(1.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0) };
            mask = mask_tile[int(mod(viewportCoord.x, 2.0))];
            mask_coverage = 2.0;
        } else if (params.SubpixelMaskPattern == 3.0) {  // 1080p/1440p
            vec3[3] mask_tile = { vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0) };
            mask = mask_tile[int(mod(viewportCoord.x, 3.0))];
            mask_coverage = 3.0;
        } else if (params.SubpixelMaskPattern == 4.0) {  // 1440p/4k
            vec3[4] mask_tile = { vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 0.0), vec3(0.0, 1.0, 1.0), vec3(0.0, 0.0, 1.0) };
            mask = mask_tile[int(mod(viewportCoord.x, 4.0))];
            mask_coverage = 2.0;
        } else if (params.SubpixelMaskPattern == 5.0) {  // 4k, lower TVL
            vec3[5] mask_tile = { vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 1.0), vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0) };
            mask = mask_tile[int(mod(viewportCoord.x, 5.0))];
            mask_coverage = 15.0 / 6.0;
        }
    } else {  // BGR subpixels, same as above but red and blue values transposed
        if (params.SubpixelMaskPattern == 2.0) {  // <=1080p
            vec3[2] mask_tile = { vec3(1.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0) };
            mask = mask_tile[int(mod(viewportCoord.x, 2.0))];
            mask_coverage = 2.0;
        } else if (params.SubpixelMaskPattern == 3.0) {  // 1080p/1440p
            vec3[3] mask_tile = { vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0) };
            mask = mask_tile[int(mod(viewportCoord.x, 3.0))];
            mask_coverage = 3.0;
        } else if (params.SubpixelMaskPattern == 4.0) {  // 1440p/4k
            vec3[4] mask_tile = { vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0) };
            mask = mask_tile[int(mod(viewportCoord.x, 4.0))];
            mask_coverage = 2.0;
        } else if (params.SubpixelMaskPattern == 5.0) {  // 4k, lower TVL
            vec3[5] mask_tile = { vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0) };
            mask = mask_tile[int(mod(viewportCoord.x, 5.0))];
            mask_coverage = 15.0 / 6.0;
        }
    }

    // Cubic phase-in. Keeps the mask strength higher for longer than linear
    // but has no discontinuity like piecewise.
    float s = mask_coverage / (mask_coverage - 1.0);
    float a = -s + 2.0;
    float b = s - 3.0;
    vec3 weight = a * (pixel_value * pixel_value * pixel_value) + b * (pixel_value * pixel_value) + 1.0;
    return pixel_value - pixel_value * weight * (1.0 - mask_coverage * mask);
}

void main() {
    // Sample locations in Source texel space: [0, SourceSize.y), snapped to the
    // middle of a texel.
    float upper1_sample_y, upper2_sample_y;
    // Distance units are *scanlines heights*.
    float upper1_distance_y, upper2_distance_y;

    // Check if we should be interlacing. If we are, grab two upper even or
    // odd lines. If we aren't, just grab the two upper lines.
    if (params.SourceSize.y > 300.0) {  // Interlacing
        if ((params.FrameCount + (params.OddFieldFirst > 0.5 ? 1u : 0u)) % 2u == 0u) {
            upper1_sample_y = ceil(0.5 * vTexCoord.y * params.SourceSize.y + 0.25) * 2.0 - 0.5;
        } else {
            upper1_sample_y = ceil(0.5 * vTexCoord.y * params.SourceSize.y - 0.25) * 2.0 + 0.5;
        }
        upper2_sample_y = upper1_sample_y + 1.0;
        // TODO 0.5 doesn't quite work with an odd number of lines. Does that ever happen?
        upper1_distance_y = 0.5 * (upper1_sample_y - vTexCoord.y * params.SourceSize.y);
        upper2_distance_y = upper1_distance_y + 1.0;
    } else {  // No interlacing
        upper1_sample_y = round(vTexCoord.y * params.SourceSize.y) + 0.5;
        upper2_sample_y = upper1_sample_y + 1.0;
        upper1_distance_y = upper1_sample_y - vTexCoord.y * params.SourceSize.y;
        upper2_distance_y = upper1_distance_y + 1.0;
    }

    // Transform *_sample_y into [0,1] coordinate space. We need them for the
    // texture sampling below.
    upper1_sample_y *= params.SourceSize.w;
    upper2_sample_y *= params.SourceSize.w;

    vec3 value = vec3(0.0);
    float start_x = params.SourceSize.z * (round(vTexCoord.x * params.SourceSize.x - (3.0 * maxSigma / delta)) + 0.5);
    float stop_x = params.SourceSize.z * round(vTexCoord.x * params.SourceSize.x + (3.0 * maxSigma / delta));
    float distance_x = delta * params.SourceSize.x * (start_x - vTexCoord.x);
    for (float sample_x = start_x; sample_x < stop_x; sample_x += params.SourceSize.z) {
        vec3 upper1_sample = textureLod(Filtered, vec2(sample_x, upper1_sample_y), 0.0).rgb;
        vec3 upper1_sigma_rcp = textureLod(Source, vec2(sample_x, upper1_sample_y), 0.0).rgb;
        vec3 upper2_sample = textureLod(Filtered, vec2(sample_x, upper2_sample_y), 0.0).rgb;
        vec3 upper2_sigma_rcp = textureLod(Source, vec2(sample_x, upper2_sample_y), 0.0).rgb;
        value += spot(upper1_sample, upper1_sigma_rcp, distance_x, upper1_distance_y);
        value += spot(upper2_sample, upper2_sigma_rcp, distance_x, upper2_distance_y);
        distance_x += delta;
    }

    // Second set of scanlines. Do essentially the same thing as above, but for
    // the lower two scanlines. By splitting this into two sets of two
    // scanlines, we can reduce register usage (increasing occupancy) and get
    // better cache hit ratios. The tradeoff is slightly higher ALU usage as we
    // iterate through the loop again.
    float lower1_sample_y, lower2_sample_y;
    float lower1_distance_y, lower2_distance_y;

    if (params.SourceSize.y > 300.0) {  // Interlacing
        lower1_sample_y = upper1_sample_y - 2.0 * params.SourceSize.w;
        lower2_sample_y = lower1_sample_y - 2.0 * params.SourceSize.w;
    } else {  // No interlacing
        lower1_sample_y = upper1_sample_y - 1.0 * params.SourceSize.w ;
        lower2_sample_y = lower1_sample_y - 1.0 * params.SourceSize.w;
    }
    lower1_distance_y = upper1_distance_y - 1.0;
    lower2_distance_y = lower1_distance_y - 1.0;

    distance_x = delta * params.SourceSize.x * (start_x - vTexCoord.x);
    for (float sample_x = start_x; sample_x < stop_x; sample_x += params.SourceSize.z) {
        vec3 lower1_sample = textureLod(Filtered, vec2(sample_x, lower1_sample_y), 0.0).rgb;
        vec3 lower1_sigma_rcp = textureLod(Source, vec2(sample_x, lower1_sample_y), 0.0).rgb;
        vec3 lower2_sample = textureLod(Filtered, vec2(sample_x, lower2_sample_y), 0.0).rgb;
        vec3 lower2_sigma_rcp = textureLod(Source, vec2(sample_x, lower2_sample_y), 0.0).rgb;
        value += spot(lower1_sample, lower1_sigma_rcp, distance_x, lower1_distance_y);
        value += spot(lower2_sample, lower2_sigma_rcp, distance_x, lower2_distance_y);
        distance_x += delta;
    }
    value = delta * scaleFactor * value;

    if (params.MaskEnabled != 0.0) {
        value = mask(value);
    }

    FragColor.rgb = value;
}
