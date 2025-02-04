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
#include "parameters_analytical.inc.slang"

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

    // The pixel values of the viewport coord (*not* processed for overscan).
    // These should get interpolated as floats with no decimal portion (i.e.
    // integers).
    viewportCoord = floor(TexCoord * params.OutputSize.xy);
}

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 1) flat in float delta;
layout(location = 2) in vec2 viewportCoord;
layout(location = 0) out vec4 FragColor;
layout(binding = 1) uniform sampler2D Source;
layout(binding = 2) uniform sampler2D Linearized;

#define PI 3.1415926535897932384626433832795

vec3 spot_fast(vec3 sample_value, vec3 w_rcp, float x, float x0_, float x1_, float y_) {
    // We need to be careful that we never get 0.0 as w_rcp so that we don't
    // end up with NaNs. That means we can't clamp_to_border on the widths or
    // we'd need to explicitly check for 0.0 here (or fudge it by adding
    // 1.0e-8 or something).
    vec3 w = 1.0 / w_rcp;
    vec3 x0 = clamp(vec3(x0_), x - w, x + w);
    vec3 x1 = clamp(vec3(x1_), x - w, x + w);
    vec3 y = clamp(abs(y_) * w_rcp, 0.0, 1.0);
    return ((y * y) * (2.0 * y - 3.0) + 1.0) * sample_value * w_rcp * w_rcp *
            (x1 - x0 + (w / PI) * (sin(w_rcp * PI * (x - x0)) - sin(w_rcp * PI * (x - x1))));
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
    float upper_sample_y, lower_sample_y;
    // Distance units are *scanlines heights*.
    float upper_distance_y, lower_distance_y;

    // Check if we should be interlacing. If we are, grab two closest even or
    // odd lines. If we aren't, just grab the two closest lines.
    if (params.SourceSize.y > 300.0 && params.OddFieldFirst != 2.0) {  // Interlacing
        if ((params.FrameCount + (params.OddFieldFirst == 1.0 ? 1u : 0u)) % 2u == 0u) {
            upper_sample_y = ceil(0.5 * vTexCoord.y * params.SourceSize.y + 0.25) * 2.0 - 0.5;
        } else {
            upper_sample_y = ceil(0.5 * vTexCoord.y * params.SourceSize.y - 0.25) * 2.0 + 0.5;
        }
        lower_sample_y = upper_sample_y - 2.0;
        // TODO 0.5 doesn't quite work with an odd number of lines. Does that ever happen?
        upper_distance_y = 0.5 * (upper_sample_y - vTexCoord.y * params.SourceSize.y);
        lower_distance_y = upper_distance_y - 1.0;
    } else {  // No interlacing
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
    // TODO check bounds
    float start_x = params.SourceSize.z * (floor(vTexCoord.x * params.SourceSize.x - (params.MaxSpotSize / delta)) + 0.5);
    float stop_x = params.SourceSize.z * (floor(vTexCoord.x * params.SourceSize.x + (params.MaxSpotSize / delta)) + 1.0);
    float x = vTexCoord.x * params.SourceSize.x * delta;
    for (float sample_x = start_x; sample_x < stop_x; sample_x += params.SourceSize.z) {
        vec3 upper_sample = textureLod(Linearized, vec2(sample_x, upper_sample_y), 0.0).rgb;
        vec3 upper_width_rcp = textureLod(Source, vec2(sample_x, upper_sample_y), 0.0).rgb;
        vec3 lower_sample = textureLod(Linearized, vec2(sample_x, lower_sample_y), 0.0).rgb;
        vec3 lower_width_rcp = textureLod(Source, vec2(sample_x, lower_sample_y), 0.0).rgb;
        float x0 = delta * (params.SourceSize.x * sample_x - 0.5);
        float x1 = x0 + delta;
        value += spot_fast(upper_sample, upper_width_rcp, x, x0, x1, upper_distance_y);
        value += spot_fast(lower_sample, lower_width_rcp, x, x0, x1, lower_distance_y);
    }
    value = params.MaxSpotSize * 0.5 * value;

    if (params.MaskEnabled != 0.0) {
        value = mask(value);
    }

    FragColor.rgb = value;
}
