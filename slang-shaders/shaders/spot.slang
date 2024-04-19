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
} params;

#pragma parameter MaxSpotSize "Maximum spot size (proportion of scanline)" 1.0 0.6 1.0 0.05
#pragma parameter MinSpotSize "Minimum spot size (proportion of maximum)" 0.5 0.3 1.0 0.1
#pragma parameter OddFieldFirst "Interlacing (phase)" 0.0 0.0 1.0 1.0

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

#define PI 3.1415926535897932384626433832795

vec3 linear_to_srgb(vec3 rgb) {
    bvec3 cutoff = lessThan(rgb, vec3(0.0031308));
    // Floating point inaccuracies could result in negative values, so we'll
    // clamp here to avoid NaNs.
    vec3 higher = vec3(1.055) * pow(clamp(rgb, 0.0, 1.0), vec3(1.0 / 2.4)) - vec3(0.055);
    vec3 lower = rgb * vec3(12.92);

    return mix(higher, lower, cutoff);
}

vec3 spot(vec3 sample_value, float distance_x, float distance_y) {
    vec3 width_rcp = 1.0 / mix(vec3(params.MinSpotSize), vec3(params.MaxSpotSize), sqrt(sample_value));
    vec3 x = clamp(abs(distance_x) * width_rcp, 0.0, 1.0);
    vec3 y = clamp(abs(distance_y) * width_rcp, 0.0, 1.0);
    return sample_value * width_rcp * ((x * x) * (2.0 * x - 3.0) + 1.0) * ((y * y) * (2.0 * y - 3.0) + 1.0);
}

void main() {
    // Sample locations in Source texel space: [0, SourceSize.y), snapped to the
    // middle of a texel.
    float upper_sample_y, lower_sample_y;
    // Distance units are *scanlines heights*. delta is the distance
    // between samples on the x-axis, and has to be adjusted by the
    // aspect ratio.
    float delta, upper_distance_y;
    // Check if we should be interlacing. If we are, grab two closest even or
    // odd lines. If we aren't, just grab the two closest lines.
    if (params.SourceSize.y > 300.0) {  // Interlacing
        if ((params.FrameCount + (params.OddFieldFirst > 0.5 ? 1u : 0u)) % 2u == 0u) {
            upper_sample_y = floor(ceil(vTexCoord.y * params.SourceSize.y + 0.5) * 0.5) * 2.0 + 0.5;
        } else {
            upper_sample_y = floor(floor(vTexCoord.y * params.SourceSize.y + 0.5) * 0.5) * 2.0 + 1.5;
        }
        lower_sample_y = upper_sample_y - 2.0;
        // TODO 0.5 doesn't quite work with an odd number of lines. Does that ever happen?
        delta = 0.5 * params.OutputSize.x * params.OutputSize.w * params.SourceSize.y * params.SourceSize.z;
        upper_distance_y = 0.5 * (upper_sample_y - vTexCoord.y * params.SourceSize.y);
    } else {  // No interlacing
        upper_sample_y = round(vTexCoord.y * params.SourceSize.y) + 0.5;
        lower_sample_y = upper_sample_y - 1.0;
        delta = params.OutputSize.x * params.OutputSize.w * params.SourceSize.y * params.SourceSize.z;
        upper_distance_y = upper_sample_y - vTexCoord.y * params.SourceSize.y;
    }
    float lower_distance_y = upper_distance_y - 1.0;

    // Transform *_sample_y into [0,1] coordinate space. We need them for the
    // texture sampling below.
    upper_sample_y *= params.SourceSize.w;
    lower_sample_y *= params.SourceSize.w;

    vec3 value = vec3(0.0);
    for (float sample_x = params.SourceSize.z * (round(vTexCoord.x * params.SourceSize.x - (params.MaxSpotSize / delta)) + 0.5);
               sample_x < params.SourceSize.z * round(vTexCoord.x * params.SourceSize.x + (params.MaxSpotSize / delta));
               sample_x += params.SourceSize.z) {
        vec3 upper_sample = texture(Source, vec2(sample_x, upper_sample_y)).rgb;
        vec3 lower_sample = texture(Source, vec2(sample_x, lower_sample_y)).rgb;
        float distance_x = delta * params.SourceSize.x * (sample_x - vTexCoord.x);
        value += spot(upper_sample, distance_x, upper_distance_y);
        value += spot(lower_sample, distance_x, lower_distance_y);
    }
    FragColor = vec4(linear_to_srgb(delta * value), 1.0);
}
