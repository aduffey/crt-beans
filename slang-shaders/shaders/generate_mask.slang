#version 450

layout(set = 0, binding = 0, std140) uniform UBO {
    mat4 MVP;
};

layout(push_constant) uniform Push {
    vec4 OutputSize;
    float MaskType;
    float DynamicMaskTriads;
} params;

#pragma format R16G16B16A16_SFLOAT

#include "parameters_common.inc.slang"

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

void main() {
    if (params.MaskType == 2.0) {
        FragColor = dynamic_aperture_grille(vTexCoord, params.OutputSize, params.DynamicMaskTriads);
    }
}
