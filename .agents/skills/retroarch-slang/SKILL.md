---
name: retroarch-slang
description: Guidance for developing, refactoring, and debugging RetroArch Slang shaders (.slang) and presets (.slangp). Use when authoring multi-pass Vulkan GLSL shaders, configuring filter chains, managing push constants, or troubleshooting runtime shader reflection.
---

# RetroArch Slang Shaders

This skill provides comprehensive technical guidance for developing, refactoring, and debugging shaders for the RetroArch **Slang** pipeline (Vulkan GLSL translated to target graphics APIs via SPIR-V Cross) and **librashader**.

---

## Quick Reference: Core Architecture

- **Language Target**: Vulkan GLSL (`#version 450` default; `#version 310 es` for mobile).
- **File Structure**: Each `.slang` file houses both vertex and fragment stages divided by `#pragma stage vertex` and `#pragma stage fragment`.
- **Descriptor Layout**: All resources must use descriptor set `0` (`set = 0`) with explicit binding indices (`binding = N`).
- **UBO (Binding 0)**: Maximum **one** UBO block per shader, reserved for the Model-View-Projection matrix (`mat4 MVP;`).
- **Push Constants**: Maximum **one** push constant block per shader, strictly capped at **128 bytes** (32 floats / 8 `vec4`s).
- **Samplers**: `sampler2D` can only be bound and sampled in the fragment stage.
- **Preset Graph (`.slangp`)**: Multi-pass configuration specifying pass order (`shaderN`), scaling (`scale_typeN`), texture filtering (`filter_linearN`), wrap modes (`wrap_modeN`), render target formats (`float_framebufferN`), and pass aliases (`aliasN`).

---

## Standard Development Workflow

```
1. Design Graph ───> 2. Author .slang ───> 3. Configure .slangp ───> 4. Validate & Budget Check
   (Pass order &       (Stages, UBO,          (Pass count, scales,     (<= 128B push block,
    color space)        Push block, math)      aliases, LUTs)           reflection parity)
```

1. **Design the Pipeline Graph**:
   - Determine pass resolutions: use `source` scale for pixel manipulation, `viewport` scale for CRT scanlines/masks/bloom.
   - Maintain color space discipline: linearize gamma in Pass 0, perform analytical math in linear FP16 (`float_framebuffer = true`), and re-apply display gamma in the final pass.
2. **Author the `.slang` Passes**:
   - Start from the [Canonical Boilerplate](#canonical-shader-boilerplate).
   - Keep `mat4 MVP;` in the UBO.
   - Count bytes in your push constant block (ensure $\le 128$ bytes total).
   - Use `vTexCoord * OutputSize.xy` rather than `gl_FragCoord`.
3. **Configure the `.slangp` Preset**:
   - Recount `shaders = N` whenever adding/removing passes.
   - Assign `aliasN = "Name"` for any pass whose output is needed downstream or across frames (`NameFeedback`).
4. **Validate**:
   - Verify that all referenced aliases and textures exist in the preset.
   - Test compilation with `librashader-cli` or RetroArch verbose logs.

---

## Canonical Shader Boilerplate

### Minimal Valid Pass (`stock.slang`)

```glsl
#version 450

// 1. Uniform Buffer Object (UBO) for MVP matrix (Binding 0)
layout(std140, set = 0, binding = 0) uniform UBO {
    mat4 MVP;
} global;

// 2. Push Constant Block (Max 128 bytes total across the shader)
layout(push_constant) uniform Push {
    vec4 SourceSize;
    vec4 OriginalSize;
    vec4 OutputSize;
    uint FrameCount;
    float Brightness;
} params;

// Optional user-tunable GUI parameters
#pragma parameter Brightness "Target Brightness" 1.0 0.0 2.0 0.05

// 3. Vertex Stage
#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;

void main() {
    gl_Position = global.MVP * Position;
    vTexCoord = TexCoord;
}

// 4. Fragment Stage
#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;

// Texture samplers begin at binding 1
layout(set = 0, binding = 1) uniform sampler2D Source;

void main() {
    vec4 color = texture(Source, vTexCoord);
    FragColor = vec4(color.rgb * params.Brightness, color.a);
}
```

### Corresponding Preset (`stock.slangp`)

```ini
shaders = 1

shader0 = "stock.slang"
filter_linear0 = false
wrap_mode0 = clamp_to_border
scale_type0 = viewport
scale0 = 1.0

Brightness = "1.100000"
```

---

## Detailed References

- [**Technical Specification**](references/specification.md): Authoritative reference for GLSL syntax, resource layout rules, render target formats, texture semantics, and the complete table of 14 built-in uniforms.
- [**Preset Specification**](references/presets.md): Comprehensive guide to `.slangp` preset properties, scaling modes (`source`, `viewport`, `absolute`), custom LUTs, and preset inheritance (`#reference`).
- [**Troubleshooting & Engineering Guidelines**](references/troubleshooting.md): In-depth solutions for push-constant budget calculations, cross-stage struct misalignments, texel snapping formulas, color spaces, and verification tooling.

---

## Pre-Flight Verification Checklist

Before finalizing any `.slang` or `.slangp` changes, verify:

- [ ] **Push Constant Budget**: Total size of the `layout(push_constant)` block is $\le 128$ bytes.
- [ ] **No `MVP` in Push Constants**: `MVP` resides in the UBO (`layout(std140, set = 0, binding = 0)`).
- [ ] **Descriptor Set #0**: All UBOs and samplers explicitly specify `set = 0`.
- [ ] **Stage Separation**: Both `#pragma stage vertex` and `#pragma stage fragment` are present.
- [ ] **No Vertex Samplers**: Textures (`sampler2D`) are declared and sampled only in the fragment stage.
- [ ] **Texel Coordinates**: Pixel offsets are calculated using `vTexCoord * OutputSize.xy`, never `gl_FragCoord`.
- [ ] **Texel Center Snapping**: Nearest-neighbor lookups at non-integer scales use `(floor(uv * Size.xy) + 0.5) * Size.zw`.
- [ ] **Preset Pass Count**: `shaders = N` in the `.slangp` matches the exact number of configured passes ($0$ through $N-1$).
- [ ] **Alias Integrity**: Every sampler `sampler2D <Name>` in a shader maps to an `aliasN = "<Name>"` or a texture in `textures = "<Name>"`.
- [ ] **Parameter Syntax**: All `#pragma parameter` statements use decimal periods (`.`), valid bounds, and have no inline comments on the same line.
