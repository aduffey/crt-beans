# RetroArch Slang Troubleshooting & Engineering Guidelines

This document details common runtime bugs, platform idiosyncrasies, memory layout rules, and verification strategies for RetroArch Slang shaders.

---

## 1. Push Constant 128-Byte Budget & Struct Alignment

Vulkan guarantees only **128 bytes** of push constant storage across all pipeline stages. Exceeding this limit causes shader compilation or pipeline creation failure on compliant drivers.

### Type Size Breakdown
| GLSL Type | Size | Notes |
| :--- | :--- | :--- |
| `float`, `int`, `uint` | 4 bytes | Aligned to 4-byte boundaries |
| `vec2` | 8 bytes | Aligned to 8-byte boundaries |
| `vec3`, `vec4` | 16 bytes | Aligned to 16-byte boundaries |
| `mat4` | 64 bytes | Consumes 50% of total budget |

### Budget Allocation Pattern
```
+-------------------------------------------------------------------------+
| Push Constant Range: 128 Bytes Total (8 x 16-byte slots)                |
+------------------------------------+------------------------------------+
| Required Size Uniforms: 48 Bytes   | Parameters & Counters: 80 Bytes    |
| - vec4 SourceSize   (16 bytes)     | - uint FrameCount        (4 bytes) |
| - vec4 OriginalSize (16 bytes)     | - int FrameDirection     (4 bytes) |
| - vec4 OutputSize   (16 bytes)     | - 18 float parameters   (72 bytes) |
+------------------------------------+------------------------------------+
```

> [!WARNING]
> **Never place `mat4 MVP;` inside the push constant block.** `MVP` consumes 64 bytes. Placing it in push constants leaves only 64 bytes for *all* sizes and parameters. Always place `mat4 MVP;` inside the Uniform Buffer Object (`layout(std140, set = 0, binding = 0) uniform UBO`).

### Cross-Stage Struct Alignment Bug
If a push constant block is declared in both the vertex and fragment stages:
- The struct layout and variable order **must match identically**.
- If the vertex shader defines a subset of uniforms (e.g. only `SourceSize`) while the fragment shader defines parameters, SPIRV-Cross translation will misalign uniform offsets on D3D11, D3D12, and OpenGL Core profiles.

**Best Practice**: Put the push constant block inside a shared `.inc` file and `#include` it, or declare the push constant block **only** in the fragment stage if the vertex stage only needs `MVP`.

---

## 2. Coordinate & Sampling Pitfalls

### Never Use `gl_FragCoord`
`gl_FragCoord` has inconsistent origin and Y-axis orientation across backends:
- Vulkan NDC has Y pointing downwards; OpenGL has Y pointing upwards.
- SPIRV-Cross compensates for this at the vertex stage, which flips the interpretation of `gl_FragCoord` relative to normalized texture coordinates.

**Fix**: Always compute pixel coordinates from `vTexCoord` and the render target size:
```glsl
// Correct, platform-agnostic pixel coordinate
vec2 pixel_coord = vTexCoord * params.OutputSize.xy;
ivec2 ipixel_coord = ivec2(floor(pixel_coord));
```

### Texel Centering for Nearest-Neighbor Sampling
When sampling a texture at arbitrary or non-integer scales, sampling directly on texel boundaries causes pixel jitter, aliasing, and rounding flicker across frames.

**Fix**: Manually snap to the center of texels:
```glsl
// Offset to texel center (+0.5 texels)
vec2 texel_center = (floor(vTexCoord * params.SourceSize.xy) + 0.5) * params.SourceSize.zw;
vec4 color = texture(Source, texel_center);
```

### Screen-Space Derivatives (`dFdx`, `dFdy`)
Avoid relying on hardware derivative instructions (`dFdx`, `dFdy`, `fwidth`) for edge detection or filter kernels:
- Driver implementations vary on edge texels and border clamping.
- Tiled mobile GPUs (Mali, Adreno) often produce block artifacts across $2 \times 2$ pixel quads.

**Fix**: Use analytical step sizes derived from the size uniforms:
```glsl
vec2 dx = vec2(params.SourceSize.z, 0.0); // 1 texel step right
vec2 dy = vec2(0.0, params.SourceSize.w); // 1 texel step down
```

---

## 3. Color Spaces & Precision

### Avoiding Double Gamma Correction
Most emulator cores deliver non-linear (gamma $\approx 2.2$ or sRGB) colors in standard `R8G8B8A8_UNORM` framebuffers.

- **Naive Filter Blending**: Performing blurs, scanline simulations, or downsampling directly on non-linear pixels leads to dark halos, muddy colors, and disproportionately thick dark scanlines.
- **Linear Workflow**:
  1. **Pass 0**: Convert core RGB to linear light ($C_{\text{linear}} = C^{\gamma}$ or `srgb_to_linear(C)`). Output to an FP16 target (`float_framebuffer0 = true` or `#pragma format R16G16B16A16_SFLOAT`).
  2. **Intermediate Passes**: Perform scanline beam modeling, mask modulation, bloom, and diffusion strictly in linear light.
  3. **Final Pass**: Convert back to display color space ($C_{\text{display}} = C^{\frac{1}{\gamma}}$ or `linear_to_srgb(C)`).

### Dithering When Truncating to 8-bit
Converting high-precision linear float data to an 8-bit target causes visible banding, particularly in dark regions or smooth blurs/glows. Apply spatial or temporal blue noise dithering prior to 8-bit quantization.

---

## 4. Parser & Syntax Traps

### Locale Parsing Bug in `#pragma parameter`
RetroArch's internal parser (`slang_preprocess.cpp`) parses floating-point tokens using the C runtime's `strtod()`.
- If numbers use commas (e.g. `0,5`), parsing fails with:
  `[ERROR] [slang]: Invalid #pragma parameter line`
- **Rule**: Always format parameters with periods: `1.0`, `0.5`, `0.0`.
- **Rule**: Do not place inline comments (`// comment`) at the end of a `#pragma parameter` line.

### Missing Pass Alias Error
If a shader references an alias (e.g. `uniform sampler2D HiresPass;`) that is not declared in the `.slangp` preset via `aliasN = "HiresPass"`, reflection fails:
`[ERROR] [slang]: Texture name 'HiresPass' not found in semantic map`

---

## 5. Shader Verification & Tooling

When verifying shaders without launching a full emulator session:

1. **`librashader-cli`**:
   The `librashader-cli` tool validates presets, compiles SPIR-V, and inspects reflected semantics:
   ```sh
   # Verify reflection and bindings against a preset
   librashader-cli reflect --preset my-preset.slangp

   # Transpile to target backend (Vulkan, GLSL, HLSL)
   librashader-cli transpile --preset my-preset.slangp --target vulkan
   ```

2. **Headless / Logged RetroArch Execution**:
   To catch runtime reflection errors directly from RetroArch:
   ```sh
   retroarch --verbose --set-shader path/to/preset.slangp
   ```
   Check output logs for `[ERROR] [slang]` or `[WARN] [slang]` messages.
