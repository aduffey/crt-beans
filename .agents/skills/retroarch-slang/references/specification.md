# RetroArch Slang Shader Specification

The RetroArch Slang shader system is based on **Vulkan GLSL** (targeting `#version 450` or `#version 310 es`). Shaders are compiled to SPIR-V binaries and reflected at runtime via SPIRV-Cross (in RetroArch and `librashader`) to dynamically deduce filter chain layouts, uniform bindings, and pass connections.

---

## 1. Shader File Structure (`.slang`)

Each `.slang` file represents a single shader pass containing both vertex and fragment stages partitioned by `#pragma stage` directives.

### Preprocessor & Pragmas

| Directive | Purpose | Syntax & Example | Notes |
| :--- | :--- | :--- | :--- |
| `#version` | GLSL language target | `#version 450` | Use `#version 450` for desktop Vulkan/GLCore. `#version 310 es` is supported for mobile GLES. |
| `#pragma stage` | Stage boundary marker | `#pragma stage vertex`<br>`#pragma stage fragment` | **Mandatory**. Both stages must be explicitly declared in every `.slang` file. |
| `#pragma name` / `#pragma alias` | Assign pass alias | `#pragma name MyPass`<br>`#pragma alias MyPass` | Registers an alias for this pass output. Subsequent passes can sample `MyPass` and `MyPassFeedback`. `#pragma alias` is an accepted synonym. |
| `#pragma format` | Render target pixel format | `#pragma format R16G16B16A16_SFLOAT`<br>`#pragma format float_framebuffer`<br>`#pragma format srgb_framebuffer` | Sets render target format. Defaults to `R8G8B8A8_UNORM` if omitted. Shorthands `float_framebuffer` and `srgb_framebuffer` are also recognized. |
| `#pragma parameter` | User-tunable GUI uniform | `#pragma parameter ID "Description" default min max [step]` | Exposes a uniform parameter in RetroArch Quick Menu. Values must use decimal points (never commas). Do not place inline comments on the parameter line. |
| `#include` / `#pragma include` | File inclusion | `#include "common.inc"`<br>`#pragma include "common.inc"` | Inlines relative file. |
| `#pragma include_optional` | Soft file inclusion | `#pragma include_optional "config.inc"` | Inlines relative file if it exists; silently ignores if missing. |

---

## 2. Resource Layout & Interface Rules

All resources must conform to Vulkan descriptor layout constraints:

1. **Descriptor Set `#0`**: All resources (UBO, samplers) must reside in `set = 0`.
2. **Explicit Bindings**: Every uniform buffer and texture sampler must define an explicit `binding = N`.
   - Convention: `binding = 0` is reserved for the Uniform Buffer Object (UBO).
   - Samplers use sequential bindings `1, 2, 3, ...`.
3. **UBO Constraints**: Maximum **one** UBO per shader pass:
   ```glsl
   layout(std140, set = 0, binding = 0) uniform UBO {
       mat4 MVP;
   } global;
   ```
4. **Push Constant Block**: Maximum **one** push constant block per shader pass:
   ```glsl
   layout(push_constant) uniform Push {
       vec4 SourceSize;
       vec4 OriginalSize;
       vec4 OutputSize;
       uint FrameCount;
       // ... user parameters ...
   } params;
   ```
   - **Guaranteed Size Limit**: Exactly **128 bytes** (8 `vec4`s or 32 `float`/`uint`/`int` words).
   - **Cross-Stage Consistency**: If a push constant block is declared in both vertex and fragment stages, the struct definition, members, order, and padding must match **identically**.
5. **Vertex Attributes**: Fixed locations:
   ```glsl
   layout(location = 0) in vec4 Position;
   layout(location = 1) in vec2 TexCoord;
   layout(location = 0) out vec2 vTexCoord;
   ```
6. **Fragment Output**: Fixed location 0:
   ```glsl
   layout(location = 0) out vec4 FragColor;
   ```
   *(Use `uvec4` or `ivec4` when rendering to integer formats).*
7. **Stage Sampling Restriction**: `sampler2D` can **only** be bound and sampled within `#pragma stage fragment`. The vertex stage cannot sample textures directly, but can access any `<Texture>Size` companion uniform.

---

## 3. Built-in Semantics

The shader runtime reflects uniforms and textures by their exact variable names.

### Built-in Textures (`sampler2D` in Fragment Stage)

| Texture Sampler | Description |
| :--- | :--- |
| `Source` | Output texture from the immediately preceding pass (or initial core frame for Pass 0). |
| `Original` | Initial frame delivered by the core emulation engine at the start of the frame. |
| `OriginalHistory0` | Equivalent alias for `Original`. |
| `OriginalHistory1` .. `OriginalHistoryN` | Historical core input frames: `OriginalHistory1` is $T-1$, `OriginalHistory2` is $T-2$, etc. Frontend allocates history ring buffers automatically based on the highest index reflected. |
| `PassOutput0` .. `PassOutputN` | Output texture of pass `N` for the **current frame**. Strictly causal: a pass can only sample `PassOutput` from earlier passes in the chain ($M < N$). |
| `PassFeedback0` .. `PassFeedbackN` | Output texture of pass `N` from the **previous frame**. Non-causal: any pass can read feedback from any pass (including itself) to build recursive/temporal feedback loops. |
| `<PassAlias>` | Custom alias assigned via `aliasN` in preset or `#pragma name <PassAlias>`. References current frame output of that pass. |
| `<PassAlias>Feedback` | Output of the aliased pass from the **previous frame**. Automatically generated by the runtime. |
| `<TextureName>` | Custom static lookup texture (LUT) declared in preset via `textures = "TextureName"`. |

### Built-in Uniforms

Every texture sampler automatically provides a companion `vec4 <Texture>Size` uniform with the layout:
$$\text{Size} = \begin{pmatrix} \text{width}, & \text{height}, & \frac{1}{\text{width}}, & \frac{1}{\text{height}} \end{pmatrix}$$

| Uniform Name | Type | Description |
| :--- | :--- | :--- |
| `MVP` / `MVPMatrix` | `mat4` | Model-View-Projection matrix for vertex coordinate transformation. |
| `OutputSize` | `vec4` | Dimensions and texel step of the current pass render target. |
| `FinalViewportSize` | `vec4` | Dimensions and texel step of the final screen/window viewport. |
| `SourceSize` | `vec4` | Dimensions and texel step of the current pass input (`Source`). |
| `OriginalSize` | `vec4` | Dimensions and texel step of the original core frame (`Original`). |
| `FrameCount` | `uint` | Monotonically incrementing frame counter (wraps via preset `frame_count_modN`). |
| `FrameDirection` | `int` | Playback direction: `1` during normal forward playback, `-1` during rewind. |
| `Rotation` | `uint` | Screen orientation: `0` = 0°, `1` = 90°, `2` = 180°, `3` = 270°. |
| `OriginalAspect` | `float` | Target display aspect ratio requested by the emulator core. |
| `OriginalAspectRot` | `float` | Target display aspect ratio adjusted for active screen rotation. |
| `OriginalFPS` | `float` | Nominal native core framerate (e.g. 59.94, 60.0). Guard with `#ifdef _HAS_FRAMETIME_UNIFORMS`. |
| `FrameTimeDelta` | `uint` | Elapsed microseconds between the current and previous frame. Guard with `#ifdef _HAS_FRAMETIME_UNIFORMS`. |
| `TotalSubFrames` | `uint` | Subframe multiplier configured for high-refresh display BFI / rolling scanline simulation. |
| `CurrentSubFrame` | `uint` | Current subframe index ($0 \le \text{CurrentSubFrame} < \text{TotalSubFrames}$). |

---

## 4. Render Target Formats

Set via `#pragma format <FORMAT>` or preset `float_framebufferN = true` / `srgb_framebufferN = true`.

| Bit Depth | Format Identifiers |
| :--- | :--- |
| **8-bit** | `R8_UNORM`, `R8_UINT`, `R8_SINT`, `R8G8_UNORM`, `R8G8_UINT`, `R8G8_SINT`, `R8G8B8A8_UNORM`, `R8G8B8A8_UINT`, `R8G8B8A8_SINT`, `R8G8B8A8_SRGB` |
| **10-bit** | `A2B10G10R10_UNORM_PACK32`, `A2B10G10R10_UINT_PACK32` |
| **16-bit** | `R16_UINT`, `R16_SINT`, `R16_SFLOAT`, `R16G16_UINT`, `R16G16_SINT`, `R16G16_SFLOAT`, `R16G16B16A16_UINT`, `R16G16B16A16_SINT`, `R16G16B16A16_SFLOAT` |
| **32-bit** | `R32_UINT`, `R32_SINT`, `R32_SFLOAT`, `R32G32_UINT`, `R32G32_SINT`, `R32G32_SFLOAT`, `R32G32B32A32_UINT`, `R32G32B32A32_SINT`, `R32G32B32A32_SFLOAT` |
| **Shorthands** | `float_framebuffer` (maps to `R16G16B16A16_SFLOAT`), `srgb_framebuffer` (maps to `R8G8B8A8_SRGB`) |

### Format Guidelines
- **Default (`R8G8B8A8_UNORM`)**: Standard dynamic range, 8-bit unnormalized. Values outside $[0.0, 1.0]$ clamp.
- **Floating Point (`R16G16B16A16_SFLOAT`)**: Mandatory for linear color pipelines, analytical beam/spot profiles, high-dynamic-range (HDR) accumulation, and multi-pass blurs to prevent banding.
- **Single / Dual Channel (`R8_UNORM`, `R16_SFLOAT`, `R16G16_SFLOAT`)**: Use for phosphor masks, Gaussian weights, or separated blur passes to cut VRAM and bandwidth.
