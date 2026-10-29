# RetroArch Slang Preset Specification (`.slangp`)

A `.slangp` file defines the complete filter chain execution graph, pass sequence, scaling rules, frame buffer formats, custom textures (LUTs), and parameter overrides.

---

## 1. Top-Level Preset Configuration

Every preset starts by declaring the total number of shader passes:

```ini
shaders = 3
```

All pass-specific keys are indexed with a `0`-based integer $N$ ($0 \le N < \text{shaders}$).

---

## 2. Pass Properties

For each pass `N`:

| Setting | Type / Allowed Values | Default | Description |
| :--- | :--- | :--- | :--- |
| `shaderN` | File path (string) | *Required* | Path to the `.slang` file relative to the preset directory. |
| `filter_linearN` | `true` \| `false` | Global setting | Sampling filter for the pass input texture. `true` = Bilinear interpolation (smooth). `false` = Nearest neighbor (sharp pixel art). |
| `wrap_modeN` | `clamp_to_border`<br>`clamp_to_edge`<br>`repeat`<br>`mirrored_repeat` | `clamp_to_border` | Texture wrapping mode when coordinates exceed $[0.0, 1.0]$. `clamp_to_border` returns transparent black (`vec4(0.0)`). |
| `scale_typeN` | `source`<br>`viewport`<br>`absolute` | `source` | Coordinate scaling reference for both axes. |
| `scale_type_xN`<br>`scale_type_yN` | `source`<br>`viewport`<br>`absolute` | `source` | Independent horizontal or vertical scaling reference. |
| `scaleN` | `float` | `1.0` | Scaling multiplier (for `source`/`viewport`) or dimension in pixels (for `absolute`). |
| `scale_xN`<br>`scale_yN` | `float` | `1.0` | Independent horizontal or vertical scaling multiplier or pixel dimension. |
| `aliasN` | Identifier (string) | *None* | Exposes this pass's output to subsequent passes as `uniform sampler2D <Alias>;` and previous-frame feedback as `uniform sampler2D <Alias>Feedback;`. |
| `float_framebufferN` | `true` \| `false` | `false` | Promotes this pass's render target to 16-bit floating point (`R16G16B16A16_SFLOAT`). Equivalent to `#pragma format R16G16B16A16_SFLOAT`. |
| `srgb_framebufferN` | `true` \| `false` | `false` | Configures the render target as sRGB (`R8G8B8A8_SRGB`). Hardware handles automatic sRGB gamma encoding on write and linear decoding on read. |
| `mipmap_inputN` | `true` \| `false` | `false` | Generates a full mipmap chain for the input texture before executing this pass. Enables `textureLod()` across mip levels. |
| `frame_count_modN` | `uint` | `0` (no mod) | Modulo divisor applied to `params.FrameCount` for this pass (useful for cyclic animations: `FrameCount % frame_count_modN`). |

---

## 3. Scaling Modes Explained

The render target dimensions for pass `N` are determined by `scale_typeN` and `scaleN`:

- **`source`**: The target resolution is scaled relative to the input texture of pass $N$.
  $$\text{target\_size} = \text{input\_size} \times \text{scale}$$
  *Example*: `scale_type0 = source`, `scale0 = 2.0` doubles the resolution of the core frame.
- **`viewport`**: The target resolution is scaled relative to the final screen/window backbuffer.
  $$\text{target\_size} = \text{viewport\_size} \times \text{scale}$$
  *Example*: `scale_type2 = viewport`, `scale2 = 1.0` renders directly at display resolution.
- **`absolute`**: The target resolution is set to a fixed pixel size regardless of input or screen size.
  *Example*: `scale_type_x = absolute`, `scale_x = 320.0`, `scale_type_y = absolute`, `scale_y = 240.0`.

### Decoupled Axes
Horizontal and vertical scaling can be controlled separately:
```ini
# Horizontal resolution scaled to display width, vertical resolution locked to native scanlines
scale_type_x1 = viewport
scale_x1 = 1.0
scale_type_y1 = source
scale_y1 = 1.0
```

---

## 4. Custom Look-Up Textures (LUTs)

Custom static textures (such as blue noise, color palettes, vignette masks, or phosphor slot masks) are declared at the top of the preset:

```ini
# Semicolon-delimited list of texture identifiers
textures = "NoiseTex;MaskTex"

# Map identifiers to image files (PNG)
NoiseTex = "shaders/textures/blue_noise.png"
NoiseTex_linear = false
NoiseTex_wrap_mode = repeat
NoiseTex_mipmap = false

MaskTex = "shaders/textures/shadow_mask.png"
MaskTex_linear = true
MaskTex_wrap_mode = clamp_to_edge
MaskTex_mipmap = false
```

In the shader code, bind the texture matching the declared identifier:
```glsl
layout(set = 0, binding = 2) uniform sampler2D NoiseTex;
// Automatically provides companion dimensions:
// vec4 NoiseTexSize; (width, height, 1/width, 1/height)
```

---

## 5. Preset Parameter Overrides

Any `#pragma parameter` declared in any `.slang` file across the chain can be initialized or overridden in the `.slangp` file:

```ini
# Format: PARAMETER_NAME = "value"
CRT_CURVATURE = "1.000000"
SCANLINE_STRENGTH = "0.750000"
GAMMA_INPUT = "2.400000"
```

---

## 6. Preset Inheritance (`#reference`)

Presets can inherit from another base preset file, applying only overrides or customizations without duplicating the entire pass chain:

```ini
#reference "crt-beans.slangp"

# Override specific parameters
GlowAmount = "0.200000"
OutputGamma = "1.000000"
```

The path is relative to the directory containing the child `.slangp` file.

---

## 7. Complete Multi-Pass Pipeline Example

```ini
shaders = 4

# Load an external dither pattern
textures = "DitherLUT"
DitherLUT = "shaders/dither.png"
DitherLUT_linear = false
DitherLUT_wrap_mode = repeat

# Pass 0: Linearize gamma (native resolution)
shader0 = "shaders/linearize.slang"
filter_linear0 = false
srgb_framebuffer0 = false
float_framebuffer0 = true
scale_type0 = source
scale0 = 1.0
alias0 = "LinearizedInput"

# Pass 1: Render scanlines at full viewport resolution
shader1 = "shaders/scanlines.slang"
filter_linear1 = false
wrap_mode1 = clamp_to_border
float_framebuffer1 = true
scale_type1 = viewport
scale1 = 1.0
alias1 = "ScanlinePass"

# Pass 2: Downsample and blur for glow accumulation (1/4 display resolution)
shader2 = "shaders/glow_blur.slang"
filter_linear2 = true
wrap_mode2 = clamp_to_border
float_framebuffer2 = true
scale_type2 = viewport
scale2 = 0.25

# Pass 3: Composite scanlines + glow, dither, and convert back to sRGB display
shader3 = "shaders/composite.slang"
filter_linear3 = true
wrap_mode3 = clamp_to_border
scale_type3 = viewport
scale3 = 1.0

# Initial parameter overrides
GLOW_RADIUS = "2.500000"
DITHER_STRENGTH = "1.000000"
```
