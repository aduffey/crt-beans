# Parameters for crt-beans

## Output gamma

This should be set to the gamma of your TV or monitor.

| Value | Output gamma |
| --- | --- |
| 0 | sRGB |
| 1 | 2.2 (default) |
| 2 | 2.4 |

Most people will not need to change this. Most monitors and many TVs are calibrated roughly to a pure power function gamma of 2.2, which is the default setting. Many TVs (especially OLEDs and sets calibrated for dark-room viewing or BT.1886) use a pure power function gamma of 2.4. Some monitors are calibrated to the actual sRGB gamma with a linear section near black. The difference is mostly noticeable in the glow, but can be significant. You can check whether your display uses sRGB or a 2.2 power gamma with [this shadertoy](https://www.shadertoy.com/view/7sjBWD).

The Python reference exposes the same options via `--output-gamma` (`srgb`, `2.2`, or `2.4`; default `srgb` for PNG output).

## Horizontal and vertical overscan

Set the overscan (the proportion of the image cropped vertically and horizontally). Most TVs had some amount of overscan. It is recommended that you turn overscan cropping _off_ in the core options and use the overscan cropping here. This makes the blurring filter more accurate.

The units are the total proportion of the image to crop, so a value of 0.06 means that 3% will be cropped on each side. The image will be stretched to fill the screen and match the expected aspect ratio after cropping.

## Maximum and minimum scanline size

These parameters control the size of the scanline (the bright part of the line).

A maximum value of 1.0 will result in scanlines at maximum brightness being fully merged, with no dark area between them. Lower values will result in more defined scanlines. The value represents the full width at half maximum of the scanline in proportion to the distance between scanline centers.

A minimum value of 1.0 will result in scanlines that do not vary with brightness. Lower values will result in darker lines being narrower than brighter lines (which is common on CRT TVs). The value is in proportion to the maximum spot size, so a value of 1.0 means the minimum width is the same as the maximum, a value of 0.4 means the minimum width is 40% of the maximum, etc.

Note that the maximum spot size will affect the brightness of the overall image due to the dark gaps between the scanlines. A value of 1.0 will have no brightness cost, but a value of 0.9 will reduce the brightness by 10%, etc. The minimum spot size does not affect the overall image brightness.

## Mask type

This parameter sets the mask simulation type.

| Value | Mask |
| --- | --- |
| 0 | Disabled |
| 1 | Subpixel |
| 2 | Dynamic (default) |
| 3 | Dynamic2 |
| 4 | Monochrome |

The dynamic mask is the default. It will have the specified number of phosphor triads regardless of the output monitor's resolution, for a consistent "look" across devices. It does not depend on the monitor's subpixel arrangement, so it can be used on OLEDs with non-standard subpixels, in TATE mode (rotated), etc. Very low resolution monitors (especially lower than 1080p) may have trouble with the dynamic mask.

The subpixel mask is an alternative for low resolution devices. Note that the number of phosphor triads that the subpixel mask simulates will vary depending on the output monitor's resolution. The "look" will therefore not be consistent across different resolutions. Because this mask type relies on awareness of the monitor's subpixel pattern, it will not work correctly for some monitors or orientations (e.g. TATE mode).

Dynamic2 is an alternative dynamic [aperture grille](https://en.wikipedia.org/wiki/Aperture_grille) synthesized from up to the first two harmonics of the phosphor pulse wave. It phases in the first harmonic over `[2.0, 3.0]` and the second harmonic over `[4.0, 5.0]` pixels per triad (`OutputSize.x / mask_triads`) via `smoothstep`. Rather than lifting dark gaps with an additive floor as soon as brightness exceeds the base mean (`dc_base = 0.5 - 0.125 * t2`), it blends per color channel by blooming bright phosphors—scaling the mask linearly up to `dc_base`, then shedding the second harmonic before the first as brightness increases so peaks stay at `1.0`, spatial mean is preserved, and black gaps remain dark deeper into midtones—and applies `t1` post-bloom so the `[2.0, 3.0]` transition attenuates uniformly across darks, midtones, and highlights. Like dynamic, it uses the dynamic mask triad count and does not regime-lock to the monitor's subpixel arrangement. Subpixel and dynamic also simulate aperture grilles.

Monochrome repeats white and black output pixels horizontally, splitting each repeat evenly and assigning an odd repeat's extra pixel to white. It applies the same mask value to all three color channels, preserving the input colors. It shares the static mask width with subpixel, supports periods of 2–5 pixels, and is independent of the dynamic triad count and monitor subpixel arrangement.

Disabling the mask is recommended for low resolution devices with uncommon subpixel layouts (e.g. the Steam Deck OLED) or to simulate masks with very high phosphor density (e.g. good CRT monitors).

With mask brightness boost at 1.0, the underlying scanline is blended into the image in bright areas to compensate for the mask's brightness loss.

### Mask smoothness

Ranges from 0.0 (default) to 1.0:

- **Dynamic:** scales the raised-cosine filter kernel width by `1 + smoothness`, from the original width at 0.0 to twice that width at 1.0.
- **Dynamic2:** applies a DC-preserving Gaussian low-pass filter to the harmonic amplitudes around each channel's mean (`(1 - smoothness)` for the first harmonic and `(1 - smoothness)^4` for the second), softening phosphor edges first and reducing mask contrast monotonically to flat at `1.0` without changing overall brightness.

Smoothness has no effect on the static masks (subpixel and monochrome) or when masking is disabled. The Python reference exposes the same control as `--mask-smoothness` with `--mask-type dynamic` (also called `bandlimit`) or `dynamic2`.

### Dynamic mask triads

This parameter controls the resolution of the phosphor mask for both dynamic and dynamic2. A value of 550 means that there will be 550 of each of the red, green, and blue phosphors across the simulated screen width.

This has no effect when the mask type is set to disabled, subpixel, or monochrome.

### Static mask width and monitor subpixel pattern

Static mask width controls how many monitor pixels are used for each repeating unit in both subpixel and monochrome masks. It supports 2, 3, 4, and 5 pixels (shader default: 4). For subpixel, each repeat represents one phosphor triad. For example, a value of 5.0 means that a 4k display would simulate `(2160 * (4/3)) / 5 = 576` phosphor triads across a 4:3 image.

For monochrome, each repeat consists of:

| Width | White pixels | Black pixels |
| --- | --- | --- |
| 2 | 1 | 1 |
| 3 | 2 | 1 |
| 4 | 2 | 2 |
| 5 | 3 | 2 |

Mask brightness boost accounts for the active fraction at each width. Static mask width has no effect on dynamic masks or when masking is disabled.

The monitor subpixel pattern (RGB/BGR) applies only to the subpixel mask and should match the monitor. If it does not, the subpixel mask may look distorted. It has no effect on monochrome.

The shader width parameter retains its internal ID `SubpixelMaskPattern` so saved subpixel presets keep their settings. The Python reference uses the existing `--mask-pattern` indices for both static masks: 0=2 pixels, 1=3 pixels, 2=4 pixels, 3=5 pixels. Its existing default remains index 1 (3 pixels), stored as `SimulationConfig.subpixel_mask_pattern`.

### Mask brightness boost

This parameter controls the amount of mitigation for the brightness loss caused by the mask effect, basically controlling the strength of the mask. A value of 0.0 means that the mask will be full strength and the brightness will be reduced accordingly. Higher values mean that the mask will be at full strength only in the darker areas, while the brighter areas will blend in the underlying scanlines to reduce the brightness loss. At 1.0, any pixels at full brightness will have the mask completely faded out and there will be no brightness loss.

## Glow amount and width

These parameters control the glow simulation. In this case, "glow" refers to the wide halos around bright parts of the image due to light scattering and refracting in the CRT glass. This is sometimes also called "bloom" or "diffusion."

Glow is implemented with an approximation of a gaussian blur mixed into the image. The width is the standard deviation of the gaussian to use (in proportion to screen height). The amount is the mix ratio of the blurred image to the original image. For example, a glow amount of 0.05 means the image is a mix of 5% blurred image and 95% original image.

Glow controls are available only in the full presets; the fast presets omit glow.

## Interlacing

Interlacing simulation is controlled by two parameters: `InterlacingType` (mode) and `OddFieldFirst` (field phase).

When the vertical input resolution is less than or equal to 300, no interlacing effect is applied (unless VGA mode is used on low-resolution content).

### Interlacing type

#### Bob (0.0)

Displays alternating odd or even fields each frame, simulating a CRT's alternating field raster. When combined with NTSC composite simulation, subcarrier modulation and comb filtering match the active field.

There may be noticeable flickering in this mode. In many cases, flickering was noticeable on the original systems and CRT TVs as well. However, some monitors and TVs can have excessive flickering in this mode, [due to the way they work](https://news.ycombinator.com/item?id=42507741), and the flickering may persist temporarily even after the interlaced content is gone.

#### Weave (1.0, default)

Draws both the odd and even fields and blends them together, avoiding the flickering of bob mode. For cores that output 480p, this produces a 480i "look." On true 480i content where the core updates fields alternately in a shared framebuffer, the NTSC simulation maintains per-field subcarrier phase consistency across stale and current fields. Note that there is a performance penalty because twice as many scanlines are evaluated.

#### VGA (2.0)

Enables [VGA line doubling](https://github.com/dosbox-staging/dosbox-staging/issues/2283#issue-1589164907). When the vertical input resolution is below 350, each line is doubled. For example, [DOS mode 13h](https://en.wikipedia.org/wiki/Mode_13h) games have a vertical resolution of 200 and will be drawn with 400 scanlines.

At 350 lines and above, VGA mode draws the input progressively without interlacing.

#### Off (3.0)

Turns all interlacing simulation off. The number of scanlines in the output equals the vertical resolution of the input.

### Interlacing phase

Controls whether even-numbered frames correspond to odd (top) fields (`0.0`, default) or even (bottom) fields (`1.0`). If you see reversed field order or judder during interlaced content in bob mode, toggle this parameter.

## Analog signal simulation

The preset determines which connection is simulated:

- `crt-beans-rgb` and `crt-beans-fast-rgb`: filter the red, green, and blue signals independently.
- `crt-beans-s-video` and `crt-beans-fast-s-video`: filter brightness (luma) and color (chroma) separately, using YUV.
- `crt-beans-ntsc` and `crt-beans-fast-ntsc`: encode and decode an NTSC composite signal, including interactions between brightness and color that can produce dot crawl and false colors.
- `crt-beans` and `crt-beans-fast`: linearize the input without analog signal filtering.
- `crt-beans-vga` and `crt-beans-fast-vga`: linearize the input without analog signal filtering, with VGA line handling, monitor-like spot sizes, and no mask by default.

### Signal bandwidth

Bandwidth controls horizontal detail. Lower values produce more horizontal blurring. Luma bandwidth controls brightness detail, while chroma bandwidth controls color detail and bleeding across edges.

All bandwidth values are in MHz.

| Parameter | Preset | Default | Range |
| --- | --- | --- | --- |
| Video bandwidth | RGB | 4.0 | 0.6–6.0 |
| Luma bandwidth | S-Video and NTSC | 4.0 | 0.6–6.0 |
| Chroma bandwidth | S-Video and NTSC | 1.0 | 0.6–2.0 |

Video bandwidth applies the same filtering to all three RGB channels. Chroma bandwidth is the nominal −6 dB (half-amplitude) frequency of the low-pass filter applied to each color channel.

In the NTSC presets, these parameters control filtering before the signals are combined into composite video. The decoder also applies a fixed carrier-locked chroma low-pass filter with a −6 dB point at approximately 1.2 MHz (subcarrier / 3).

### Luma trap depth

Available in the generic and console-specific NTSC presets.

This reduces brightness detail near the NTSC color subcarrier (approximately 3.58 MHz) before brightness and color are combined into composite video. It can reduce false colors and rainbows on fine text and patterns. The trap acts on the encoder's gamma-domain luma signal, independently of the decoder's Composite Y/C filter setting.

`LumaTrapDepth` ranges from **0 to 32**. The generic NTSC presets default to **0** (bypassed). Values are positive attenuation amounts in dB: `6` gives approximately −6 dB, or half the previous luma amplitude at the carrier; `32` leaves approximately 2.5%. The Python CLI uses `--ntsc-luma-trap-depth DB`.

The notch width is fixed at roughly 2 MHz. Strong settings can introduce ringing around sharp edges, and enabling the trap increases filtering cost. Use the lowest depth that suppresses unwanted false color.

### Composite Y/C filter

Available in the generic and console-specific NTSC presets.

This parameter controls how the decoder separates brightness (Y) from color (C):

- **Notch (0.0):** A simple notch filter as found on early TVs. This is the blurriest option.
- **Comb (1.0):** A simple 3-line comb filter as found on high end TVs in the 80's, and common in the 90's. This is a sharper option but causes artifacts on horizontal edges (on some consoles more than others).
- **Adaptive comb (2.0, generic default):** A 2D adaptive comb filter as found on TVs in the late 90's into the 2000's. This should avoid most of the artifacts of the 3-line comb but might not be as sharp.

In bob and weave modes on interlaced input, comb filtering uses neighboring scanlines from the same field.

### NTSC system

Available in the generic and console-specific NTSC presets.

This parameter selects the timing model used for the color subcarrier:

| Value | System |
| --- | --- |
| 0.0 | Standard NTSC (generic default) |
| 1.0 | NES |
| 2.0 | Master System |
| 3.0 | PC Engine / TurboGrafx-16 |
| 4.0 | Genesis / Mega Drive |
| 5.0 | SNES |
| 6.0 | Saturn |
| 7.0 | PlayStation |
| 8.0 | Nintendo 64 |

This changes the subcarrier spacing across the image, its phase relationship between lines and frames, and the alignment used by the comb filter. It can therefore change artifact patterns as well as their movement.

Choose the system being emulated, or Standard NTSC for a generic timing model. Several models infer the video mode from the input width, so use the core's native output resolution where possible (i.e. turn off any upscaling).

### Console preset defaults

The [console presets](../../ntsc-systems/) share NTSC settings between the normal and fast variants. Encoder settings approximate representative hardware revisions within the shader's available bandwidths and fixed luma trap width.

The shader's chroma filter reaches −3 dB at about 72% of its bandwidth setting. The presets use this response to match estimated hardware −3 dB frequencies and round the resulting settings to the nearest 0.1 MHz.

| System | Luma (MHz) | Chroma (−6 dB, MHz) | Luma trap depth (dB) | Y/C filter | Encoder basis |
| --- | ---: | ---: | ---: | --- | --- |
| NES | 6.0 | 2.0 | 0 | Notch | Direct composite synthesis by the RP2C02 (widest available bandwidths). |
| Master System | 2.0 | 1.1 | 0 | Notch | CXA1145 discrete LC luma/chroma filter circuit (matching Sega's PAL SMS and NTSC Genesis circuits). |
| PC Engine / TurboGrafx-16 | 5.0 | 2.0 | 0 | Notch | HuC6260 resistive summing network with no external chroma filter or luma trap. |
| Genesis / Mega Drive | 2.0 | 1.1 | 0 | Notch | Model 1/2 CXA1145 discrete LC luma and chroma filter circuit (no separate luma trap pass). |
| SNES | 5.0 | 0.8 | 0 | Notch | 1-CHIP / SNES Jr. (S-CPUN + S-RGB); provisional composite estimates below measured RGB edge bandwidth. |
| Saturn | 6.0 | 0.8 | 16 | Notch | CXA1645M circuit; trap depth is an estimate based on the VA0.5 series-LC topology. |
| PlayStation | 2.8 | 0.8 | 14 | Adaptive comb | PU-22/23 BH7240AKV circuit with a 56 µH / 33 pF resonant luma trap. |
| Nintendo 64 | 6.0 | 0.8 | 0 | Adaptive comb | Early ENC-NUS (BA7242F) circuit with wideband external passives and no dedicated luma trap. |

The Saturn and PlayStation encoder circuits are shown in the [Saturn VA0.5 schematic](https://consolemods.org/wiki/images/1/1b/Sega_saturn_va0.5_schematics.pdf) and the [Sony SCPH-9000 service manual](https://archive.org/details/PlayStation_SCPH-9000_Series_Service_Manual_3rd_Edition) (`PU-23`). The [Sony CXA1645 datasheet](https://www.cieri.net/Documenti/Data%20Sheets/Radio%20and%20Video%20ICs/CXA1645P%2C%20CXA1645M%20%28Sony%29.pdf) and [ROHM BH7236AF / BH7240AKV datasheet](https://www.mouser.com/datasheet/2/348/BH7236AF_BH7240AKV_NTSC_PAL_Analog_Video_Encoders-2313627.pdf) describe the encoders' resonant luma traps and internal chroma filters.

The early N64 schematic shows wideband output coupling and protection around the ENC-NUS. The pin-compatible ROHM encoder's block diagram places filtering on the subcarrier-input path, leaving output chroma filtering undocumented; no dedicated luma trap is present. [NUS-CPU-03/04 schematic](https://www.acidmods.com/RDC/NINTENDO/N64/N64_NUS_CPU_03_04.pdf), [ROHM BA7242F datasheet](https://www.alldatasheet.com/datasheet-pdf/pdf/36172/ROHM/BA7242F.html)
