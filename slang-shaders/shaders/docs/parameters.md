# Parameters for crt-beans

## Output gamma

This should be set to the gamma of your TV or monitor. Most people will not need to change this. Most TVs and monitors are calibrated roughly to a pure power function gamma of 2.2, which is the default setting. Some monitors are calibrated to the actual sRGB gamma with a linear section near black. The difference is mostly noticeable in the glow, but can be significant. You can check which one your monitor uses with [this shadertoy](https://www.shadertoy.com/view/7sjBWD).

## Horizontal and vertical overscan

Set the overscan (the proportion of the image cropped vertically and horizontally). Most TVs had some amount of overscan. It is recommended that you turn overscan cropping _off_ in the core options and use the overscan cropping here. This makes the blurring filter more accurate.

The units are the total proportion of the image to crop, so a value of 0.06 means that 3% will be cropped on each side. The image will be stretched to fill the screen and match the expected aspect ratio after cropping.

## Maximum and minimum scanline size

These parameters control the size of the scanline (the bright part of the line).

A maximum value of 1.0 will result in scanlines at maximum brightness being fully merged, with no dark area between them. Lower values will result in more defined scanlines. The value represents the full width at half maximum of the scanline in proportion to the distance between scanline centers.

A minumum value of 1.0 will result in scanlines that do not vary with brightness. Lower values will result in darker lines being narrower than brighter lines (which is common on CRT TVs). The value is in proportion to the maximum spot size, so a value of 1.0 means the minimum width is the same as the maximum, a value of 0.4 means the minimum width is 40% of the maximum, etc.

Note that the maximum spot size will affect the brightness of the overall image due to the dark gaps between the scanlines. A value of 1.0 will have no brightness cost, but a value of 0.9 will reduce the brightness by 10%, etc. The minimum spot size does not affect the overall image brightness.

## Mask type

This parameter sets the mask simulation type.

The dynamic mask is the default. It will have the specified number of phosphor triads regardless of the output monitor's resolution, for a consistent "look" across devices. It does not depend on the monitor's subpixel arrangement, so it can be used on OLEDs with non-standard subpixels, in TATE mode (rotated), etc. Very low resolution monitors (especially lower than 1080p) may have trouble with the dynamic mask.

The subpixel mask is an alternative for low resolution devices. Note that the number of phosphor triads that the subpixel mask simulates will vary depending on the output monitor's resolution. The "look" will therefore not be consistent across different resolutions. Because this mask type relies on awareness of the monitor's subpixel pattern, it will not work correctly for some monitors or orientations (e.g. TATE mode).

Currently, both mask types (subpixel and dynamic) simulate an [aperture grille](https://en.wikipedia.org/wiki/Aperture_grille).

Disabling the mask is recommended for low resolution devices with uncommon subpixel layouts (e.g. the Steam Deck OLED) or to simulate masks with very high phosphor density (e.g. good CRT monitors).

Masks have no brightness cost. The underlying scanline is blended into the image in bright areas so that the brightness is not reduced.

### Dynamic mask triads

This parameter controls the resolution of the phosphor mask for the dynamic mask type only. A value of 550 means that there will be 550 of each of the red, green, and blue phosphors across the simulated screen width.

This has no effect when the mask type is set to disabled or subpixel.

### Subpixel mask pattern and width

The monitor subpixel pattern should be set to match the monitor. If it is not, the mask may look distorted.

The subpixel mask width controls how many monitor pixels are used for each phosphor triad. For example, a value of 5.0 means that a 4k display would simulate `(2160 * (4/3)) / 5 = 576` phosphor triads across a 4:3 image.

### Mask brightness boost

This parameter controls the amount of mitigation for the brightness loss caused by the mask effect, basically controlling the strength of the mask. A value of 0.0 means that the mask will be full strength and the brightness will be reduced accordingly. Higher values mean that the mask will be at full strength only in the darker areas, while the brighter areas will blend in the underlying scanlines to reduce the brightness loss. At 1.0, any pixels at full brightness will have the mask completely faded out and there will be no brightness loss.

## Glow amount and width

These parameters control the glow simulation. In this case, "glow" refers to the wide halos around bright parts of the image due to light scattering and refracting in the CRT glass. This is sometimes also called "bloom" or "diffusion."

Glow is implemented with an approximation of a gaussian blur mixed into the image. The width is the standard deviation of the gaussian to use (in proportion to screen height). The amount is the mix ratio of the blurred image to the original image. For example, a glow amount of 0.05 means the image is a mix of 5% blurred image and 95% original image.

## Interlacing

Interlacing simulation is controlled by two parameters: `InterlacingType` (mode) and `OddFieldFirst` (field phase).

When the vertical input resolution is less than or equal to 300, no interlacing effect is applied (unless VGA mode is used on low-resolution content).

### Interlacing type

#### Bob (0.0)

Displays alternating odd or even fields each frame, simulating a CRT's alternating field raster. When combined with NTSC composite simulation, subcarrier modulation and comb filtering match the active field.

There may be noticeable flickering in this mode. In many cases, flickering was noticeable on the original systems and CRT TVs as well. However, some monitors and TVs can have excessive flickering in this mode, [due to the way they work](https://news.ycombinator.com/item?id=42507741), and the flickering may persist temporarily even after the interlaced content is gone.

#### Weave (1.0)

Draws both the odd and even fields and blends them together, avoiding the flickering of bob mode. For cores that output 480p, this produces a 480i "look." On true 480i content where the core updates fields alternately in a shared framebuffer, the NTSC simulation maintains per-field subcarrier phase consistency across stale and current fields. Note that there is a performance penalty because twice as many scanlines are evaluated.

#### VGA (2.0)

Enables [VGA line doubling](https://github.com/dosbox-staging/dosbox-staging/issues/2283#issue-1589164907). When the vertical input resolution is below 350, each line is doubled. For example, [DOS mode 13h](https://en.wikipedia.org/wiki/Mode_13h) games have a vertical resolution of 200 and will be drawn with 400 scanlines.

#### Off (3.0)

Turns all interlacing simulation off. The number of scanlines in the output equals the vertical resolution of the input.

### Interlacing phase

Controls whether even-numbered frames correspond to odd (top) fields (`0.0`, default) or even (bottom) fields (`1.0`). If you see reversed field order or judder during interlaced content in bob mode, toggle this parameter.

## Analog signal simulation

The preset determines which connection is simulated:

- `crt-beans-rgb`: filters the red, green, and blue signals independently.
- `crt-beans-s-video`: filters brightness (luma) and color (chroma) separately, using YUV.
- `crt-beans-ntsc`: encodes and decodes an NTSC composite signal, including interactions between brightness and color that can produce dot crawl and false colors.
- `crt-beans-vga` and `crt-beans-fast`: do not apply analog signal filtering.

### Signal bandwidth

Bandwidth controls horizontal detail. Lower values produce more horizontal blurring. Luma bandwidth controls brightness detail, while chroma bandwidth controls color detail and bleeding across edges.

All bandwidth values are in MHz.

| Parameter | Preset | Default | Range |
| --- | --- | --- | --- |
| Video bandwidth | RGB | 4.0 | 0.6–6.0 |
| Luma bandwidth | S-Video and NTSC | 4.0 | 0.6–6.0 |
| Chroma bandwidth | S-Video | 1.2 | 0.6–1.6 |
| Chroma bandwidth | NTSC | 1.2 | 0.6–6.0 |

Video bandwidth applies the same filtering to all three RGB channels. Chroma bandwidth applies the same filtering to both color channels.

In the NTSC preset, these parameters control filtering before the signals are combined into composite video. The decoder also applies a fixed chroma low-pass filter of approximately 1.3 MHz, so increasing chroma bandwidth does not remove all color blurring.

### Composite Y/C filter

Available only in `crt-beans-ntsc`.

This parameter controls how the decoder separates brightness (Y) from color (C):

- **Notch (0.0):** A simple notch filter as found on early TVs. This is the blurriest option.
- **Comb (1.0):** A simple 3-line comb filter as found on high end TVs in the 80's, and common in the 90's. This is a sharper option but causes artifacts on horizontal edges (on some consoles more than others).
- **Adaptive comb (2.0, default):** A 2D adaptive comb filter as found on TVs in the late 90's into the 2000's. This should avoid most of the artifacts of the 3-line comb but might not be as sharp.

In bob and weave modes on interlaced input, comb filtering uses neighboring scanlines from the same field.

### NTSC system

Available only in `crt-beans-ntsc`.

This parameter selects the timing model used for the color subcarrier:

| Value | System |
| --- | --- |
| 0.0 | Standard NTSC (default) |
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
