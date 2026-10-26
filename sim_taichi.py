import taichi as ti

# Taichi must be initialized before importing modules that define Taichi fields (e.g. masking.py)
ti.init()

import argparse
import numpy as np
from imageio.v3 import imwrite, imread

# Import configuration, helpers, and submodules
from config import SimulationConfig, Config, DEFAULT_CONFIG
from common import gamma_to_linear, linear_to_srgb
from filtering import simulate_analog
from spot_sim import simulate_spot
from masking import apply_mask
from diffusion import apply_diffusion


def simulate(img, config: SimulationConfig = None):
    if config is None:
        config = DEFAULT_CONFIG

    # Stage 1: Analog simulation frontend
    if config.enable_analog and config.analog_type not in (None, 'none', 'disabled'):
        print(f'Simulating analog frontend ({config.analog_type})...')
        img_linear = simulate_analog(
            img,
            analog_type=config.analog_type,
            samples=config.samples,
            gamma=config.gamma,
            rgb_bandwidth=config.rgb_bandwidth,
            luma_bandwidth=config.luma_bandwidth,
            chroma_bandwidth=config.chroma_bandwidth,
            ntsc_filter_type=config.ntsc_filter_type,
            ntsc_system=config.ntsc_system,
            ntsc_frame_count=config.ntsc_frame_count,
            interlacing_type=config.interlacing_type,
            odd_field_first=config.odd_field_first,
        )
    else:
        print('Analog simulation bypassed (linearize)...')
        img_linear = gamma_to_linear(img, config.gamma)

    # Stage 2: CRT scanline spot simulation
    img_linear = img_linear.astype(np.float32)
    print(f'Simulating CRT spot ({config.scanline_method})...')
    output_dim = (config.output_resolution[0], config.output_resolution[1], 3)
    img_spot = simulate_spot(
        img_linear,
        output_dim,
        config.scanline_method,
        config.max_spot_size,
        config.min_spot_size,
        config.overscan_horizontal,
        config.overscan_vertical,
        interlacing_type=config.interlacing_type,
        odd_field_first=config.odd_field_first,
        frame_count=config.ntsc_frame_count,
    )

    # Stage 3: Mask
    if config.enable_masking and config.mask_type not in (None, 'none'):
        print(f'Masking ({config.mask_type})...')
        img_masked = apply_mask(
            img_spot,
            config.mask_type,
            config.subpixel_mask_pattern,
            config.tiled_mask_type,
            config.mask_triads,
            config.mask_amount,
            mask_smoothness=config.mask_smoothness,
        )
    else:
        img_masked = img_spot

    # Stage 4: Diffusion
    if config.enable_diffusion and config.diffusion_type not in (None, 'none'):
        print(f'Blurring ({config.diffusion_type})...')
        img_diffused = apply_diffusion(
            img_masked,
            config.diffusion_type,
            config.blur_sigma,
            config.blur_amount,
        )
    else:
        img_diffused = img_masked

    return img_diffused


def build_parser():
    parser = argparse.ArgumentParser(description='Generate a CRT-simulated image')
    parser.add_argument('input', help='Input image file path')
    parser.add_argument('output', help='Output image file path')

    # Scanline options
    parser.add_argument('--scanline-method', '--scanlines', dest='scanline_method',
                        choices=['sampled', 'fast', 'analytical'],
                        default=DEFAULT_CONFIG.scanline_method,
                        help='Scanline simulation method')
    parser.add_argument('--output-resolution', '--res', dest='output_resolution',
                        nargs=2, type=int, default=list(DEFAULT_CONFIG.output_resolution),
                        metavar=('HEIGHT', 'WIDTH'), help='Output resolution in pixels (height width)')
    parser.add_argument('--max-spot-size', type=float, default=DEFAULT_CONFIG.max_spot_size,
                        help='Maximum beam spot size')
    parser.add_argument('--min-spot-size', type=float, default=DEFAULT_CONFIG.min_spot_size,
                        help='Minimum beam spot size')
    parser.add_argument('--overscan-h', dest='overscan_horizontal', type=float,
                        default=DEFAULT_CONFIG.overscan_horizontal, help='Horizontal overscan crop ratio')
    parser.add_argument('--overscan-v', dest='overscan_vertical', type=float,
                        default=DEFAULT_CONFIG.overscan_vertical, help='Vertical overscan crop ratio')
    parser.add_argument('--interlacing-type', dest='interlacing_type',
                        choices=['bob', 'weave', 'vga', 'off'],
                        default=DEFAULT_CONFIG.interlacing_type,
                        help='Interlacing simulation type')
    parser.add_argument('--odd-field-first', dest='odd_field_first',
                        action=argparse.BooleanOptionalAction,
                        default=DEFAULT_CONFIG.odd_field_first,
                        help='Assume odd field first on even frame counts')

    # Analog simulation options
    parser.add_argument('--analog', dest='enable_analog', action=argparse.BooleanOptionalAction,
                        default=DEFAULT_CONFIG.enable_analog,
                        help='Enable or disable analog simulation frontend')
    parser.add_argument('--analog-type', '--analog-frontend', dest='analog_type',
                        choices=['none', 'linearize', 'rgb', 'svideo', 's-video', 'ntsc', 'composite'],
                        default=DEFAULT_CONFIG.analog_type,
                        help='Type of analog simulation frontend')
    parser.add_argument('--samples', type=int, default=DEFAULT_CONFIG.samples,
                        help='Horizontal sampling rate for analog frontend')
    parser.add_argument('--gamma', type=float, default=DEFAULT_CONFIG.gamma,
                        help='Gamma value for color conversion')
    parser.add_argument('--rgb-bandwidth', type=float, default=DEFAULT_CONFIG.rgb_bandwidth,
                        help='RGB signal bandwidth in Hz')
    parser.add_argument('--luma-bandwidth', type=float, default=DEFAULT_CONFIG.luma_bandwidth,
                        help='Luma bandwidth in Hz for S-Video and NTSC')
    parser.add_argument('--chroma-bandwidth', type=float, default=DEFAULT_CONFIG.chroma_bandwidth,
                        help='Chroma bandwidth in Hz for S-Video and NTSC')
    parser.add_argument('--ntsc-filter-type', dest='ntsc_filter_type',
                        choices=['adaptive', 'comb', 'notch'],
                        default=DEFAULT_CONFIG.ntsc_filter_type,
                        help='NTSC composite Y/C separation filter type')
    parser.add_argument('--ntsc-system', dest='ntsc_system',
                        choices=['standard', 'nes', 'sms', 'tg16', 'pce', 'genesis', 'snes', 'saturn', 'ps1', 'psx', 'n64'],
                        default=DEFAULT_CONFIG.ntsc_system,
                        help='NTSC system timing preset')
    parser.add_argument('--ntsc-frame-count', dest='ntsc_frame_count', type=int,
                        default=DEFAULT_CONFIG.ntsc_frame_count,
                        help='NTSC frame count for color subcarrier phase alternation')

    # Masking options
    parser.add_argument('--masking', action=argparse.BooleanOptionalAction,
                        default=DEFAULT_CONFIG.enable_masking,
                        help='Enable or disable CRT phosphor mask')
    parser.add_argument('--mask-type', '--mask', dest='mask_type',
                        choices=['bandlimit', 'dynamic', 'dynamic2', 'monochrome', 'subpixel', 'coverage', 'additive', 'additive2', 'tiled', 'none'],
                        default=DEFAULT_CONFIG.mask_type,
                        help='Type of phosphor mask to apply')
    parser.add_argument('--mask-pattern', dest='subpixel_mask_pattern', type=int, choices=[0, 1, 2, 3],
                        default=DEFAULT_CONFIG.subpixel_mask_pattern,
                        help='Subpixel mask pattern index (0=1080p, 1=1440p, 2=4k, 3=4k low TVL)')
    parser.add_argument('--tiled-mask-type', choices=['slot', 'aperture'],
                        default=DEFAULT_CONFIG.tiled_mask_type,
                        help='Tiled mask texture type')
    parser.add_argument('--mask-triads', type=float, default=DEFAULT_CONFIG.mask_triads,
                        help='Number of mask triads across screen')
    parser.add_argument('--mask-amount', type=float, default=DEFAULT_CONFIG.mask_amount,
                        help='Phosphor mask blend amount')
    parser.add_argument('--mask-smoothness', type=float, default=DEFAULT_CONFIG.mask_smoothness,
                        help='Mask smoothness from 0 to 1 (dynamic, dynamic2, monochrome)')

    # Diffusion options
    parser.add_argument('--diffusion', action=argparse.BooleanOptionalAction,
                        default=DEFAULT_CONFIG.enable_diffusion,
                        help='Enable or disable glow diffusion')
    parser.add_argument('--diffusion-type', choices=['gaussian', 'box', 'none'],
                        default=DEFAULT_CONFIG.diffusion_type,
                        help='Diffusion blur algorithm')
    parser.add_argument('--blur-sigma', type=float, default=DEFAULT_CONFIG.blur_sigma,
                        help='Diffusion blur sigma fraction of screen height')
    parser.add_argument('--blur-amount', type=float, default=DEFAULT_CONFIG.blur_amount,
                        help='Diffusion glow blend amount')

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = SimulationConfig(
        enable_analog=args.enable_analog,
        analog_type=args.analog_type,
        samples=args.samples,
        gamma=args.gamma,
        rgb_bandwidth=args.rgb_bandwidth,
        luma_bandwidth=args.luma_bandwidth,
        chroma_bandwidth=args.chroma_bandwidth,
        ntsc_filter_type=args.ntsc_filter_type,
        ntsc_system=args.ntsc_system,
        ntsc_frame_count=args.ntsc_frame_count,
        scanline_method=args.scanline_method,
        output_resolution=tuple(args.output_resolution),
        max_spot_size=args.max_spot_size,
        min_spot_size=args.min_spot_size,
        overscan_horizontal=args.overscan_horizontal,
        overscan_vertical=args.overscan_vertical,
        interlacing_type=args.interlacing_type,
        odd_field_first=args.odd_field_first,
        enable_masking=args.masking,
        mask_type=args.mask_type,
        subpixel_mask_pattern=args.subpixel_mask_pattern,
        tiled_mask_type=args.tiled_mask_type,
        mask_triads=args.mask_triads,
        mask_amount=args.mask_amount,
        mask_smoothness=args.mask_smoothness,
        enable_diffusion=args.diffusion,
        diffusion_type=args.diffusion_type,
        blur_sigma=args.blur_sigma,
        blur_amount=args.blur_amount,
    )

    # Read image
    img_original = imread(args.input)
    img_float = img_original.astype(np.float32) / 255.0

    # Simulate
    img_crt = simulate(img_float, config=config)

    # To sRGB
    print('Color transform and save...')
    img_final_srgb = linear_to_srgb(img_crt)

    imwrite(args.output, img_final_srgb)


if __name__ == '__main__':
    main()
