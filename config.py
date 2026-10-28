from dataclasses import dataclass


@dataclass
class SimulationConfig:
    # Analog simulation frontend
    enable_analog: bool = True
    analog_type: str = 'rgb'  # 'none'/'linearize', 'rgb', 'svideo', 'ntsc'
    samples: int = 900  # 1024, 2880, 907, 1400
    gamma: float = 2.4
    rgb_bandwidth: float = 4.0e6      # 4.0 MHz for RGB
    luma_bandwidth: float = 4.0e6     # 4.0 MHz for S-Video and NTSC luma
    chroma_bandwidth: float = 1.2e6   # 1.2 MHz for S-Video and NTSC chroma

    # NTSC composite options
    ntsc_filter_type: str = 'adaptive'  # 'adaptive', 'comb', 'notch'
    ntsc_system: str = 'standard'       # 'standard', 'nes', 'sms', 'tg16', 'genesis', 'snes', 'saturn', 'ps1', 'n64'
    ntsc_frame_count: int = 0

    # For scanlines, interlacing, and overscan
    scanline_method: str = 'analytical'  # 'sampled', 'fast', 'analytical'
    output_resolution: tuple[int, int] = (2160, 2880)
    max_spot_size: float = 0.9
    min_spot_size: float = 0.4
    overscan_horizontal: float = 0.0
    overscan_vertical: float = 0.0
    interlacing_type: str = 'bob'        # 'bob', 'weave', 'vga', 'none'/'off'
    odd_field_first: bool = True

    # For masks
    enable_masking: bool = True
    mask_type: str = 'bandlimit'  # 'dynamic'/'bandlimit', 'dynamic2', 'monochrome', 'subpixel', 'coverage', 'additive', 'additive2', 'tiled', 'none'
    mask_smoothness: float = 0.0
    # Shared static mask size for subpixel and monochrome; keep the legacy name.
    subpixel_mask_pattern: int = 1  # 0=2 pixels, 1=3 pixels, 2=4 pixels, 3=5 pixels
    tiled_mask_type: str = 'slot'  # 'slot' or 'aperture'
    mask_triads: float = 550.0
    mask_amount: float = 1.0

    # Diffusion
    enable_diffusion: bool = True
    diffusion_type: str = 'gaussian'  # 'gaussian', 'box', 'none'
    blur_sigma: float = 0.05
    blur_amount: float = 0.04  # 0.15, 0.13



Config = SimulationConfig
DEFAULT_CONFIG = SimulationConfig()
