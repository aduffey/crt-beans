This is a work-in-progress CRT simulation, prototyped in [https://www.taichi-lang.org/](Taichi) and implemented as a set of [Slang shaders](https://github.com/libretro/slang-shaders) for [RetroArch](https://github.com/libretro/RetroArch) or [librashader](https://github.com/SnowflakePowered/librashader)-compatible software.

Currently, supported feature include:
* A configurable low-pass filter to simulate the limited bandwidth of an analog connection.
* Optional YIQ encoding to simulate a composite or s-video connection (actual NTSC encoding/decoding is not currently supported).
* Configurable scanline simulation.
* Configurable overscan cropping.
* "Glow" simulation, recreating the glowing areas around bright parts of the screen caused by the scattering of light in the glass.

The Slang shaders have been tested to run in 4k at 60 FPS on a Ryzen 7000 integrated GPU in their default configuration.
