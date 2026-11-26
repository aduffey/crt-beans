# AGENTS.md

## Project overview

`crt-beans` is a work-in-progress CRT simulation with two related implementations:

- The Python/NumPy/Taichi code in the repository root is the reference model and experimentation harness.
- `slang-shaders/` contains the production Vulkan GLSL shader passes and RetroArch/librashader presets.

Favor image quality, numerical correctness, and GPU performance. The project deliberately keeps its user-facing feature set and parameter set small.

## Repository map

- `sim_taichi.py`: reference pipeline and command-line entry point.
- `config.py`: reference-model defaults and supported modes.
- `filtering.py`, `spot_sim.py`, `masking.py`, `diffusion.py`: simulation stages.
- `slang-shaders/*.slangp`: complete preset pipelines; pass numbers and aliases are significant.
- `slang-shaders/shaders/`: reusable `.slang` passes and `.inc` files.
- `slang-shaders/shaders/docs/parameters.md`: user-facing shader parameter documentation.
- `test/test_simulation.py`: automated reference-model coverage.
- `experiments/`: exploratory code and results; do not treat every file here as production code.
- `test-images/` and root-level generated images/PDFs: visual references and local research artifacts. Do not modify, delete, or commit them unless the task explicitly requires it.

## Setup and verification

Use the locked `uv` environment and Python 3.12:

```sh
uv sync
uv run python -m unittest discover -s test -p 'test_*.py'
```

Run a focused test while iterating when practical, then run the complete suite before handing off changes:

```sh
uv run python -m unittest test.test_simulation.TestAnalogSimulation
```

There is no configured formatter or linter. Follow the surrounding style and keep unrelated formatting out of patches.

For image-level checks, invoke the reference simulation at 1440x1080 (4:3) by default to match a CRT's aspect ratio. Full-resolution output is expensive, but smaller resolutions are not representative of the intended rendering. The CLI accepts resolution as height followed by width:

```sh
uv run python sim_taichi.py INPUT.png /tmp/crt-beans-output.png --output-resolution 1080 1440
```

Automated tests do not validate rendered Slang output. Shader changes require compilation/runtime or visual validation in a RetroArch- or librashader-compatible frontend when one is available. If it is not available, say so in the handoff rather than claiming shader validation.

## Change guidelines

- Preserve parity between the reference model and the Slang implementation when changing shared algorithms, constants, defaults, or parameter meanings. Update both sides or document why they intentionally differ.
- Add or update focused unit tests for reference-model behavior. Numerical tests should check finite values, shapes, invariants, and tolerances rather than relying only on generated screenshots.
- Treat color spaces explicitly. The pipeline generally performs analog, scanline, mask, and diffusion work in linear light and converts to sRGB at output; do not introduce implicit gamma conversions.
- Keep hot shader paths efficient. Avoid extra passes, dependent texture reads, branches, or larger intermediate formats unless the quality gain is demonstrated.
- When changing a preset, recount `shaders`, keep every `shaderN` and its `filter_*N`, `wrap_modeN`, `scale_*N`, and `aliasN` settings aligned, and verify references to earlier pass aliases.
- When adding or changing shader parameters, keep the declarations, defaults, ranges, presets, Python configuration/CLI where applicable, and `slang-shaders/shaders/docs/parameters.md` consistent.
- Preserve SPDX headers on shader/include files.
- Keep generated comparison images and experiment outputs out of a change unless they are requested evidence or intentional documentation assets.

## Slang constraints

The shader sources are Vulkan GLSL compiled to SPIR-V. Follow the existing resource layout:

- Separate vertex and fragment code with `#pragma stage vertex` and `#pragma stage fragment`.
- Use descriptor set `0` and explicit bindings.
- Use at most one UBO and one push-constant block per shader; keep push constants at or below 128 bytes.
- Retain the standard vertex inputs (`Position` at location 0 and `TexCoord` at location 1) and fragment output (`FragColor` at location 0).
- Prefer size uniforms and normalized coordinates over `gl_FragCoord`; use explicit texel quantization for nearest-neighbor behavior at non-integer scales.
- Be deliberate about render-target formats, sampling filters, wrap modes, and linear-versus-sRGB data at every pass boundary.

## Working-tree hygiene

This repository may contain many untracked research documents, experiment outputs, and comparison images. They belong to the user. Inspect `git status --short` before and after work, do not clean the tree, and limit edits to files required by the task.

For releases, `release.sh NAME` packages only `slang-shaders/` into `NAME.zip`. Do not run it unless a release artifact is requested.
