import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import taichi as ti
import taichi.math as tm

import config
import filtering
import sim_taichi
import spot_sim


def reference_zoh_horizontal(img, out_width, aspect_ratio, max_spot_size, min_spot_size):
    height, width, _ = img.shape
    delta = aspect_ratio * height / width
    min_width = min_spot_size * max_spot_size
    width_range = max_spot_size - min_width
    energy = np.zeros((height, out_width, 3), dtype=np.float64)
    moment = np.zeros_like(energy)
    for y in range(height):
        for i in range(out_width):
            x = (i + 0.5) / out_width * width * delta
            for j in range(width):
                s = img[y, j].astype(np.float64)
                w = min_width + np.sqrt(s) * width_range
                u0 = np.clip((x - j * delta) / w, -1.0, 1.0)
                u1 = np.clip((x - (j + 1) * delta) / w, -1.0, 1.0)
                f0 = u0 * (1.0 + u0 * u0 * (0.5 * np.abs(u0) - 1.0))
                f1 = u1 * (1.0 + u1 * u1 * (0.5 * np.abs(u1) - 1.0))
                piece = s * (f0 - f1)
                energy[y, i] += piece
                moment[y, i] += w * piece
    widths = np.divide(moment, energy, out=np.full_like(energy, min_width), where=energy > 0)
    return energy, np.minimum(widths, max_spot_size)


def reference_pq_horizontal(img, out_width, aspect_ratio, max_spot_size, min_spot_size):
    height, width, _ = img.shape
    delta = aspect_ratio * height / width
    min_width = min_spot_size * max_spot_size
    width_range = max_spot_size - min_width
    energy = np.zeros((height, out_width, 3), dtype=np.float64)
    moment = np.zeros_like(energy)
    for y in range(height):
        for i in range(out_width):
            tex_x = (i + 0.5) / out_width * width
            for j in range(width):
                s = img[y, j].astype(np.float64)
                w = min_width + np.sqrt(s) * width_range
                abs_dx = abs(tex_x - (j + 0.5)) * delta
                u = np.minimum(abs_dx / w, 1.0)
                weight = s * ((u * u) * (2.0 * u - 3.0) + 1.0)
                energy[y, i] += delta * weight / w
                moment[y, i] += delta * weight
    widths = np.divide(moment, energy, out=np.full_like(energy, min_width), where=energy > 0)
    return energy, np.minimum(widths, max_spot_size)


class TestPointQuadratureAndZOHFallback(unittest.TestCase):
    def _eval_horizontal(self, img, out_width, max_spot_size, min_spot_size, force_zoh=False):
        in_h, in_w, _ = img.shape
        aspect = out_width / in_h
        source = ti.Vector.field(3, dtype=ti.f32)
        energy = ti.Vector.field(3, dtype=ti.f32)
        widths = ti.Vector.field(3, dtype=ti.f32)
        builder = ti.FieldsBuilder()
        builder.dense(ti.ij, (in_h, in_w)).place(source)
        builder.dense(ti.ij, (in_h, out_width)).place(energy, widths)
        tree = builder.finalize()
        try:
            source.from_numpy(img.astype(np.float32))
            spot_sim.taichi_spot_fast2_horizontal_fragment(
                source, energy, widths, max_spot_size, min_spot_size, 0.0, 0.0, aspect,
                force_zoh=force_zoh,
            )
            return energy.to_numpy(), widths.to_numpy()
        finally:
            tree.destroy()

    def test_threshold_selects_pq_or_zoh(self):
        rng = np.random.default_rng(123)
        img = rng.uniform(0.02, 1.0, (2, 16, 3)).astype(np.float32)
        img[1, 4:7] = 0.0
        # For height=2, width=16, out_width=12 (aspect=6): delta = 6 * 2 / 16 = 0.75.
        # 1.5 * delta = 1.125.
        # Case 1 (ZOH): min_width = 0.4 * 0.9 = 0.36 < 1.125.
        e_zoh, w_zoh = self._eval_horizontal(img, 12, 0.9, 0.4)
        exp_e_zoh, exp_w_zoh = reference_zoh_horizontal(img, 12, 12 / 2, 0.9, 0.4)
        np.testing.assert_allclose(e_zoh, exp_e_zoh, rtol=2e-5, atol=1e-6)
        np.testing.assert_allclose(w_zoh, exp_w_zoh, rtol=2e-5, atol=1e-6)

        # Case 2 (PQ): out_width=2 -> aspect=1.0 -> delta = 2 / 16 = 0.125, 1.5*delta = 0.1875.
        # min_width = 0.4 * 0.9 = 0.36 >= 0.1875.
        img_wide = rng.uniform(0.02, 1.0, (2, 32, 3)).astype(np.float32)
        img_wide[1, 8:12] = 0.0
        # height=2, width=32, out_width=6 -> aspect=3 -> delta = 3 * 2 / 32 = 0.1875 -> 1.5*delta = 0.28125 <= 0.36.
        e_pq, w_pq = self._eval_horizontal(img_wide, 6, 0.9, 0.4)
        exp_e_pq, exp_w_pq = reference_pq_horizontal(img_wide, 6, 6 / 2, 0.9, 0.4)
        np.testing.assert_allclose(e_pq, exp_e_pq, rtol=2e-5, atol=1e-6)
        np.testing.assert_allclose(w_pq, exp_w_pq, rtol=2e-5, atol=1e-6)

        # Verify PQ differs from ZOH on mixed intensities when min_width >= 1.5 * delta.
        exp_e_zoh2, exp_w_zoh2 = reference_zoh_horizontal(img_wide, 6, 6 / 2, 0.9, 0.4)
        self.assertGreater(float(np.max(np.abs(e_pq - exp_e_zoh2))), 1e-3)

        # Verify force_zoh=True forces ZOH even when min_width >= 1.5 * delta.
        e_forced, w_forced = self._eval_horizontal(img_wide, 6, 0.9, 0.4, force_zoh=True)
        np.testing.assert_allclose(e_forced, exp_e_zoh2, rtol=2e-5, atol=1e-6)
        np.testing.assert_allclose(w_forced, exp_w_zoh2, rtol=2e-5, atol=1e-6)

    def test_simulate_forces_zoh_when_analog_filtering_disabled(self):
        # 2x16 image rendered to 4x6 -> delta = (6/4)*2/16 = 0.1875 -> PQ unless forced ZOH.
        rng = np.random.default_rng(7)
        img = rng.uniform(0.05, 1.0, (2, 16, 3)).astype(np.float32)
        for analog_type, enable_analog in [('linearize', True), ('rgb', False)]:
            cfg = config.SimulationConfig(
                enable_analog=enable_analog,
                analog_type=analog_type,
                scanline_method='fast',
                output_resolution=(4, 6),
                enable_masking=False,
                enable_diffusion=False,
            )
            out = sim_taichi.simulate(img, config=cfg)
            linear_img = filtering.simulate_linearize(img, cfg.gamma)
            expected = spot_sim.simulate_spot(
                linear_img, (4, 6, 3), 'fast',
                cfg.max_spot_size, cfg.min_spot_size, 0.0, 0.0,
                force_zoh=True,
            )
            np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-7)

    def test_exact_boundary_and_zero_input_in_pq(self):
        # height=2, width=24, out_width=8 -> aspect=4 -> delta = 4 * 2 / 24 = 1/3.
        # 1.5 * delta = 0.5.
        zeros = np.zeros((2, 24, 3), dtype=np.float32)
        # Just below threshold: 0.49 < 0.5 -> ZOH
        e_below, w_below = self._eval_horizontal(zeros, 8, 1.0, 0.49)
        np.testing.assert_array_equal(e_below, 0.0)
        np.testing.assert_allclose(w_below, 0.49, atol=1e-6)
        # At threshold: 0.51 >= 0.5 -> PQ
        e_above, w_above = self._eval_horizontal(zeros, 8, 1.0, 0.51)
        np.testing.assert_array_equal(e_above, 0.0)
        np.testing.assert_allclose(w_above, 0.51, atol=1e-6)


class TestNTSCChromaFilterResampling(unittest.TestCase):
    def test_defaults_use_1800_samples_and_900_internal_ntsc(self):
        cfg = config.SimulationConfig()
        self.assertEqual(cfg.samples, 1800)
        args = sim_taichi.build_parser().parse_args(['in.png', 'out.png'])
        self.assertEqual(args.samples, 1800)

        img = np.full((8, 32, 3), 0.5, dtype=np.float32)
        out_default = filtering.simulate_ntsc(img)
        out_explicit = filtering.simulate_ntsc(img, samples=1800, ntsc_samples=900)
        self.assertEqual(out_default.shape, (8, 1800, 3))
        self.assertTrue(np.all(np.isfinite(out_default)))
        np.testing.assert_array_equal(out_default, out_explicit)

    def test_resample_900_to_1800_preserves_flat_and_high_frequency_luma(self):
        periods = filtering.chroma_info_standard().periods_per_active_line
        chroma_width = 1.5 * 900.0 / periods

        # 1. Flat YUV signal -> exact RGB recovery across interior
        yuv_flat = np.empty((1, 900, 3), dtype=np.float32)
        yuv_flat[:] = [0.6, 0.05, -0.08]
        # 2. 4 MHz luma cosine wave -> Lanczos-3 preserves > 98% amplitude
        t_900 = (np.arange(900) + 0.5) / 900.0
        cycles_4mhz = 4.0e6 * filtering.ACTIVE_LINE_TIME
        yuv_wave = np.zeros((1, 900, 3), dtype=np.float32)
        yuv_wave[0, :, 0] = 0.5 + 0.25 * np.cos(2.0 * np.pi * cycles_4mhz * t_900)
        # 3. Subcarrier oscillation in chroma -> suppressed at both 1800 sample phases
        yuv_carrier = np.zeros((1, 900, 3), dtype=np.float32)
        yuv_carrier[0, :, 0] = 0.5
        yuv_carrier[0, :, 1] = 0.2 * np.cos(2.0 * np.pi * periods * t_900)
        yuv_carrier[0, :, 2] = 0.2 * np.sin(2.0 * np.pi * periods * t_900)

        field_in = ti.Vector.field(3, dtype=ti.f32)
        field_out_1800 = ti.Vector.field(3, dtype=ti.f32)
        field_out_900 = ti.Vector.field(3, dtype=ti.f32)
        builder = ti.FieldsBuilder()
        builder.dense(ti.ij, (1, 900)).place(field_in, field_out_900)
        builder.dense(ti.ij, (1, 1800)).place(field_out_1800)
        tree = builder.finalize()
        try:
            # Flat preservation (gamma=1.0 for direct linear comparison)
            field_in.from_numpy(yuv_flat)
            filtering.taichi_ntsc_decode_chroma_filter(field_in, field_out_1800, chroma_width, 1.0)
            out_flat = field_out_1800.to_numpy()[0, 60:-60]
            expected_rgb = filtering.yuv_to_linear(yuv_flat[0, :1], 1.0)
            np.testing.assert_allclose(out_flat, np.broadcast_to(expected_rgb, out_flat.shape), atol=1e-5)

            # 1:1 pass-through identity on luma
            field_in.from_numpy(yuv_wave)
            filtering.taichi_ntsc_decode_chroma_filter(field_in, field_out_900, chroma_width, 1.0)
            np.testing.assert_allclose(field_out_900.to_numpy()[0, :, 0], yuv_wave[0, :, 0], atol=2e-6)

            # 900 -> 1800 high-frequency 4 MHz luma preservation
            filtering.taichi_ntsc_decode_chroma_filter(field_in, field_out_1800, chroma_width, 1.0)
            t_1800 = (np.arange(1800) + 0.5) / 1800.0
            expected_wave = 0.5 + 0.25 * np.cos(2.0 * np.pi * cycles_4mhz * t_1800)
            np.testing.assert_allclose(
                field_out_1800.to_numpy()[0, 40:-40, 0],
                expected_wave[40:-40],
                atol=5e-3,
            )

            # Subcarrier ripple rejection in chroma after resampling
            field_in.from_numpy(yuv_carrier)
            filtering.taichi_ntsc_decode_chroma_filter(field_in, field_out_1800, chroma_width, 1.0)
            out_carrier = field_out_1800.to_numpy()[0, 60:-60]
            self.assertLess(float(np.max(np.abs(out_carrier - 0.5))), 5e-4)
        finally:
            tree.destroy()


def run_isolated_tests():
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'unittest',
            'test.test_resampling_and_quadrature.TestPointQuadratureAndZOHFallback',
            'test.test_resampling_and_quadrature.TestNTSCChromaFilterResampling',
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite([unittest.FunctionTestCase(run_isolated_tests)])


if __name__ == '__main__':
    unittest.main()
