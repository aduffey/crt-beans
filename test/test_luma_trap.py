import contextlib
import io
import unittest

import numpy as np
import taichi.math as tm

import sim_taichi
import filtering


def check_ntsc_pipeline():
    image = np.random.default_rng(42).uniform(0.1, 0.9, (4, 32, 3)).astype(np.float32)
    baseline = filtering.simulate_ntsc(image, samples=900)
    bypass = filtering.simulate_ntsc(image, samples=900, ntsc_luma_trap_depth=0.0)
    np.testing.assert_array_equal(bypass, baseline)
    trapped = filtering.simulate_ntsc(image, samples=900, ntsc_luma_trap_depth=32.0)
    dispatched = filtering.simulate_analog(
        image, analog_type='ntsc', samples=900, ntsc_luma_trap_depth=32.0,
    )
    np.testing.assert_array_equal(dispatched, trapped)
    assert np.all(np.isfinite(trapped))
    assert np.max(np.abs(trapped - baseline)) > 1e-4


class TestLumaTrap(unittest.TestCase):
    @staticmethod
    def filter_yuv(image, depth, periods=None, bandwidth=4.0e6, samples=900):
        if periods is None:
            periods = filtering.chroma_info_standard().periods_per_active_line
        active_time = periods / filtering.CHROMA_SUBCARRIER
        L = tm.vec3(
            1.0 / (2.0 * bandwidth * active_time),
            1.0 / (2.0 * 1.2e6 * active_time),
            1.0 / (2.0 * 1.2e6 * active_time),
        )
        strength = filtering.luma_trap_strength(depth)
        source = np.ascontiguousarray(image, dtype=np.float32)
        filtered = np.empty((image.shape[0], samples, 3), dtype=np.float32)
        output = np.empty_like(filtered)
        filtering.taichi_filter_fragment(source, filtered, L)
        filtering.taichi_luma_trap_fragment(filtered, output, periods, strength)
        return output

    def test_depth_range(self):
        for depth in (-1.0, 32.01, np.nan, np.inf, -np.inf):
            with self.subTest(depth=depth):
                with self.assertRaises(ValueError):
                    filtering.luma_trap_strength(depth)
        self.assertEqual(filtering.luma_trap_strength(0.0), 0.0)
        for depth in (6.0, 16.0, 32.0):
            with self.subTest(depth=depth):
                strength = filtering.luma_trap_strength(depth)
                self.assertTrue(np.isfinite(strength))
                self.assertAlmostEqual(1.0 - strength, 10.0 ** (-depth / 20.0), places=6)

    def test_impulse_response_is_symmetric_and_compact(self):
        samples = 900
        center = samples // 2
        impulse = np.zeros((1, samples, 3), dtype=np.float32)
        impulse[0, center, 0] = 1.0
        output = np.empty_like(impulse)
        for periods in (filtering.chroma_info_standard().periods_per_active_line,
                        filtering.chroma_info_nes(256).periods_per_active_line):
            with self.subTest(periods=periods):
                strength = filtering.luma_trap_strength(32.0)
                filtering.taichi_luma_trap_fragment(impulse, output, periods, strength)
                response = output[0, :, 0]
                radius = int(np.floor(3.0 * samples / periods))
                self.assertTrue(np.all(np.isfinite(response)))
                np.testing.assert_allclose(
                    response[center - radius:center + radius + 1],
                    response[center - radius:center + radius + 1][::-1],
                    atol=2e-7, rtol=0,
                )
                np.testing.assert_array_equal(response[:center - radius], 0.0)
                np.testing.assert_array_equal(response[center + radius + 1:], 0.0)
        filtering.taichi_luma_trap_fragment(impulse, output, periods, 0.0)
        np.testing.assert_array_equal(output, impulse)

    def test_flat_color_and_chroma_are_preserved(self):
        image = np.empty((2, 320, 3), dtype=np.float32)
        image[:] = [0.42, 0.11, -0.17]
        baseline = self.filter_yuv(image, 0.0)
        for depth in (6.0, 16.0, 32.0):
            with self.subTest(depth=depth):
                output = self.filter_yuv(image, depth)
                interior = output[:, 40:-40]
                expected = np.broadcast_to(image[0, 0], interior.shape)
                np.testing.assert_allclose(interior, expected, atol=2e-6, rtol=0)
                np.testing.assert_allclose(output[..., 1:], baseline[..., 1:], atol=2e-6, rtol=0)

    def test_carrier_attenuation_is_in_decibels(self):
        # Oversample the source so pixel-hold aliases are negligible. Measure
        # the cascaded low-pass and discrete luma-trap passes.
        width, samples = 4096, 900
        source_x = (np.arange(width) + 0.5) / width
        output_x = (np.arange(samples) + 0.5) / samples
        for periods in (filtering.chroma_info_standard().periods_per_active_line,
                        filtering.chroma_info_nes(256).periods_per_active_line):
            image = np.zeros((1, width, 3), dtype=np.float32)
            image[0, :, 0] = 0.5 + 0.1 * np.cos(2.0 * np.pi * periods * source_x)
            phase = 2.0 * np.pi * periods * output_x[50:-50]
            basis = np.column_stack((np.ones(phase.shape), np.cos(phase), np.sin(phase)))

            def amplitude(output):
                fit = np.linalg.lstsq(basis, output[0, 50:-50, 0], rcond=None)[0]
                return np.hypot(fit[1], fit[2])

            baseline = amplitude(self.filter_yuv(image, 0.0, periods=periods))
            for depth in (6.0, 16.0, 32.0):
                with self.subTest(periods=periods, depth=depth):
                    output = self.filter_yuv(image, depth, periods=periods)
                    attenuation = -20.0 * np.log10(amplitude(output) / baseline)
                    self.assertAlmostEqual(attenuation, depth, delta=0.01)

    def test_signed_ringing_survives_encoding_filter(self):
        image = np.zeros((1, 2048, 3), dtype=np.float32)
        image[:, 1024:, 0] = 1.0
        output = self.filter_yuv(image, 32.0)
        self.assertGreater(float(output[..., 0].max()), 1.02)
        self.assertLess(float(output[..., 0].max()), 1.06)
        self.assertLess(float(output[..., 0].min()), -0.02)
        self.assertGreater(float(output[..., 0].min()), -0.06)

    def test_sub_nyquist_bypasses_without_nan_or_blowup(self):
        strength = filtering.luma_trap_strength(32.0)
        rng = np.random.default_rng(7)
        cases = [
            ('radius_zero', 50, 170.6667),
            ('width_one', 50, 150.0),
            ('s_half', 100, 200.0),
            ('s_one', 200, 200.0),
            ('near_norm_root', 200, 200.0 / 1.14223),
            ('zero_periods', 64, 0.0),
        ]
        for label, samples, periods in cases:
            with self.subTest(case=label):
                yuv = rng.uniform(0.1, 0.9, (2, samples, 3)).astype(np.float32)
                result = np.empty_like(yuv)
                filtering.taichi_luma_trap_fragment(yuv, result, periods, strength)
                self.assertTrue(np.all(np.isfinite(result)))
                np.testing.assert_array_equal(result, yuv)

        # At exact Nyquist (s = 2.0, width = 6.0), the trap remains well-conditioned.
        samples = 400
        periods = 200.0
        yuv = np.zeros((1, samples, 3), dtype=np.float32)
        yuv[0, ::2, 0] = 0.8
        yuv[0, 1::2, 0] = 0.2
        result = np.empty_like(yuv)
        filtering.taichi_luma_trap_fragment(yuv, result, periods, strength)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertLess(float(np.std(result[0, 20:-20, 0])), 0.02)

        # End-to-end simulate_ntsc at sub-Nyquist sample count stays finite and bypasses trap.
        small_img = rng.uniform(0.1, 0.9, (4, 32, 3)).astype(np.float32)
        baseline = filtering.simulate_ntsc(small_img, samples=50, ntsc_luma_trap_depth=0.0)
        trapped = filtering.simulate_ntsc(small_img, samples=50, ntsc_luma_trap_depth=32.0)
        self.assertTrue(np.all(np.isfinite(trapped)))
        np.testing.assert_array_equal(trapped, baseline)

    def test_ntsc_default_is_exact_bypass_and_dispatches_depth(self):
        check_ntsc_pipeline()

    def test_cli_rejects_invalid_depth(self):
        parser = sim_taichi.build_parser()
        for value in ('-1', '32.1', 'nan', 'inf', '-inf'):
            with self.subTest(value=value):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parser.parse_args(['in.png', 'out.png', '--ntsc-luma-trap-depth', value])


if __name__ == '__main__':
    unittest.main()
