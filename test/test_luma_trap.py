import contextlib
import io
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import taichi as ti
import taichi.math as tm

import sim_taichi
import filtering


@ti.kernel
def evaluate_trap_primitive(
    positions: ti.template(), values: ti.template(), coefficients: tm.vec3,
):
    for i in positions:
        values[i] = filtering.luma_trap_primitive(positions[i], coefficients)


def check_ntsc_pipeline():
    image = np.random.default_rng(42).uniform(0.1, 0.9, (4, 32, 3)).astype(np.float32)
    baseline = filtering.simulate_ntsc(image, samples=64)
    bypass = filtering.simulate_ntsc(image, samples=64, ntsc_luma_trap_depth=0.0)
    np.testing.assert_array_equal(bypass, baseline)
    trapped = filtering.simulate_ntsc(image, samples=64, ntsc_luma_trap_depth=32.0)
    dispatched = filtering.simulate_analog(
        image, analog_type='ntsc', samples=64, ntsc_luma_trap_depth=32.0,
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
        source = ti.Vector.field(3, dtype=ti.f32)
        output = ti.Vector.field(3, dtype=ti.f32)
        builder = ti.FieldsBuilder()
        builder.dense(ti.ij, image.shape[:2]).place(source)
        builder.dense(ti.ij, (image.shape[0], samples)).place(output)
        tree = builder.finalize()
        try:
            source.from_numpy(image.astype(np.float32))
            if depth == 0.0:
                filtering.taichi_filter_fragment(source, output, L)
            else:
                coefficients = filtering.luma_trap_coefficients(bandwidth, depth)
                filtering.taichi_filter_fragment_luma_trap(
                    source, output, L, 3.0 / periods, tm.vec3(coefficients),
                )
            return output.to_numpy()
        finally:
            tree.destroy()

    def test_depth_range(self):
        for depth in (-1.0, 32.01, np.nan, np.inf, -np.inf):
            with self.subTest(depth=depth):
                with self.assertRaises(ValueError):
                    filtering.luma_trap_coefficients(4.0e6, depth)
        for bandwidth in (0.6e6, 1.2e6, filtering.CHROMA_SUBCARRIER, 4.0e6, 6.0e6):
            with self.subTest(bandwidth=bandwidth):
                for depth in (0.0, 6.0, 32.0):
                    coefficients = filtering.luma_trap_coefficients(bandwidth, depth)
                    self.assertEqual(coefficients.shape, (3,))
                    self.assertTrue(np.all(np.isfinite(coefficients)))

    def test_primitive_is_odd_and_compact(self):
        x = np.linspace(-1.25, 1.25, 1001, dtype=np.float32)
        positions = ti.field(dtype=ti.f32)
        values = ti.field(dtype=ti.f32)
        builder = ti.FieldsBuilder()
        builder.dense(ti.i, x.shape[0]).place(positions, values)
        tree = builder.finalize()
        try:
            positions.from_numpy(x)
            for bandwidth in (0.6e6, filtering.CHROMA_SUBCARRIER, 4.0e6, 6.0e6):
                with self.subTest(bandwidth=bandwidth):
                    coefficients = filtering.luma_trap_coefficients(bandwidth, 32.0)
                    evaluate_trap_primitive(positions, values, tm.vec3(coefficients))
                    result = values.to_numpy()
                    self.assertTrue(np.all(np.isfinite(result)))
                    np.testing.assert_allclose(result, -result[::-1], atol=2e-7, rtol=0)
                    np.testing.assert_array_equal(result[np.abs(x) >= 1.0], 0.0)
            evaluate_trap_primitive(positions, values, tm.vec3(0.0))
            np.testing.assert_array_equal(values.to_numpy(), 0.0)
        finally:
            tree.destroy()

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
        # the actual integrated filter rather than only its coefficient formula.
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
                    self.assertAlmostEqual(attenuation, depth, delta=0.08)

    def test_signed_ringing_survives_encoding_filter(self):
        image = np.zeros((1, 2048, 3), dtype=np.float32)
        image[:, 1024:, 0] = 1.0
        output = self.filter_yuv(image, 32.0)
        self.assertGreater(float(output[..., 0].max()), 1.02)
        self.assertLess(float(output[..., 0].max()), 1.05)
        self.assertLess(float(output[..., 0].min()), -0.02)
        self.assertGreater(float(output[..., 0].min()), -0.05)

    def test_ntsc_default_is_exact_bypass_and_dispatches_depth(self):
        check_ntsc_pipeline()

    def test_cli_rejects_invalid_depth(self):
        parser = sim_taichi.build_parser()
        for value in ('-1', '32.1', 'nan', 'inf', '-inf'):
            with self.subTest(value=value):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parser.parse_args(['in.png', 'out.png', '--ntsc-luma-trap-depth', value])


def run_isolated_luma_trap_tests():
    # Compiled Taichi field metadata outlives explicit tree cleanup. Keep the
    # new cases in a separate runtime so discovery does not exhaust the
    # field-tree limit in the existing large simulation suite.
    result = subprocess.run(
        [sys.executable, '-m', 'unittest', 'test.test_luma_trap.TestLumaTrap'],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite([unittest.FunctionTestCase(run_isolated_luma_trap_tests)])


if __name__ == '__main__':
    unittest.main()
