"""Regression checks for the cubic spot and three cosine width approximations."""
import unittest

import numpy as np
import taichi as ti
import taichi.math as tm

import sim_taichi  # Initialize Taichi before creating fields, as in the main tests.
import spot_sim


@ti.func
def cosine_horizontal(uv, source: ti.template(), source_size, output_size, delta, maximum, minimum):
    return spot_sim.spot_sim_fast2_horizontal(
        uv, source, source_size, output_size, delta, maximum, minimum, cubic_horizontal=False)


@ti.kernel
def evaluate_horizontal(source: ti.types.ndarray(dtype=tm.vec3, ndim=2),
                        energy: ti.types.ndarray(dtype=tm.vec3, ndim=2),
                        widths: ti.types.ndarray(dtype=tm.vec3, ndim=2),
                        implementation: ti.template(), maximum: float, minimum: float):
    ih, iw = source.shape
    oh, ow = energy.shape
    source_size = tm.vec4(iw, ih, 1 / iw, 1 / ih)
    output_size = tm.vec4(ow, oh, 1 / ow, 1 / oh)
    delta = (4 / 3) * ih / iw
    for y, x in energy:
        uv = tm.vec2((x + 0.5) / ow, (y + 0.5) / oh)
        energy[y, x], widths[y, x] = implementation(
            uv, source, source_size, output_size, delta, maximum, minimum)


class TestSpotHorizontalVariants(unittest.TestCase):
    def evaluate(self, image):
        source = np.ascontiguousarray(image, dtype=np.float32)
        energy = np.empty((image.shape[0], 120, 3), dtype=np.float32)
        widths = np.empty_like(energy)
        results = []
        for implementation in [spot_sim.spot_sim_fast2_horizontal,
                               cosine_horizontal,
                               spot_sim.spot_sim_fast3_horizontal,
                               spot_sim.spot_sim_fast4_horizontal]:
            evaluate_horizontal(source, energy, widths, implementation, .9, .4)
            result = energy.copy(), widths.copy()
            self.assertTrue(np.isfinite(result[0]).all())
            self.assertTrue(np.isfinite(result[1]).all())
            self.assertTrue(np.all(result[1] > 0))
            self.assertTrue(np.all(result[1] <= .9))
            results.append(result)
        return results

    def test_black_has_zero_energy_and_finite_fallback_width(self):
        for energy, widths in self.evaluate(np.zeros((4, 12, 3), np.float32)):
            np.testing.assert_array_equal(energy, 0)
            np.testing.assert_allclose(widths, .9 * .4, atol=1e-7)

    def test_constant_input_preserves_its_width(self):
        # The expected width is .9 * (.4 + .6 * sqrt(.25)) = .63.
        for energy, widths in self.evaluate(np.full((4, 12, 3), .25, np.float32)):
            self.assertTrue(np.all(energy > 0))
            np.testing.assert_allclose(widths, .63, atol=2e-6)

    def test_mixed_intensities_share_energy_and_order_width_means(self):
        row = np.array([0, .01, .09, .25, .64, 1, .64, .25, .09, .01, 0, 1], np.float32)
        image = np.broadcast_to(row[None, :, None], (4, 12, 3)).copy()
        _, arithmetic, rms, harmonic = self.evaluate(image)
        np.testing.assert_array_equal(arithmetic[0], rms[0])
        np.testing.assert_array_equal(arithmetic[0], harmonic[0])
        self.assertTrue(np.all(harmonic[1] <= arithmetic[1] + 2e-6))
        self.assertTrue(np.all(arithmetic[1] <= rms[1] + 2e-6))
        self.assertGreater(float(np.max(arithmetic[1] - harmonic[1])), .001)
        self.assertGreater(float(np.max(rms[1] - arithmetic[1])), .001)

    def test_dark_constant_matches_analytical_for_each_variant(self):
        source = np.full((4, 12, 3), .1 ** 2.4, np.float32)
        energy = np.empty((4, 120, 3), dtype=np.float32)
        widths = np.empty_like(energy)
        output = np.empty((90, 120, 3), dtype=np.float32)
        for maximum, minimum in [(.9, .4), (.5, .8)]:
            spot_sim.taichi_spot_analytical_fragment(source, output, maximum, minimum, 0., 0.)
            reference = output.copy()
            # The analytical reference uses a raised-cosine spot on both axes.
            for implementation in [cosine_horizontal,
                                   spot_sim.spot_sim_fast3_horizontal,
                                   spot_sim.spot_sim_fast4_horizontal]:
                with self.subTest(maximum=maximum, minimum=minimum, variant=implementation.__name__):
                    evaluate_horizontal(source, energy, widths, implementation, maximum, minimum)
                    spot_sim.taichi_spot_fast2_vertical_fragment(
                        energy, widths, output, maximum, minimum, 0., 0., cubic_vertical=False)
                    self.assertTrue(np.isfinite(output).all())
                    np.testing.assert_allclose(output, reference, atol=1e-7, rtol=1e-4)

    def test_zero_widths_and_energy_produce_finite_black(self):
        energy = np.zeros((4, 12, 3), dtype=np.float32)
        widths = np.zeros_like(energy)
        output = np.empty((90, 120, 3), dtype=np.float32)
        spot_sim.taichi_spot_fast2_vertical_fragment(energy, widths, output, .9, .4, 0., 0.)
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_array_equal(output, 0.)


if __name__ == '__main__':
    unittest.main()
