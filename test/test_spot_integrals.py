import unittest

import numpy as np
import sim_taichi  # Initialize Taichi before defining kernels.
import spot_sim
import taichi as ti


@ti.kernel
def evaluate_integrals(cases: ti.types.ndarray(dtype=ti.f32, ndim=2),
                       output: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    for i in range(cases.shape[0]):
        sample = ti.Vector([cases[i, 0], cases[i, 1], cases[i, 2]])
        width = ti.Vector([cases[i, 3], cases[i, 4], cases[i, 5]])
        cubic = spot_sim.spot_fast2_horizontal_cubic(sample, width, cases[i, 6], cases[i, 7], cases[i, 8])
        cosine = spot_sim.spot_fast2_horizontal(sample, width, cases[i, 6], cases[i, 7], cases[i, 8])
        for c in ti.static(range(3)):
            output[i, c] = cubic[c]
            output[i, c + 3] = cosine[c]


QUADRATURE_NODES, QUADRATURE_WEIGHTS = np.polynomial.legendre.leggauss(12)


def integrate_spot(sample, width, x, x0, x1, cubic):
    """Integrate the normalized spot profile directly, splitting at its knots."""
    result = np.zeros(3)
    for c in range(3):
        lo, hi = max(x0, x - width[c]), min(x1, x + width[c])
        if hi <= lo:
            continue
        cuts = [lo, x, hi] if lo < x < hi else [lo, hi]
        for a, b in zip(cuts, cuts[1:]):
            t = (a + b) / 2 + (b - a) / 2 * QUADRATURE_NODES
            u = np.abs((x - t) / width[c])
            profile = 1 - 3 * u**2 + 2 * u**3 if cubic else 0.5 + 0.5 * np.cos(np.pi * u)
            result[c] += sample[c] / width[c] * (b - a) / 2 * np.dot(QUADRATURE_WEIGHTS, profile)
    return result


def reference_horizontal(img, out_width, aspect_ratio, cubic):
    height, width, _ = img.shape
    delta = aspect_ratio * height / width
    energy = np.zeros((height, out_width, 3))
    moment = np.zeros_like(energy)
    for y in range(height):
        for i in range(out_width):
            x = (i + 0.5) / out_width * width * delta
            for j in range(width):
                sample = img[y, j].astype(np.float64)
                spot_width = 0.36 + 0.54 * np.sqrt(sample)
                piece = integrate_spot(sample, spot_width, x, j * delta, (j + 1) * delta, cubic)
                energy[y, i] += piece
                moment[y, i] += spot_width * piece
    widths = np.divide(moment, energy, out=np.full_like(energy, 0.36), where=energy > 0)
    return energy, np.minimum(widths, 0.9)


class TestSpotIntegrals(unittest.TestCase):
    def test_integrals_match_quadrature_and_preserve_energy(self):
        sample = [0.2, 0.65, 1.0]
        width = [0.36, 0.5, 0.9]
        cases = [sample + width + [0.0, a, b] for a, b in
                 [(-2.0, 2.0), (-2.0, 0.0), (0.0, 2.0), (1.1, 2.0), (0.1, 0.1)]]
        rng = np.random.default_rng(42)
        for _ in range(100):
            a, b = sorted(rng.uniform(-2.0, 2.0, 2))
            cases.append([*rng.uniform(0.0, 1.0, 3), *rng.uniform(0.18, 1.0, 3),
                          rng.uniform(-1.0, 1.0), a, b])
        cases = np.array(cases, dtype=np.float32)
        output = np.empty((len(cases), 6), dtype=np.float32)
        evaluate_integrals(cases, output)
        self.assertTrue(np.isfinite(output).all())
        for cubic, offset in [(True, 0), (False, 3)]:
            actual = output[:, offset:offset + 3]
            expected = np.array([integrate_spot(row[:3], row[3:6], *row[6:], cubic)
                                 for row in cases.astype(np.float64)])
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-7)
            self.assertTrue((actual >= -2e-7).all())
            self.assertTrue((actual <= cases[:, :3] + 2e-7).all())
            np.testing.assert_allclose(actual[0], sample, atol=1e-7)
            np.testing.assert_allclose(actual[1] + actual[2], sample, atol=1e-7)
            np.testing.assert_array_equal(actual[3:5], 0.0)

    def test_fast_models_match_reference_at_noninteger_scale(self):
        img = np.random.default_rng(7).uniform(0.01, 1.0, (4, 8, 3)).astype(np.float32)
        img[0] = 0.0
        img[:, :, 2] = 0.0
        out_height, out_width = 13, 19
        source = ti.Vector.field(n=3, dtype=float, shape=img.shape[:2])
        source.from_numpy(img)
        energy = ti.Vector.field(n=3, dtype=float, shape=(4, out_width))
        widths = ti.Vector.field(n=3, dtype=float, shape=(4, out_width))
        for method, cubic in [('fast', True), ('fast-cosine', False)]:
            with self.subTest(method=method):
                expected_energy, expected_widths = reference_horizontal(img, out_width, out_width / out_height, cubic)
                args = (source, energy, widths, 0.9, 0.4, 0.0, 0.0, out_width / out_height)
                if cubic:
                    # Omitting the model flag must select the cubic integral.
                    spot_sim.taichi_spot_fast2_horizontal_fragment(*args)
                else:
                    spot_sim.taichi_spot_fast2_horizontal_fragment(*args, cubic_horizontal=False)
                np.testing.assert_allclose(energy.to_numpy(), expected_energy, rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(widths.to_numpy(), expected_widths, rtol=2e-5, atol=2e-6)

                expected = np.zeros((out_height, out_width, 3))
                for y in range(out_height):
                    for j in range(img.shape[0]):
                        distance = j + 0.5 - (y + 0.5) / out_height * img.shape[0]
                        u = np.minimum(abs(distance) / expected_widths[j], 1.0)
                        profile = 1 - 3 * u**2 + 2 * u**3 if cubic else 0.5 + 0.5 * np.cos(np.pi * u)
                        expected[y] += expected_energy[j] * 0.9 / expected_widths[j] * profile
                actual = spot_sim.simulate_spot(img, expected.shape, method, 0.9, 0.4, 0.0, 0.0,
                                                interlacing_type='off')
                self.assertEqual(actual.shape, expected.shape)
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)

    def test_cosine_vertical_matches_analytical_in_all_line_modes(self):
        # Constant colors have a fixed spot width, so the separable cosine
        # approximation must agree with the full analytical cosine model.
        img = np.full((480, 8, 3), [0.02, 0.1, 0.5], dtype=np.float32)
        source = ti.Vector.field(n=3, dtype=float, shape=img.shape[:2])
        source.from_numpy(img)
        energy = ti.Vector.field(n=3, dtype=float, shape=(480, 19))
        widths = ti.Vector.field(n=3, dtype=float, shape=energy.shape)
        output = ti.Vector.field(n=3, dtype=float, shape=(37, 19))
        for mode in [0, 1, 2, 3]:
            spot_sim.taichi_spot_fast2_horizontal_fragment(
                source, energy, widths, 0.9, 0.4, 0.0, 0.0, 19 / 37,
                mode, cubic_horizontal=False)
            for frame in [0, 1]:
                with self.subTest(mode=mode, frame=frame):
                    spot_sim.taichi_spot_analytical_fragment(
                        source, output, 0.9, 0.4, 0.0, 0.0, mode, 1, frame)
                    expected = output.to_numpy()
                    spot_sim.taichi_spot_fast2_vertical_fragment(
                        energy, widths, output, 0.9, 0.4, 0.0, 0.0,
                        mode, 1, frame, cubic_vertical=False)
                    actual = output.to_numpy()
                    self.assertTrue(np.isfinite(actual).all())
                    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)

    def test_cosine_method_is_available_from_cli(self):
        args = sim_taichi.build_parser().parse_args(['in.png', 'out.png', '--scanline-method', 'fast-cosine'])
        self.assertEqual(args.scanline_method, 'fast-cosine')


if __name__ == '__main__':
    unittest.main()
