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


def integrate_vertical_cubic(energy, width, distance_y, half_h, max_spot_size):
    """Integrate the vertical cubic scanline profile over [distance_y - half_h, distance_y + half_h]."""
    result = np.zeros(3)
    for c in range(3):
        w = width[c]
        lo = max(distance_y - half_h, -w)
        hi = min(distance_y + half_h, w)
        if hi <= lo:
            continue
        cuts = [lo, 0.0, hi] if lo < 0.0 < hi else [lo, hi]
        for a, b in zip(cuts, cuts[1:]):
            t = (a + b) / 2 + (b - a) / 2 * QUADRATURE_NODES
            u = np.abs(t) / w
            profile = 1.0 - 3.0 * u**2 + 2.0 * u**3
            result[c] += energy[c] * max_spot_size / w * (b - a) / 2 * np.dot(QUADRATURE_WEIGHTS, profile)
    return result / (2.0 * half_h)


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
        source = np.ascontiguousarray(img, dtype=np.float32)
        energy = np.empty((4, out_width, 3), dtype=np.float32)
        widths = np.empty((4, out_width, 3), dtype=np.float32)
        for method, cubic, integrated in [
            ('fast', True, True),
            ('fast-point', True, False),
            ('fast-cosine', False, False),
        ]:
            with self.subTest(method=method):
                expected_energy, expected_widths = reference_horizontal(img, out_width, out_width / out_height, cubic)
                args = (source, energy, widths, 0.9, 0.4, 0.0, 0.0, out_width / out_height)
                if cubic:
                    # Omitting the model flag must select the cubic integral.
                    spot_sim.taichi_spot_fast2_horizontal_fragment(*args)
                else:
                    spot_sim.taichi_spot_fast2_horizontal_fragment(*args, cubic_horizontal=False)
                np.testing.assert_allclose(energy, expected_energy, rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(widths, expected_widths, rtol=2e-5, atol=2e-6)

                expected = np.zeros((out_height, out_width, 3))
                pixel_height = img.shape[0] / out_height
                for y in range(out_height):
                    for j in range(img.shape[0]):
                        distance = j + 0.5 - (y + 0.5) / out_height * img.shape[0]
                        if integrated:
                            for x in range(out_width):
                                expected[y, x] += integrate_vertical_cubic(
                                    expected_energy[j, x], expected_widths[j, x],
                                    distance, 0.5 * pixel_height, 0.9)
                        else:
                            u = np.minimum(abs(distance) / expected_widths[j], 1.0)
                            profile = 1 - 3 * u**2 + 2 * u**3 if cubic else 0.5 + 0.5 * np.cos(np.pi * u)
                            expected[y] += expected_energy[j] * 0.9 / expected_widths[j] * profile
                actual = spot_sim.simulate_spot(img, expected.shape, method, 0.9, 0.4, 0.0, 0.0,
                                                interlacing_type='off')
                self.assertEqual(actual.shape, expected.shape)
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)

    def test_vertical_integration_matches_quadrature_all_modes_and_overscan(self):
        rng = np.random.default_rng(19)
        cases = [
            # (src_h, out_h, mode, overscan_v, max_spot, frame, odd_first)
            (8, 13, 3, 0.0, 0.9, 0, 1),       # progressive non-integer upscale
            (8, 6, 3, 0.0, 1.0, 0, 1),        # progressive downscale (>2 overlapping lines)
            (8, 17, 3, 0.08, 0.95, 0, 1),     # progressive with vertical overscan
            (8, 11, 2, 0.05, 0.9, 0, 1),      # VGA line doubling (< 350 lines)
            (310, 37, 0, 0.06, 0.9, 0, 1),    # bob odd field
            (310, 37, 0, 0.06, 0.9, 1, 1),    # bob even field
            (310, 37, 0, 0.06, 0.9, 0, 0),    # bob even field first
            (310, 37, 1, 0.04, 1.0, 0, 1),    # weave (>2 field lines per pixel)
            (350, 37, 2, 0.0, 0.9, 0, 1),     # VGA progressive fallback (>= 350 lines)
        ]
        for src_h, out_h, mode, overscan_v, max_spot, frame, odd_first in cases:
            with self.subTest(src_h=src_h, out_h=out_h, mode=mode, overscan_v=overscan_v,
                              max_spot=max_spot, frame=frame, odd_first=odd_first):
                out_w = 3
                energy = rng.uniform(0.0, 1.0, (src_h, out_w, 3)).astype(np.float32)
                widths = rng.uniform(0.4 * max_spot, max_spot, (src_h, out_w, 3)).astype(np.float32)
                actual = np.empty((out_h, out_w, 3), dtype=np.float32)
                spot_sim.taichi_spot_fast2_vertical_fragment(
                    energy, widths, actual, max_spot, 0.4, 0.0, overscan_v,
                    mode, odd_first, frame, cubic_vertical=True, integrate_vertical=True)
                self.assertTrue(np.isfinite(actual).all())

                # Verify selected rows against direct Gauss-Legendre quadrature over all scanlines.
                sample_rows = [0, 1, out_h // 2, out_h - 2, out_h - 1]
                src_h_pix = (1.0 - overscan_v) * src_h / out_h
                for y_out in sample_rows:
                    v = (1.0 - overscan_v) * ((y_out + 0.5) / out_h - 0.5) + 0.5
                    y_src = v * src_h
                    expected_row = np.zeros((out_w, 3), dtype=np.float64)
                    if mode == 2 and src_h < 350:
                        pix_h = 2.0 * src_h_pix
                        for k in range(2 * src_h):
                            d = (k + 0.5) - 2.0 * y_src
                            if abs(d) < max_spot + 0.5 * pix_h:
                                for x in range(out_w):
                                    expected_row[x] += integrate_vertical_cubic(
                                        energy[k // 2, x], widths[k // 2, x], d, 0.5 * pix_h, max_spot)
                    elif mode >= 2 or src_h <= 300:
                        pix_h = src_h_pix
                        for k in range(src_h):
                            d = (k + 0.5) - y_src
                            if abs(d) < max_spot + 0.5 * pix_h:
                                for x in range(out_w):
                                    expected_row[x] += integrate_vertical_cubic(
                                        energy[k, x], widths[k, x], d, 0.5 * pix_h, max_spot)
                    else:
                        pix_h = 0.5 * src_h_pix
                        p = 0.5 * y_src
                        fields = []
                        if mode == 1:
                            fields = [(0.25, 0, 0.5), (0.75, 1, 0.5)]
                        else:
                            phase = 1 if odd_first == 0 else 0
                            fields = [(0.25, 0, 1.0)] if (frame + phase) % 2 == 0 else [(0.75, 1, 1.0)]
                        for offset, row_offset, weight in fields:
                            for k in range((src_h - 1 - row_offset) // 2 + 1):
                                d = (k + offset) - p
                                if abs(d) < max_spot + 0.5 * pix_h:
                                    for x in range(out_w):
                                        expected_row[x] += weight * integrate_vertical_cubic(
                                            energy[2 * k + row_offset, x],
                                            widths[2 * k + row_offset, x],
                                            d, 0.5 * pix_h, max_spot)
                    np.testing.assert_allclose(actual[y_out], expected_row, rtol=3e-5, atol=3e-6)

    def test_vertical_integration_conserves_energy_at_noninteger_scale(self):
        # A single interior scanline must integrate to energy * max_spot_size regardless of scale.
        src_h = 10
        energy = np.zeros((src_h, 2, 3), dtype=np.float32)
        energy[5] = [0.2, 0.6, 1.0]
        widths = np.full((src_h, 2, 3), 0.36, dtype=np.float32)
        widths[5] = [0.36, 0.6, 0.9]
        for out_h in [7, 13, 31, 47]:
            output = np.empty((out_h, 2, 3), dtype=np.float32)
            spot_sim.taichi_spot_fast2_vertical_fragment(
                energy, widths, output, 0.9, 0.4, 0.0, 0.0, 3, 1, 0)
            total_energy = output.sum(axis=0) * (src_h / out_h)
            np.testing.assert_allclose(total_energy, energy[5] * 0.9, rtol=1e-5, atol=1e-6)

    def test_cosine_vertical_matches_analytical_in_all_line_modes(self):
        # Constant colors have a fixed spot width, so the separable cosine
        # approximation must agree with the full analytical cosine model.
        source = np.full((480, 8, 3), [0.02, 0.1, 0.5], dtype=np.float32)
        energy = np.empty((480, 19, 3), dtype=np.float32)
        widths = np.empty_like(energy)
        output = np.empty((37, 19, 3), dtype=np.float32)
        for mode in [0, 1, 2, 3]:
            spot_sim.taichi_spot_fast2_horizontal_fragment(
                source, energy, widths, 0.9, 0.4, 0.0, 0.0, 19 / 37,
                mode, cubic_horizontal=False)
            for frame in [0, 1]:
                with self.subTest(mode=mode, frame=frame):
                    spot_sim.taichi_spot_analytical_fragment(
                        source, output, 0.9, 0.4, 0.0, 0.0, mode, 1, frame)
                    expected = output.copy()
                    spot_sim.taichi_spot_fast2_vertical_fragment(
                        energy, widths, output, 0.9, 0.4, 0.0, 0.0,
                        mode, 1, frame, cubic_vertical=False)
                    self.assertTrue(np.isfinite(output).all())
                    np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-6)

    def test_fast_variant_methods_are_available_from_cli(self):
        for method in ['fast', 'fast-point', 'fast-cosine']:
            args = sim_taichi.build_parser().parse_args(['in.png', 'out.png', '--scanline-method', method])
            self.assertEqual(args.scanline_method, method)


if __name__ == '__main__':
    unittest.main()
