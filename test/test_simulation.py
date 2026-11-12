import unittest
import numpy as np
from config import SimulationConfig, Config, DEFAULT_CONFIG
import sim_taichi
import filtering
import spot_sim
import masking
import diffusion


class TestSimulationConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = SimulationConfig()
        self.assertTrue(cfg.enable_analog)
        self.assertEqual(cfg.analog_type, 'rgb')
        self.assertEqual(cfg.samples, 900)
        self.assertEqual(cfg.rgb_bandwidth, 4.0e6)
        self.assertEqual(cfg.luma_bandwidth, 4.0e6)
        self.assertEqual(cfg.chroma_bandwidth, 1.2e6)
        self.assertEqual(cfg.ntsc_filter_type, 'adaptive')
        self.assertEqual(cfg.ntsc_luma_trap_depth, 0.0)
        self.assertEqual(cfg.ntsc_system, 'standard')
        self.assertEqual(cfg.ntsc_frame_count, 0)
        self.assertTrue(cfg.enable_masking)
        self.assertTrue(cfg.enable_diffusion)
        self.assertEqual(cfg.scanline_method, DEFAULT_CONFIG.scanline_method)
        self.assertEqual(cfg.mask_type, 'bandlimit')
        self.assertEqual(cfg.diffusion_type, 'gaussian')
        self.assertEqual(cfg.output_resolution, (2160, 2880))
        self.assertEqual(cfg.max_spot_size, 0.9)
        self.assertEqual(cfg.min_spot_size, 0.4)
        self.assertEqual(cfg.interlacing_type, 'bob')
        self.assertTrue(cfg.odd_field_first)

    def test_custom_config(self):
        cfg = SimulationConfig(
            enable_analog=False,
            analog_type='svideo',
            rgb_bandwidth=3.5e6,
            luma_bandwidth=3.8e6,
            chroma_bandwidth=1.0e6,
            ntsc_filter_type='comb',
            ntsc_luma_trap_depth=16.0,
            ntsc_system='snes',
            ntsc_frame_count=1,
            scanline_method='fast',
            mask_type='subpixel',
            diffusion_type='box',
            output_resolution=(240, 320),
            interlacing_type='weave',
            odd_field_first=False,
        )
        self.assertFalse(cfg.enable_analog)
        self.assertEqual(cfg.analog_type, 'svideo')
        self.assertEqual(cfg.rgb_bandwidth, 3.5e6)
        self.assertEqual(cfg.luma_bandwidth, 3.8e6)
        self.assertEqual(cfg.chroma_bandwidth, 1.0e6)
        self.assertEqual(cfg.ntsc_filter_type, 'comb')
        self.assertEqual(cfg.ntsc_luma_trap_depth, 16.0)
        self.assertEqual(cfg.ntsc_system, 'snes')
        self.assertEqual(cfg.ntsc_frame_count, 1)
        self.assertEqual(cfg.scanline_method, 'fast')
        self.assertEqual(cfg.mask_type, 'subpixel')
        self.assertEqual(cfg.diffusion_type, 'box')
        self.assertEqual(cfg.output_resolution, (240, 320))
        self.assertEqual(cfg.interlacing_type, 'weave')
        self.assertFalse(cfg.odd_field_first)


class TestAnalogSimulation(unittest.TestCase):
    def setUp(self):
        # Create synthetic test image
        np.random.seed(42)
        self.img = np.random.uniform(0.1, 0.9, (24, 32, 3)).astype(np.float32)

    def test_linearize(self):
        out_direct = filtering.simulate_linearize(self.img, gamma=2.4)
        self.assertEqual(out_direct.shape, self.img.shape)
        self.assertTrue(np.all(np.isfinite(out_direct)))

        # Via simulate_analog dispatcher
        out_none = filtering.simulate_analog(self.img, analog_type='none')
        out_lin = filtering.simulate_analog(self.img, analog_type='linearize')
        np.testing.assert_allclose(out_none, out_direct)
        np.testing.assert_allclose(out_lin, out_direct)

    def test_rgb(self):
        samples = 64
        out = filtering.simulate_rgb(self.img, samples=samples, gamma=2.4, rgb_bandwidth=4.0e6)
        self.assertEqual(out.shape, (24, samples, 3))
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertTrue(np.all((out >= 0.0) & (out <= 1.0)))

        # Via dispatcher
        out_dispatch = filtering.simulate_analog(self.img, analog_type='rgb', samples=samples, rgb_bandwidth=4.0e6)
        np.testing.assert_allclose(out, out_dispatch)

    def test_svideo(self):
        samples = 64
        out = filtering.simulate_svideo(self.img, samples=samples, gamma=2.4, luma_bandwidth=4.0e6, chroma_bandwidth=1.2e6)
        self.assertEqual(out.shape, (24, samples, 3))
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertTrue(np.all((out >= 0.0) & (out <= 1.0)))

        # Via dispatcher with aliases
        out_svideo = filtering.simulate_analog(self.img, analog_type='svideo', samples=samples)
        out_dash = filtering.simulate_analog(self.img, analog_type='s-video', samples=samples)
        np.testing.assert_allclose(out, out_svideo)
        np.testing.assert_allclose(out, out_dash)

    def test_ntsc_filters(self):
        samples = 64
        for filter_type in ['notch', 'comb', 'adaptive']:
            out = filtering.simulate_ntsc(
                self.img,
                samples=samples,
                gamma=2.4,
                luma_bandwidth=4.0e6,
                chroma_bandwidth=1.2e6,
                ntsc_filter_type=filter_type,
                ntsc_system='standard',
                ntsc_frame_count=0,
            )
            self.assertEqual(out.shape, (24, samples, 3))
            self.assertTrue(np.all(np.isfinite(out)))

    def test_ntsc_systems(self):
        samples = 64
        systems = ['standard', 'nes', 'sms', 'tg16', 'genesis', 'snes', 'saturn', 'ps1', 'n64']
        for sys_name in systems:
            out = filtering.simulate_ntsc(
                self.img,
                samples=samples,
                gamma=2.4,
                ntsc_filter_type='adaptive',
                ntsc_system=sys_name,
                ntsc_frame_count=0,
            )
            self.assertEqual(out.shape, (24, samples, 3))
            self.assertTrue(np.all(np.isfinite(out)))

    def test_ntsc_frame_alternation(self):
        samples = 64
        # For NES, offset_per_frame is 1/3, so frame 0 and frame 1 produce different subcarrier phases
        out_f0 = filtering.simulate_ntsc(self.img, samples=samples, ntsc_system='nes', ntsc_frame_count=0)
        out_f1 = filtering.simulate_ntsc(self.img, samples=samples, ntsc_system='nes', ntsc_frame_count=1)
        self.assertFalse(np.allclose(out_f0, out_f1))

    def test_invalid_analog_type(self):
        with self.assertRaises(ValueError):
            filtering.simulate_analog(self.img, analog_type='unknown_analog')

    def test_ntsc_comb_cross_color_suppression(self):
        from common import RGB_TO_YUV

        # Sample above carrier Nyquist; compare full and low contrast bars.
        for contrast in [1.0, 0.1]:
            with self.subTest(contrast=contrast):
                pattern = np.zeros((16, 256, 3), dtype=np.float32)
                pattern[:, ::2] = contrast
                outputs = {
                    mode: filtering.simulate_ntsc(
                        pattern, samples=900, ntsc_filter_type=mode,
                    )
                    for mode in ['notch', 'comb', 'adaptive']
                }
                errors = {
                    mode: np.linalg.norm(
                        (output @ RGB_TO_YUV.T)[2:-2, 100:800, 1:], axis=-1,
                    ).mean()
                    for mode, output in outputs.items()
                }
                self.assertLess(errors['comb'], errors['notch'] * 0.5)
                self.assertLess(errors['adaptive'], errors['notch'] * 0.5)
                np.testing.assert_allclose(
                    outputs['adaptive'][2:-2, 100:800],
                    outputs['comb'][2:-2, 100:800], atol=1e-5, rtol=0,
                )

        # Verify one-sided directional combing on a 2-scanline-tall striped bar:
        # suppresses cross-color on both boundary scanlines without bleeding outside.
        bar_pattern = np.full((16, 256, 3), 0.25, dtype=np.float32)
        bar_pattern[7:9, ::2] = 1.0
        bar_outputs = {
            mode: filtering.simulate_ntsc(
                bar_pattern, samples=900, ntsc_filter_type=mode,
            )
            for mode in ['notch', 'comb', 'adaptive']
        }
        bar_errors = {
            mode: np.linalg.norm(
                (output @ RGB_TO_YUV.T)[7:9, 100:800, 1:], axis=-1,
            ).mean()
            for mode, output in bar_outputs.items()
        }
        bleed_comb = np.linalg.norm(
            (bar_outputs['comb'] @ RGB_TO_YUV.T)[[6, 9], 100:800, 1:], axis=-1,
        ).mean()
        bleed_adaptive = np.linalg.norm(
            (bar_outputs['adaptive'] @ RGB_TO_YUV.T)[[6, 9], 100:800, 1:], axis=-1,
        ).mean()
        self.assertLess(bar_errors['adaptive'], bar_errors['comb'] * 0.1)
        self.assertGreater(bleed_comb, 0.005)
        self.assertLess(bleed_adaptive, 1e-5)

    def test_ntsc_comb_hanging_dots_reduction(self):
        # Sharp horizontal color boundary (Red top half, Blue bottom half)
        h, w = 16, 32
        edge_pattern = np.zeros((h, w, 3), dtype=np.float32)
        edge_pattern[:8, :] = [1.0, 0.0, 0.0]  # Red
        edge_pattern[8:, :] = [0.0, 0.0, 1.0]  # Blue

        samples = 128
        out_comb = filtering.simulate_ntsc(edge_pattern, samples=samples, ntsc_filter_type='comb')
        out_adapt = filtering.simulate_ntsc(edge_pattern, samples=samples, ntsc_filter_type='adaptive')

        from common import RGB_TO_YUV
        yuv_comb = np.dot(out_comb, RGB_TO_YUV.T)
        yuv_adapt = np.dot(out_adapt, RGB_TO_YUV.T)

        # Measure ripple along the boundary scanlines (y=7 and y=8)
        ripple_comb = np.mean(np.abs(np.diff(yuv_comb[7:9, :, 0], axis=1)))
        ripple_adapt = np.mean(np.abs(np.diff(yuv_adapt[7:9, :, 0], axis=1)))

        # Adaptive directional comb must significantly reduce hanging dots compared to 3-line comb
        self.assertLess(ripple_adapt, ripple_comb)

    def test_ntsc_comb_isolated_line_fallback(self):
        # 1-pixel high horizontal line (Green line on dark background)
        h, w = 16, 32
        line_pattern = np.zeros((h, w, 3), dtype=np.float32)
        line_pattern[8, :] = [0.0, 1.0, 0.0]  # Single green scanline

        samples = 128
        out_notch = filtering.simulate_ntsc(line_pattern, samples=samples, ntsc_filter_type='notch')
        out_comb = filtering.simulate_ntsc(line_pattern, samples=samples, ntsc_filter_type='comb')
        out_adapt = filtering.simulate_ntsc(line_pattern, samples=samples, ntsc_filter_type='adaptive')

        from common import RGB_TO_YUV
        yuv_notch = np.dot(out_notch, RGB_TO_YUV.T)
        yuv_comb = np.dot(out_comb, RGB_TO_YUV.T)
        yuv_adapt = np.dot(out_adapt, RGB_TO_YUV.T)

        ripple_notch = np.mean(np.abs(np.diff(yuv_notch[8, :, 0])))
        ripple_comb = np.mean(np.abs(np.diff(yuv_comb[8, :, 0])))
        ripple_adapt = np.mean(np.abs(np.diff(yuv_adapt[8, :, 0])))

        # On an isolated line, adaptive comb must fall back to notch
        self.assertAlmostEqual(float(ripple_adapt), float(ripple_notch), places=3)
        self.assertLess(ripple_adapt, ripple_comb * 0.5)

        # On adjacent line (y=7), comb induces massive hanging dots; adapt must eliminate them
        ripple_comb_adj = np.mean(np.abs(np.diff(yuv_comb[7, :, 0])))
        ripple_adapt_adj = np.mean(np.abs(np.diff(yuv_adapt[7, :, 0])))
        self.assertGreater(ripple_comb_adj, 0.2)
        self.assertLess(ripple_adapt_adj, 0.01)

    def test_ntsc_comb_complementary_line_fallback(self):
        # Complementary colors are anti-phase too, but must retain their chroma.
        delta = np.array([0.35, -0.413 * 0.35 / 0.587, 0.35], dtype=np.float32)
        for system in ['standard', 'snes', 'genesis']:
            for line_color, background in [
                ([1.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
                ([0.0, 1.0, 0.0], [1.0, 0.0, 1.0]),
                (0.5 + delta, 0.5 - delta),  # Equal-luma complementary colors
            ]:
                with self.subTest(system=system, line_color=line_color):
                    pattern = np.empty((16, 256, 3), dtype=np.float32)
                    pattern[:] = background
                    pattern[8] = line_color
                    outputs = {
                        mode: filtering.simulate_ntsc(
                            pattern, samples=900, ntsc_system=system,
                            ntsc_filter_type=mode,
                        )
                        for mode in ['notch', 'adaptive']
                    }
                    # Exclude the horizontal filter's boundary transients.
                    line = outputs['adaptive'][8, 100:800]
                    np.testing.assert_allclose(
                        line, outputs['notch'][8, 100:800], atol=1e-5, rtol=0,
                    )
                    np.testing.assert_allclose(
                        np.power(line, 1.0 / 2.4).mean(axis=0),
                        line_color, atol=0.02, rtol=0,
                    )

    def test_ntsc_comb_alternating_equal_luma_fallback(self):
        # Alternating chroma and carrier phase can produce repeated composite
        # waveforms, so spatial signature agreement alone cannot justify combing.
        delta = np.array([0.35, -0.413 * 0.35 / 0.587, 0.35], dtype=np.float32)
        pattern = np.empty((16, 256, 3), dtype=np.float32)
        pattern[::2] = 0.5 + delta
        pattern[1::2] = 0.5 - delta
        for system in ['standard', 'snes', 'genesis']:
            with self.subTest(system=system):
                notch = filtering.simulate_ntsc(
                    pattern, samples=900, ntsc_system=system, ntsc_filter_type='notch',
                )
                adaptive = filtering.simulate_ntsc(
                    pattern, samples=900, ntsc_system=system, ntsc_filter_type='adaptive',
                )
                np.testing.assert_allclose(
                    adaptive[2:-2, 100:800], notch[2:-2, 100:800], atol=1e-5, rtol=0,
                )

    def test_ntsc_boundary_safety(self):
        # Image with small height to test boundary conditions
        small_img = np.full((3, 16, 3), 0.5, dtype=np.float32)
        out = filtering.simulate_ntsc(small_img, samples=32, ntsc_filter_type='adaptive')
        self.assertEqual(out.shape, (3, 32, 3))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_ntsc_carrier_exact_interpolator_invariants(self):
        # Verify algebraic invariants of the 4-tap carrier-exact comb interpolator
        import taichi as ti
        import taichi.math as tm

        @ti.kernel
        def eval_kernel(u: float, wc: float) -> tm.vec4:
            return filtering.carrier_exact_4tap(u, wc)

        # Nodes relative to midpoint u in [-0.5, 0.5]
        nodes = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
        for wc in [1.1914, 1.2709, 1.3069]:
            for u in np.linspace(-0.5, 0.5, 11):
                w_vec = eval_kernel(float(u), float(wc))
                w = np.array([w_vec.x, w_vec.y, w_vec.z, w_vec.w], dtype=np.float64)

                # 1. Partition of unity (exact DC preservation: sum w_k == 1)
                self.assertAlmostEqual(float(np.sum(w)), 1.0, places=6)

                # 2. Linear moment preservation (sum x_k w_k == u)
                self.assertAlmostEqual(float(np.sum(nodes * w)), float(u), places=6)

                # 3. Exact response at carrier frequency wc
                expected = np.exp(1j * u * wc)
                actual = np.sum(w * np.exp(1j * nodes * wc))
                self.assertAlmostEqual(float(np.real(actual)), float(np.real(expected)), places=6)
                self.assertAlmostEqual(float(np.imag(actual)), float(np.imag(expected)), places=6)

            # 4. Kronecker delta at exact grid points
            w_left = np.array(eval_kernel(-0.5, float(wc)), dtype=np.float64)
            np.testing.assert_allclose(w_left, [0.0, 1.0, 0.0, 0.0], atol=1e-6)

            w_right = np.array(eval_kernel(0.5, float(wc)), dtype=np.float64)
            np.testing.assert_allclose(w_right, [0.0, 0.0, 1.0, 0.0], atol=1e-6)

    def test_ntsc_chroma_lpf_carrier_nulls(self):
        # Carrier-locked Hann window of half-width W = 1.5 * samples / periods_per_active_line
        # places spectral nulls near fc (k=3) and 2*fc (k=6).
        for periods in [170.667, 182.044, 187.2]:
            samples = 900.0
            W = 1.5 * samples / periods
            i_bound = int(np.floor(W))
            n = np.arange(-i_bound, i_bound + 1)
            hann = 0.5 + 0.5 * np.cos(np.pi * n / W)
            hann /= np.sum(hann)

            f_c = periods / samples
            h_fc = np.abs(np.sum(hann * np.exp(-2j * np.pi * f_c * n)))
            h_2fc = np.abs(np.sum(hann * np.exp(-2j * np.pi * 2.0 * f_c * n)))

            self.assertLess(h_fc, 0.0003)
            self.assertLess(h_2fc, 0.0007)

    def test_ntsc_interlaced_carrier_phase_patterns(self):
        # Verify 4-field carrier phase patterns at the horizontal start of the first
        # visible line of the active field (line 0 for odd field, line 1 for even field).
        expected_patterns = {
            'snes': [0.0, 2.0 / 3.0, 0.0, 2.0 / 3.0],
            'genesis': [0.0, 0.0, 0.0, 0.0],
            'standard': [0.0, 0.5, 0.5, 0.0],
            'saturn': [0.0, 0.5, 0.5, 0.0],
            'ps1': [0.0, 0.5, 0.5, 0.0],
            'n64': [0.0, 0.5, 0.5, 0.0],
        }

        for system, expected_phases in expected_patterns.items():
            ci = filtering.get_chroma_info(system, 320, 0)
            actual_phases = []
            for frame in range(4):
                active_line = 1 if frame % 2 == 1 else 0
                phase_rad = filtering.eval_chroma_carrier(
                    0.0, active_line, frame,
                    ci.periods_per_active_line, ci.periods_per_line, ci.offset_per_frame,
                    interlaced=True, odd_field_first=True,
                )
                phase_norm = (phase_rad / (2.0 * np.pi)) % 1.0
                actual_phases.append(phase_norm)

            with self.subTest(system=system):
                np.testing.assert_allclose(actual_phases, expected_phases, atol=1e-5)

    def test_ntsc_interlaced_stale_field_phase_invariance(self):
        # In an interlaced weave buffer, the inactive field is stale from (frame - 1).
        # Its carrier phase must not change between its active frame and its stale frame.
        for system in ['standard', 'snes', 'genesis']:
            ci = filtering.get_chroma_info(system, 320, 0)
            # Frame 0: odd field (line 0) active. Frame 1: odd field (line 0) stale.
            phase_f0_line0 = filtering.eval_chroma_carrier(
                0.0, 0, 0, ci.periods_per_active_line, ci.periods_per_line, ci.offset_per_frame,
                interlaced=True, odd_field_first=True,
            )
            phase_f1_line0 = filtering.eval_chroma_carrier(
                0.0, 0, 1, ci.periods_per_active_line, ci.periods_per_line, ci.offset_per_frame,
                interlaced=True, odd_field_first=True,
            )
            self.assertAlmostEqual(phase_f0_line0, phase_f1_line0, places=5)

            # Frame 1: even field (line 1) active. Frame 2: even field (line 1) stale.
            phase_f1_line1 = filtering.eval_chroma_carrier(
                0.0, 1, 1, ci.periods_per_active_line, ci.periods_per_line, ci.offset_per_frame,
                interlaced=True, odd_field_first=True,
            )
            phase_f2_line1 = filtering.eval_chroma_carrier(
                0.0, 1, 2, ci.periods_per_active_line, ci.periods_per_line, ci.offset_per_frame,
                interlaced=True, odd_field_first=True,
            )
            self.assertAlmostEqual(phase_f1_line1, phase_f2_line1, places=5)

    def test_ntsc_interlaced_simulation_runs(self):
        # 480i test image (height > 300 triggers interlacing)
        img_interlaced = np.random.uniform(0.1, 0.9, (480, 32, 3)).astype(np.float32)
        samples = 64
        for itype in ['bob', 'weave', 'off']:
            out = filtering.simulate_ntsc(
                img_interlaced,
                samples=samples,
                interlacing_type=itype,
                odd_field_first=True,
                ntsc_frame_count=0,
            )
            self.assertEqual(out.shape, (480, samples, 3))
            self.assertTrue(np.all(np.isfinite(out)))

    def test_chroma_info_functions(self):
        # Standard
        std = filtering.chroma_info_standard()
        self.assertAlmostEqual(std.periods_per_active_line, filtering.CHROMA_SUBCARRIER * 52.5e-6)
        self.assertAlmostEqual(std.periods_per_line, 227.5)
        self.assertAlmostEqual(std.offset_per_frame, 0.5)

        # NES
        nes = filtering.chroma_info_nes(256.0)
        self.assertAlmostEqual(nes.periods_per_active_line, (4.0 / 6.0) * 256.0)
        self.assertAlmostEqual(nes.periods_per_line, 682.0 / 3.0)
        self.assertAlmostEqual(nes.offset_per_frame, 1.0 / 3.0)

        # SMS
        sms = filtering.chroma_info_sms(256.0)
        self.assertAlmostEqual(sms.periods_per_active_line, (10.0 / 15.0) * 256.0)
        self.assertAlmostEqual(sms.periods_per_line, 228.0)
        self.assertAlmostEqual(sms.offset_per_frame, 0.0)

        # PCE / TG16
        pce_256 = filtering.chroma_info_pce(256.0)
        self.assertAlmostEqual(pce_256.periods_per_active_line, (4.0 / 6.0) * 256.0)
        self.assertAlmostEqual(pce_256.periods_per_line, 227.5)
        self.assertAlmostEqual(pce_256.offset_per_frame, 0.5)

        pce_320 = filtering.chroma_info_pce(320.0)
        self.assertAlmostEqual(pce_320.periods_per_active_line, (3.0 / 6.0) * 320.0)
        self.assertAlmostEqual(pce_320.periods_per_line, 227.5)
        self.assertAlmostEqual(pce_320.offset_per_frame, 0.5)

        pce_512 = filtering.chroma_info_pce(512.0)
        self.assertAlmostEqual(pce_512.periods_per_active_line, (2.0 / 6.0) * 512.0)
        self.assertAlmostEqual(pce_512.periods_per_line, 227.5)
        self.assertAlmostEqual(pce_512.offset_per_frame, 0.5)

        # Genesis
        gen_h32 = filtering.chroma_info_genesis(256.0)
        self.assertAlmostEqual(gen_h32.periods_per_active_line, (10.0 / 15.0) * 256.0)
        self.assertAlmostEqual(gen_h32.periods_per_line, 228.0)
        self.assertAlmostEqual(gen_h32.offset_per_frame, 0.0)

        gen_h40 = filtering.chroma_info_genesis(320.0)
        self.assertAlmostEqual(gen_h40.periods_per_active_line, (8.0 / 15.0) * 320.0)
        self.assertAlmostEqual(gen_h40.periods_per_line, 228.0)
        self.assertAlmostEqual(gen_h40.offset_per_frame, 0.0)

        # SNES
        snes_256 = filtering.chroma_info_snes(256.0)
        self.assertAlmostEqual(snes_256.periods_per_active_line, (4.0 / 6.0) * 256.0)
        self.assertAlmostEqual(snes_256.periods_per_line, 682.0 / 3.0)
        self.assertAlmostEqual(snes_256.offset_per_frame, 1.0 / 3.0)

        snes_512 = filtering.chroma_info_snes(512.0)
        self.assertAlmostEqual(snes_512.periods_per_active_line, (2.0 / 6.0) * 512.0)
        self.assertAlmostEqual(snes_512.periods_per_line, 682.0 / 3.0)
        self.assertAlmostEqual(snes_512.offset_per_frame, 1.0 / 3.0)

        # Saturn
        sat_320 = filtering.chroma_info_saturn(320.0)
        self.assertAlmostEqual(sat_320.periods_per_active_line, (65.0 / (2.0 * 61.0)) * 320.0)
        self.assertAlmostEqual(sat_320.periods_per_line, 227.5)
        self.assertAlmostEqual(sat_320.offset_per_frame, 0.5)

        sat_352 = filtering.chroma_info_saturn(352.0)
        self.assertAlmostEqual(sat_352.periods_per_active_line, 352.0 / 2.0)
        self.assertAlmostEqual(sat_352.periods_per_line, 227.5)
        self.assertAlmostEqual(sat_352.offset_per_frame, 0.5)

        sat_640 = filtering.chroma_info_saturn(640.0)
        self.assertAlmostEqual(sat_640.periods_per_active_line, (65.0 / (4.0 * 61.0)) * 640.0)
        self.assertAlmostEqual(sat_640.periods_per_line, 227.5)
        self.assertAlmostEqual(sat_640.offset_per_frame, 0.5)

        sat_704 = filtering.chroma_info_saturn(704.0)
        self.assertAlmostEqual(sat_704.periods_per_active_line, 704.0 / 4.0)
        self.assertAlmostEqual(sat_704.periods_per_line, 227.5)
        self.assertAlmostEqual(sat_704.offset_per_frame, 0.5)

        # PSX
        psx_256 = filtering.chroma_info_psx(256.0)
        self.assertAlmostEqual(psx_256.periods_per_active_line, 256.0 * 10.0 / 15.0)
        self.assertAlmostEqual(psx_256.periods_per_line, 227.5)
        self.assertAlmostEqual(psx_256.offset_per_frame, 0.5)

        psx_320 = filtering.chroma_info_psx(320.0)
        self.assertAlmostEqual(psx_320.periods_per_active_line, 320.0 * 8.0 / 15.0)
        self.assertAlmostEqual(psx_320.periods_per_line, 227.5)
        self.assertAlmostEqual(psx_320.offset_per_frame, 0.5)

        psx_368 = filtering.chroma_info_psx(368.0)
        self.assertAlmostEqual(psx_368.periods_per_active_line, 368.0 * 7.0 / 15.0)
        self.assertAlmostEqual(psx_368.periods_per_line, 227.5)
        self.assertAlmostEqual(psx_368.offset_per_frame, 0.5)

        psx_512 = filtering.chroma_info_psx(512.0)
        self.assertAlmostEqual(psx_512.periods_per_active_line, 512.0 * 5.0 / 15.0)
        self.assertAlmostEqual(psx_512.periods_per_line, 227.5)
        self.assertAlmostEqual(psx_512.offset_per_frame, 0.5)

        psx_640 = filtering.chroma_info_psx(640.0)
        self.assertAlmostEqual(psx_640.periods_per_active_line, 640.0 * 4.0 / 15.0)
        self.assertAlmostEqual(psx_640.periods_per_line, 227.5)
        self.assertAlmostEqual(psx_640.offset_per_frame, 0.5)

        # N64
        n64 = filtering.chroma_info_n64()
        self.assertAlmostEqual(n64.periods_per_active_line, (640.0 * 227.5) / 773.5)
        self.assertAlmostEqual(n64.periods_per_line, 227.5)
        self.assertAlmostEqual(n64.offset_per_frame, 0.5)

        # get_chroma_info dispatch and aliases
        self.assertEqual(filtering.get_chroma_info('standard', 320.0), std)
        self.assertEqual(filtering.get_chroma_info('unknown', 320.0), std)
        self.assertEqual(filtering.get_chroma_info('nes', 256.0), nes)
        self.assertEqual(filtering.get_chroma_info('sms', 256.0), sms)
        self.assertEqual(filtering.get_chroma_info('pce', 256.0), pce_256)
        self.assertEqual(filtering.get_chroma_info('tg16', 256.0), pce_256)
        self.assertEqual(filtering.get_chroma_info('genesis', 320.0), gen_h40)
        self.assertEqual(filtering.get_chroma_info('megadrive', 320.0), gen_h40)
        self.assertEqual(filtering.get_chroma_info('snes', 256.0), snes_256)
        self.assertEqual(filtering.get_chroma_info('saturn', 352.0), sat_352)
        self.assertEqual(filtering.get_chroma_info('ps1', 320.0), psx_320)
        self.assertEqual(filtering.get_chroma_info('psx', 320.0), psx_320)
        self.assertEqual(filtering.get_chroma_info('n64', 640.0), n64)

    def test_psx_active_line_duration_across_modes(self):
        # These native modes all display 2560 GPU clocks per active line.
        for width in [256, 320, 512, 640]:
            with self.subTest(width=width):
                ci = filtering.get_chroma_info('ps1', width)
                self.assertAlmostEqual(ci.periods_per_active_line, 2560.0 / 15.0)

        # The 368-pixel mode uses seven GPU clocks per pixel.
        ci = filtering.get_chroma_info('ps1', 368)
        self.assertAlmostEqual(ci.periods_per_active_line, 2576.0 / 15.0)

        # Padded captures span 2800 clocks, regardless of pixel-clock mode.
        for width in [280, 350, 400, 560, 700]:
            with self.subTest(capture_width=width):
                ci = filtering.get_chroma_info('ps1', width)
                self.assertAlmostEqual(ci.periods_per_active_line, 2800.0 / 15.0)
                self.assertLess(ci.periods_per_active_line, ci.periods_per_line)
                self.assertAlmostEqual(ci.periods_per_line, 227.5)
                self.assertAlmostEqual(ci.offset_per_frame, 0.5)

    def test_psx_capture_mode_boundaries(self):
        # A width just beyond a padded mode selects the next pixel clock.
        for width, clocks_per_pixel in [(281, 8), (351, 7), (401, 5), (561, 4)]:
            with self.subTest(width=width):
                ci = filtering.get_chroma_info('psx', width)
                self.assertAlmostEqual(
                    ci.periods_per_active_line, width * clocks_per_pixel / 15.0,
                )


class TestScanlineMethods(unittest.TestCase):
    def setUp(self):
        self.img = np.full((24, 32, 3), 0.5, dtype=np.float32)
        self.out_dim = (120, 160, 3)

    def test_sampled_method(self):
        out = spot_sim.simulate_spot(self.img, self.out_dim, 'sampled', 0.9, 0.4, 0.0, 0.0)
        self.assertEqual(out.shape, self.out_dim)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertGreater(float(np.mean(out)), 0.0)

    def test_fast_method(self):
        out = spot_sim.simulate_spot(self.img, self.out_dim, 'fast', 0.9, 0.4, 0.0, 0.0)
        self.assertEqual(out.shape, self.out_dim)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertGreater(float(np.mean(out)), 0.0)

    def test_analytical_method(self):
        out = spot_sim.simulate_spot(self.img, self.out_dim, 'analytical', 0.9, 0.4, 0.0, 0.0)
        self.assertEqual(out.shape, self.out_dim)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertGreater(float(np.mean(out)), 0.0)

    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            spot_sim.simulate_spot(self.img, self.out_dim, 'unknown_method', 0.9, 0.4, 0.0, 0.0)

    def test_overscan_spot_simulation(self):
        img = np.zeros((24, 32, 3), dtype=np.float32)
        img[:, 10:22, :] = 1.0
        out_dim = (120, 160, 3)
        for method in ['fast', 'sampled', 'analytical']:
            with self.subTest(method=method):
                out_overscan = spot_sim.simulate_spot(
                    img, out_dim, method, 0.9, 0.4, 0.05, 0.05,
                )
                self.assertEqual(out_overscan.shape, out_dim)
                self.assertTrue(np.all(np.isfinite(out_overscan)))
                self.assertGreater(float(np.mean(out_overscan)), 0.0)

                out_no_overscan = spot_sim.simulate_spot(
                    img, out_dim, method, 0.9, 0.4, 0.0, 0.0,
                )
                self.assertGreater(float(np.max(np.abs(out_overscan - out_no_overscan))), 1e-3)

    def test_interlaced_spot_simulation(self):
        img_480 = np.full((480, 32, 3), 0.5, dtype=np.float32)
        out_dim = (480, 64, 3)
        for method in ['sampled', 'fast', 'analytical']:
            for itype in ['bob', 'weave', 'off']:
                for frame in [0, 1]:
                    out = spot_sim.simulate_spot(
                        img_480, out_dim, method, 0.9, 0.4, 0.0, 0.0,
                        interlacing_type=itype, odd_field_first=True, frame_count=frame,
                    )
                    self.assertEqual(out.shape, out_dim)
                    self.assertTrue(np.all(np.isfinite(out)))
                    self.assertGreater(float(np.mean(out)), 0.0)

    def test_vga_line_doubling_spot_simulation(self):
        img_200 = np.full((200, 32, 3), 0.5, dtype=np.float32)
        out_dim = (400, 64, 3)
        for method in ['sampled', 'fast', 'analytical']:
            out = spot_sim.simulate_spot(
                img_200, out_dim, method, 0.9, 0.4, 0.0, 0.0,
                interlacing_type='vga',
            )
            self.assertEqual(out.shape, out_dim)
            self.assertTrue(np.all(np.isfinite(out)))
            self.assertGreater(float(np.mean(out)), 0.0)

    def test_vga_progressive_threshold_and_field_invariance(self):
        # Different neighboring rows expose accidental bob field selection.
        for height in [349, 350, 400, 480]:
            img = np.full((height, 8, 3), 0.1, dtype=np.float32)
            img[::2] = 0.8
            out_dim = (height * 2, 16, 3)
            # Reuse fields across modes/phases to avoid accumulating Taichi
            # field trees for every render in the full suite.
            field_in = spot_sim.ti.Vector.field(n=3, dtype=float, shape=img.shape[:2])
            field_in.from_numpy(img)
            field_out = spot_sim.ti.Vector.field(n=3, dtype=float, shape=out_dim[:2])
            energy = spot_sim.ti.Vector.field(n=3, dtype=float, shape=(height, 16))
            width = spot_sim.ti.Vector.field(n=3, dtype=float, shape=(height, 16))

            def render(method, mode, frame=0, odd_first=1):
                if method == 'fast':
                    spot_sim.taichi_spot_fast2_horizontal_fragment(
                        field_in, energy, width, 0.9, 0.4, 0.0, 0.0,
                        out_dim[1] / out_dim[0], mode,
                    )
                    spot_sim.taichi_spot_fast2_vertical_fragment(
                        energy, width, field_out, 0.9, 0.4, 0.0, 0.0,
                        mode, odd_first, frame,
                    )
                else:
                    fragment = (spot_sim.taichi_spot_fragment if method == 'sampled'
                                else spot_sim.taichi_spot_analytical_fragment)
                    fragment(field_in, field_out, 0.9, 0.4, 0.0, 0.0,
                             mode, odd_first, frame)
                return field_out.to_numpy()

            for method in ['sampled', 'analytical', 'fast']:
                with self.subTest(height=height, method=method):
                    progressive = render(method, 3)
                    vga = render(method, 2)
                    self.assertEqual(vga.shape, out_dim)
                    self.assertTrue(np.all(np.isfinite(vga)))
                    self.assertGreater(float(np.mean(vga)), 0.0)
                    if height >= 350:
                        np.testing.assert_allclose(vga, progressive, rtol=1e-5, atol=1e-6)
                    else:
                        self.assertGreater(float(np.max(np.abs(vga - progressive))), 1e-3)

                    for frame, odd_first in [(1, True), (0, False)]:
                        other_phase = render(method, 2, frame, int(odd_first))
                        np.testing.assert_allclose(other_phase, vga, rtol=1e-5, atol=1e-6)


class TestMaskTypes(unittest.TestCase):
    def setUp(self):
        self.img = np.full((120, 160, 3), 0.5, dtype=np.float32)

    def test_subpixel_mask(self):
        for pattern in [0, 1, 2, 3]:
            out = masking.apply_mask(self.img, 'subpixel', pattern, 'slot', 550.0, 1.0)
            self.assertEqual(out.shape, self.img.shape)
            self.assertTrue(np.all(np.isfinite(out)))

    def test_bandlimit_mask(self):
        out = masking.apply_mask(self.img, 'bandlimit', 1, 'slot', 550.0, 1.0)
        self.assertEqual(out.shape, self.img.shape)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_coverage_mask(self):
        out = masking.apply_mask(self.img, 'coverage', 1, 'slot', 550.0, 1.0)
        self.assertEqual(out.shape, self.img.shape)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_additive_mask(self):
        out = masking.apply_mask(self.img, 'additive', 1, 'slot', 550.0, 1.0)
        self.assertEqual(out.shape, self.img.shape)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_tiled_mask(self):
        for tiled_type in ['slot', 'aperture']:
            out = masking.apply_mask(self.img, 'tiled', 1, tiled_type, 550.0, 1.0)
            self.assertEqual(out.shape, self.img.shape)
            self.assertTrue(np.all(np.isfinite(out)))

    def test_none_mask(self):
        out = masking.apply_mask(self.img, 'none', 1, 'slot', 550.0, 1.0)
        np.testing.assert_array_equal(out, self.img)

    def test_blend_mask_brightness_mitigation(self):
        mask = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=np.float32)
        coverage = 3.0

        for val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            pixel_val = np.full_like(mask, val)
            blended = masking.blend_mask(mask, pixel_val, coverage, mitigation=1.0)
            self.assertAlmostEqual(float(np.mean(blended)), val, places=5)

        pixel_val = np.full_like(mask, 0.6)
        blended_nomit = masking.blend_mask(mask, pixel_val, coverage, mitigation=0.0)
        expected_nomit = pixel_val * mask
        np.testing.assert_allclose(blended_nomit, expected_nomit, atol=1e-5)

        blended_cov1 = masking.blend_mask(mask, pixel_val, 1.0, mitigation=1.0)
        np.testing.assert_allclose(blended_cov1, pixel_val, atol=1e-5)

    def test_mask_amount(self):
        out_mit_full = masking.apply_mask(self.img, 'bandlimit', 1, 'slot', 550.0, 1.0)
        out_mit_zero = masking.apply_mask(self.img, 'bandlimit', 1, 'slot', 550.0, 0.0)
        out_mit_half = masking.apply_mask(self.img, 'bandlimit', 1, 'slot', 550.0, 0.5)

        self.assertTrue(np.all(np.isfinite(out_mit_full)))
        self.assertTrue(np.all(np.isfinite(out_mit_zero)))
        self.assertTrue(np.all(np.isfinite(out_mit_half)))

        self.assertGreater(float(np.mean(out_mit_full)), float(np.mean(out_mit_zero)))
        self.assertGreater(float(np.mean(out_mit_half)), float(np.mean(out_mit_zero)))
        self.assertGreater(float(np.mean(out_mit_full)), float(np.mean(out_mit_half)))


class TestDiffusionTypes(unittest.TestCase):
    def setUp(self):
        self.img = np.zeros((100, 100, 3), dtype=np.float32)
        self.img[40:60, 40:60] = 1.0

    def test_gaussian_diffusion(self):
        out = diffusion.apply_diffusion(self.img, 'gaussian', 0.05, 0.04)
        self.assertEqual(out.shape, self.img.shape)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertAlmostEqual(float(np.mean(out)), float(np.mean(self.img)), places=4)

    def test_box_diffusion(self):
        out = diffusion.apply_diffusion(self.img, 'box', 0.05, 0.04)
        self.assertEqual(out.shape, self.img.shape)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertAlmostEqual(float(np.mean(out)), float(np.mean(self.img)), places=4)

    def test_none_diffusion(self):
        out = diffusion.apply_diffusion(self.img, 'none', 0.05, 0.04)
        np.testing.assert_array_equal(out, self.img)


class TestFullSimulation(unittest.TestCase):
    def setUp(self):
        self.img = np.full((24, 32, 3), 0.5, dtype=np.float32)

    def test_default_simulation_runs(self):
        cfg = SimulationConfig(
            output_resolution=(120, 160),
            samples=320,
        )
        out = sim_taichi.simulate(self.img, config=cfg)
        self.assertEqual(out.shape, (120, 160, 3))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_all_stages_disabled(self):
        cfg = SimulationConfig(
            enable_analog=False,
            enable_masking=False,
            enable_diffusion=False,
            output_resolution=(120, 160),
            scanline_method='fast'
        )
        out = sim_taichi.simulate(self.img, config=cfg)
        self.assertEqual(out.shape, (120, 160, 3))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_all_analog_modes(self):
        for atype in ['linearize', 'rgb', 'svideo', 'ntsc']:
            cfg = SimulationConfig(
                enable_analog=True,
                analog_type=atype,
                enable_masking=False,
                enable_diffusion=False,
                output_resolution=(120, 160),
                samples=320,
                scanline_method='fast'
            )
            out = sim_taichi.simulate(self.img, config=cfg)
            self.assertEqual(out.shape, (120, 160, 3))
            self.assertTrue(np.all(np.isfinite(out)))


class TestCLIParser(unittest.TestCase):
    def test_parser_defaults(self):
        parser = sim_taichi.build_parser()
        args = parser.parse_args(['in.png', 'out.png'])
        self.assertEqual(args.input, 'in.png')
        self.assertEqual(args.output, 'out.png')
        self.assertEqual(args.scanline_method, DEFAULT_CONFIG.scanline_method)
        self.assertTrue(args.enable_analog)
        self.assertEqual(args.analog_type, 'rgb')
        self.assertEqual(args.rgb_bandwidth, 4.0e6)
        self.assertEqual(args.luma_bandwidth, 4.0e6)
        self.assertEqual(args.chroma_bandwidth, 1.2e6)
        self.assertEqual(args.ntsc_filter_type, 'adaptive')
        self.assertEqual(args.ntsc_luma_trap_depth, 0.0)
        self.assertEqual(args.ntsc_system, 'standard')
        self.assertEqual(args.ntsc_frame_count, 0)
        self.assertTrue(args.masking)
        self.assertTrue(args.diffusion)
        self.assertEqual(args.mask_type, 'bandlimit')
        self.assertEqual(args.diffusion_type, 'gaussian')
        self.assertEqual(args.interlacing_type, 'bob')
        self.assertTrue(args.odd_field_first)

    def test_parser_overrides(self):
        parser = sim_taichi.build_parser()
        args = parser.parse_args([
            'in.png', 'out.png',
            '--scanline-method', 'fast',
            '--no-analog',
            '--analog-type', 'svideo',
            '--rgb-bandwidth', '3.5e6',
            '--luma-bandwidth', '3.8e6',
            '--chroma-bandwidth', '1.0e6',
            '--ntsc-filter-type', 'comb',
            '--ntsc-luma-trap-depth', '32',
            '--ntsc-system', 'snes',
            '--ntsc-frame-count', '1',
            '--interlacing-type', 'weave',
            '--no-odd-field-first',
            '--no-masking',
            '--no-diffusion',
            '--output-resolution', '720', '960'
        ])
        self.assertEqual(args.scanline_method, 'fast')
        self.assertFalse(args.enable_analog)
        self.assertEqual(args.analog_type, 'svideo')
        self.assertEqual(args.rgb_bandwidth, 3.5e6)
        self.assertEqual(args.luma_bandwidth, 3.8e6)
        self.assertEqual(args.chroma_bandwidth, 1.0e6)
        self.assertEqual(args.ntsc_filter_type, 'comb')
        self.assertEqual(args.ntsc_luma_trap_depth, 32.0)
        self.assertEqual(args.ntsc_system, 'snes')
        self.assertEqual(args.ntsc_frame_count, 1)
        self.assertEqual(args.interlacing_type, 'weave')
        self.assertFalse(args.odd_field_first)
        self.assertFalse(args.masking)
        self.assertFalse(args.diffusion)
        self.assertEqual(args.output_resolution, [720, 960])


if __name__ == '__main__':
    unittest.main()
