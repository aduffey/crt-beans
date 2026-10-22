"""Benchmark protocol checks; fake frontends do not claim to validate GPU rendering."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import benchmark_shader as bench


VALID_LOG = '''[INFO] [Vulkan] Found vulkan context: "vk_wayland".
[INFO] [Vulkan] Using GPU: "AMD Radeon Graphics".
[INFO] [Vulkan] Using resolution 2880x2160.
[DEBUG] [Vulkan] Creating swapchain with present mode: MAILBOX.
'''


class TestBenchmarkValidation(unittest.TestCase):
    def test_rejects_invalid_rendering_even_after_a_successful_initialization(self):
        for log in [
                VALID_LOG + '[ERROR] [Vulkan] Failed to create preset: "bad.slangp".\n',
                VALID_LOG + '[WARN] Falling back to stock.\n',
                VALID_LOG + '[Vulkan] Loading stock shader.\n',
                VALID_LOG.replace('AMD Radeon Graphics', 'llvmpipe (LLVM 20)'),
                VALID_LOG.replace('2880x2160', '1440x1080'),
                VALID_LOG.replace('MAILBOX', 'FIFO'),
                VALID_LOG + '[DEBUG] Creating swapchain with present mode: FIFO.\n',
                VALID_LOG.replace('vk_wayland', 'x'),
                '']:
            with self.subTest(log=log), self.assertRaises(bench.BenchmarkError):
                bench.validate_log(log, True)
        self.assertEqual(bench.validate_log(VALID_LOG, True)['gpu'], 'AMD Radeon Graphics')
        bench.validate_log(VALID_LOG + '[Vulkan] Loading stock shader.\n', False)
        bench.validate_log(VALID_LOG + '[ERROR] failed_to_start_audio_driver\n', True)

    def test_initial_swapchain_reports_supported_modes_only(self):
        log = VALID_LOG.replace('Creating swapchain with present mode: MAILBOX.',
                                'Swapchain supports present mode: MAILBOX.')
        metadata = bench.validate_log(log, True)
        self.assertEqual(metadata['present_mode'], 'MAILBOX')
        self.assertIn('inferred', metadata['present_mode_evidence'])
        log += '[DEBUG] [Vulkan] Swapchain supports present mode: IMMEDIATE.\n'
        self.assertEqual(bench.validate_log(log, True)['present_mode'], 'IMMEDIATE')
        with self.assertRaises(bench.BenchmarkError):
            bench.validate_log(log.replace('MAILBOX', 'FIFO').replace('IMMEDIATE', 'FIFO'), True)

    def test_slope_removes_constant_launch_cost(self):
        pairs = [{'short_seconds': 12, 'long_seconds': 24},
                 {'short_seconds': 17, 'long_seconds': 29},
                 {'short_seconds': 4, 'long_seconds': 16}]
        summary = bench.summarize(pairs, 1000, 7000)
        self.assertEqual(summary['median_ms_per_frame'], 2)
        self.assertEqual(summary['fps'], 500)
        self.assertEqual(summary['median_absolute_deviation_ms'], 0)
        for elapsed in (1, 0, float('nan')):
            with self.assertRaises(bench.BenchmarkError):
                bench.summarize([{'short_seconds': 1, 'long_seconds': elapsed}], 1, 2)

    def test_screenshot_must_have_output_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'shot.png'
            for width, height in [(2880, 2160), (320, 240)]:
                path.write_bytes(b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13)
                                 + b'IHDR' + struct.pack('>II', width, height))
                if width == 2880:
                    bench.validate_screenshot(path)
                else:
                    with self.assertRaises(bench.BenchmarkError):
                        bench.validate_screenshot(path)
            path.write_bytes(b'bad png')
            with self.assertRaises(bench.BenchmarkError):
                bench.validate_screenshot(path)

    def test_timeout_stops_the_child_and_keeps_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pid_path = root / 'pid'
            script = ('import os, time; from pathlib import Path; '
                      f'Path({str(pid_path)!r}).write_text(str(os.getpid())); '
                      'print("started", flush=True); time.sleep(60)')
            with self.assertRaisesRegex(bench.BenchmarkError, 'exceeded'):
                bench.execute([sys.executable, '-c', script], os.environ.copy(),
                              root / 'timeout.log', 1)
            self.assertIn('started', (root / 'timeout.log').read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pid_path.read_text()), 0)


class TestBenchmarkProtocol(unittest.TestCase):
    def test_full_protocol_with_fake_frontends(self):
        with tempfile.TemporaryDirectory(prefix='benchmark protocol ') as directory:
            root = Path(directory)
            bin_dir = root / 'bin'
            bin_dir.mkdir()
            runtime = root / 'runtime'
            runtime.mkdir()
            weston_pid = root / 'weston.pid'
            weston = bin_dir / 'weston'
            weston.write_text(f'''#!{sys.executable}
import os, pathlib, signal, socket, sys, time
if '--help' in sys.argv:
    print('--renderer=gl')
    sys.exit(0)
pathlib.Path({str(weston_pid)!r}).write_text(str(os.getpid()))
name = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--socket='))
server = socket.socket(socket.AF_UNIX)
server.bind(os.path.join(os.environ['XDG_RUNTIME_DIR'], name))
while True:
    time.sleep(1)
''')
            flatpak = bin_dir / 'flatpak'
            flatpak.write_text(f'''#!{sys.executable}
import pathlib, struct, sys, time
if '--version' in sys.argv:
    print('RetroArch fake for protocol testing')
    sys.exit(0)
def option(name):
    return next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith(name + '='))
time.sleep(int(option('--max-frames')) * .002)
print({VALID_LOG!r})
if not option('--set-shader'):
    print('[Vulkan] Loading stock shader.')
if '--max-frames-ss' in sys.argv:
    pathlib.Path(option('--max-frames-ss-path')).write_bytes(
        b'\\x89PNG\\r\\n\\x1a\\n' + struct.pack('>I', 13) + b'IHDR' + struct.pack('>II', 2880, 2160))
''')
            weston.chmod(0o755)
            flatpak.chmod(0o755)
            preset = root / 'preset $literal.slangp'
            preset.write_text('shaders = 1\n')
            image = root / 'source image.png'
            image.write_bytes(b'fake image')
            output = root / 'artifacts'
            with patch.dict(os.environ, {'PATH': str(bin_dir) + os.pathsep + os.environ['PATH'],
                                         'XDG_RUNTIME_DIR': str(runtime)}), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = bench.main([str(preset), str(image), '--output-dir', str(output),
                                     '--short-frames', '10', '--long-frames', '40',
                                     '--warmup-frames', '2', '--repeats', '2'])
            self.assertEqual(status, 0)
            report = json.loads((output / 'results.json').read_text())
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['output_size'], [2880, 2160])
            config = (output / 'retroarch.cfg').read_text()
            self.assertIn('vulkan_gpu_index = "0"', config)
            self.assertIn('aspect_ratio_index = "0"', config)
            for name in ('baseline', 'shader'):
                workload = report['workloads'][name]
                self.assertEqual([pair['order'] for pair in workload['pairs']],
                                 [['short', 'long'], ['long', 'short']])
                self.assertGreater(workload['summary']['fps'], 65)
                self.assertTrue((output / f'{name}.png').is_file())
                for pair in workload['pairs']:
                    for run in pair['runs'].values():
                        command = run['command']
                        self.assertNotIn('--max-frames-ss', command)
                        self.assertEqual(command[-1], str(image))
                        shader_arg = next(arg for arg in command if arg.startswith('--set-shader='))
                        self.assertEqual(shader_arg, f'--set-shader={preset if name == "shader" else ""}')
            with self.assertRaises(ProcessLookupError):
                os.kill(int(weston_pid.read_text()), 0)


if __name__ == '__main__':
    unittest.main()
