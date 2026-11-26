#!/usr/bin/env python3
"""Benchmark a Slang preset in Flatpak RetroArch on a private headless Weston display."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid


WIDTH, HEIGHT = 2880, 2160
APP_ID = 'org.libretro.RetroArch'


class BenchmarkError(RuntimeError):
    pass


def stop_process(process):
    """Stop our process group, including children, without touching other sessions."""
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait()
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def execute(command, env, log_path, timeout):
    with log_path.open('w') as log:
        start = time.perf_counter()
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        expired = threading.Event()

        def expire():
            if process.poll() is None:
                expired.set()
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

        # Popen.wait(timeout=...) polls in up to 50 ms steps on POSIX, which
        # quantizes short runs. A watchdog lets the timed wait block precisely.
        watchdog = threading.Timer(timeout, expire)
        watchdog.daemon = True
        watchdog.start()
        try:
            returncode = process.wait()
            elapsed = time.perf_counter() - start
        finally:
            watchdog.cancel()
            watchdog.join()
            stop_process(process)
    if expired.is_set():
        raise BenchmarkError(f'Run exceeded {timeout:g}s; see {log_path}')
    if returncode:
        raise BenchmarkError(f'Command exited with status {returncode}; see {log_path}')
    return elapsed


@contextmanager
def headless_display(output, env):
    runtime = env.get('XDG_RUNTIME_DIR')
    if not runtime or not Path(runtime).is_dir():
        raise BenchmarkError('XDG_RUNTIME_DIR must name an existing user runtime directory.')
    help_result = subprocess.run(['weston', '--help'], capture_output=True,
                                 text=True, timeout=10, env=env)
    # Older Weston versions expose --use-gl instead of the renderer option.
    renderer = '--renderer=gl' if '--renderer=' in help_result.stdout else '--use-gl'
    socket_name = f'crt-beans-bench-{uuid.uuid4().hex[:12]}'
    command = ['weston', '--backend=headless-backend.so', renderer,
               '--shell=kiosk-shell.so', '--no-config', '--idle-time=0',
               f'--width={WIDTH}', f'--height={HEIGHT}', '--scale=1',
               f'--socket={socket_name}']
    log_path = output / 'weston.log'
    with log_path.open('w') as log:
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            deadline = time.monotonic() + 20
            while not (Path(runtime) / socket_name).is_socket():
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise BenchmarkError(f'Headless Weston did not start; see {log_path}')
                time.sleep(.05)
            yield {**env, 'WAYLAND_DISPLAY': socket_name}, command, process
        finally:
            stop_process(process)


def write_config(path, output, gpu_index):
    settings = {
        'video_driver': 'vulkan', 'video_context_driver': 'vk_wayland',
        'vulkan_gpu_index': gpu_index, 'video_fullscreen': True,
        'video_fullscreen_x': WIDTH, 'video_fullscreen_y': HEIGHT,
        # Index 0 is RetroArch's fixed 4:3 mode. Image Viewer otherwise resets
        # the viewport to the image's pixel aspect when it sends SET_GEOMETRY.
        'aspect_ratio_index': 0, 'video_aspect_ratio': WIDTH / HEIGHT,
        'video_aspect_ratio_auto': False, 'video_force_aspect': True,
        'video_scale_integer': False,
        'video_vsync': False, 'video_threaded': False,
        'video_shader_enable': True, 'video_shader_delay': 0,
        'video_shader_watch_files': False, 'video_shader_subframes': 1,
        'video_black_frame_insertion': 0, 'video_frame_delay': 0,
        'video_frame_delay_auto': False, 'video_hdr_enable': False,
        'video_gpu_screenshot': True, 'video_font_enable': False,
        'video_smooth': False, 'video_max_swapchain_images': 3,
        'audio_driver': 'null', 'audio_enable': False, 'audio_sync': False,
        'microphone_enable': False, 'vrr_runloop_enable': False,
        'fastforward_ratio': 0.0, 'fastforward_frameskip': False,
        'pause_nonactive': False, 'pause_on_disconnect': False,
        'run_ahead_enabled': False, 'preemptive_frames_enable': False,
        'rewind_enable': False, 'fps_show': False, 'statistics_show': False,
        'menu_enable_widgets': False, 'config_save_on_exit': False,
        'ui_companion_enable': False, 'ui_companion_start_on_boot': False,
        'desktop_menu_enable': False, 'input_driver': 'wayland',
        'input_joypad_driver': 'null', 'input_autodetect_enable': False,
        'suspend_screensaver_enable': False,
        'history_list_enable': False, 'auto_overrides_enable': False,
        'content_runtime_log': False, 'content_runtime_log_aggregate': False,
        'savestate_auto_save': False, 'savestate_auto_load': False,
        'video_shader_dir': str(output), 'config_directory': str(output),
        'core_options_path': str(output / 'core-options.cfg'),
        'screenshot_directory': str(output), 'savefile_directory': str(output),
        'savestate_directory': str(output), 'log_verbosity': True,
        'frontend_log_level': 0,
    }
    lines = []
    for key, value in settings.items():
        if isinstance(value, bool):
            value = str(value).lower()
        else:
            value = str(value)
        if any(character in value for character in '\n\r"'):
            raise BenchmarkError('Output paths cannot contain quotes or newlines.')
        lines.append(f'{key} = "{value}"\n')
    path.write_text(''.join(lines))


def retroarch_command(config, output, preset, image, frames, screenshot=None):
    # Read the original preset in place so #reference, includes and LUT paths
    # retain their meaning. Only the artifacts directory needs host write access.
    command = ['flatpak', 'run', '--socket=wayland', '--nosocket=x11',
               '--nosocket=fallback-x11', '--nodevice=all', '--device=dri',
               '--filesystem=host:ro',
               '--filesystem=/tmp:ro',
               f'--filesystem={output}:rw', APP_ID, '--verbose',
               f'--config={config}', '-L', 'imageviewer',
               f'--set-shader={preset or ""}', f'--max-frames={frames}']
    if screenshot:
        command.extend(['--max-frames-ss', f'--max-frames-ss-path={screenshot}'])
    return command + [str(image)]


def validate_log(log, shader_enabled):
    """Fail closed on fallback, pacing or an unexpected graphics backend."""
    errors = re.findall(r'^.*(?:\[ERROR\]|\[Error\]|\[FATAL\]).*$', log, re.M)
    # 1.22.2 emits this UI notification even when audio is disabled on purpose.
    errors = [line for line in errors
              if not re.search(r'\[ERROR\] failed_to_start_audio_driver\s*$', line)]
    if errors:
        raise BenchmarkError('RetroArch reported an error: ' + errors[0])
    if 'falling back to stock' in log.lower():
        raise BenchmarkError('RetroArch fell back to its stock shader.')
    if shader_enabled and '[Vulkan] Loading stock shader.' in log:
        raise BenchmarkError('RetroArch loaded its stock shader instead of the preset.')
    if '[Vulkan] Found vulkan context: "vk_wayland"' not in log:
        raise BenchmarkError('The requested Vulkan Wayland context was not confirmed.')
    resolutions = re.findall(r'\[Vulkan\] Using resolution (\d+)x(\d+)', log)
    if not resolutions or any(tuple(map(int, size)) != (WIDTH, HEIGHT)
                              for size in resolutions):
        raise BenchmarkError(f'RetroArch did not confirm {WIDTH}x{HEIGHT} output.')
    gpus = re.findall(r'\[Vulkan\] Using GPU: "([^"]+)"', log)
    if not gpus:
        raise BenchmarkError('The selected Vulkan GPU was not reported.')
    gpu = gpus[-1]
    if re.search(r'llvmpipe|lavapipe|softpipe|swiftshader|software|llvmbox', gpu, re.I):
        raise BenchmarkError(f'Software rendering is unsuitable for this benchmark: {gpu}')
    modes = re.findall(r'Creating swapchain with present mode: (\w+)', log)
    evidence = 'created swapchain'
    if not modes:
        # RetroArch 1.22.2 logs only supported modes for the first swapchain.
        # With vsync/swap interval disabled it chooses IMMEDIATE, then MAILBOX.
        supported = re.findall(r'Swapchain supports present mode: (\w+)', log)
        modes = [next((mode for mode in ('IMMEDIATE', 'MAILBOX') if mode in supported), '')]
        evidence = 'inferred from supported modes with vsync disabled'
    if any(mode not in ('IMMEDIATE', 'MAILBOX') for mode in modes):
        raise BenchmarkError('An unthrottled Vulkan present mode was not confirmed.')
    return {'gpu': gpu, 'present_mode': modes[-1], 'present_mode_evidence': evidence}


def validate_screenshot(path):
    try:
        with path.open('rb') as file:
            header = file.read(24)
    except FileNotFoundError as exc:
        raise BenchmarkError(f'RetroArch did not produce {path}') from exc
    if (len(header) != 24 or header[:8] != b'\x89PNG\r\n\x1a\n'
            or header[12:16] != b'IHDR'):
        raise BenchmarkError(f'Invalid PNG screenshot: {path}')
    size = struct.unpack('>II', header[16:24])
    if size != (WIDTH, HEIGHT):
        raise BenchmarkError(f'Screenshot is {size[0]}x{size[1]}, expected {WIDTH}x{HEIGHT}.')


def summarize(pairs, short_frames, long_frames):
    samples = [1000 * (pair['long_seconds'] - pair['short_seconds'])
               / (long_frames - short_frames) for pair in pairs]
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise BenchmarkError('Nonpositive timing difference; increase the frame counts.')
    median = statistics.median(samples)
    return {'samples_ms_per_frame': samples, 'median_ms_per_frame': median,
            'fps': 1000 / median, 'min_ms_per_frame': min(samples),
            'max_ms_per_frame': max(samples),
            'median_absolute_deviation_ms': statistics.median(
                abs(value - median) for value in samples)}


def benchmark(args, output):
    env = {**os.environ, 'LC_ALL': 'C.UTF-8'}
    config = output / 'retroarch.cfg'
    write_config(config, output, args.gpu_index)
    version = subprocess.run(['flatpak', 'run', APP_ID, '--version'], env=env,
                             capture_output=True, text=True, timeout=20)
    if version.returncode:
        raise BenchmarkError('Cannot start Flatpak RetroArch: ' + version.stderr.strip())
    report = {'status': 'incomplete', 'created_at': datetime.now(timezone.utc).isoformat(),
              'preset': str(args.preset), 'image': str(args.image),
              'output_size': [WIDTH, HEIGHT], 'retroarch_version': version.stdout.strip(),
              'short_frames': args.short_frames, 'long_frames': args.long_frames,
              'warmup_frames': args.warmup_frames, 'repeats': args.repeats,
              'metric': 'RetroArch throughput including presentation; paired wall-time slope',
              'workloads': {}}
    with headless_display(output, env) as (display_env, weston_command, weston):
        report['weston_command'] = weston_command
        selected_gpu = None

        def run(label, preset, frames, screenshot=None):
            nonlocal selected_gpu
            if weston.poll() is not None:
                raise BenchmarkError(f'Weston exited unexpectedly; see {output / "weston.log"}')
            log_path = output / f'{label}.log'
            command = retroarch_command(config, output, preset, args.image, frames, screenshot)
            elapsed = execute(command, display_env, log_path, args.timeout)
            try:
                metadata = validate_log(log_path.read_text(errors='replace'), preset is not None)
                if selected_gpu is not None and metadata['gpu'] != selected_gpu:
                    raise BenchmarkError('The selected GPU changed between runs.')
                selected_gpu = metadata['gpu']
                if screenshot:
                    validate_screenshot(screenshot)
            except BenchmarkError as exc:
                raise BenchmarkError(f'{exc} See {log_path}') from exc
            return {'frames': frames, 'seconds': elapsed, **metadata, 'command': command}

        for name, preset in [('baseline', None), ('shader', args.preset)]:
            print(f'{name}: validating and warming up...', flush=True)
            validation = run(f'{name}-validation', preset, args.warmup_frames,
                             output / f'{name}.png')
            pairs = []
            for repeat in range(args.repeats):
                order = ['short', 'long'] if repeat % 2 == 0 else ['long', 'short']
                pair = {'order': order, 'runs': {}}
                for length in order:
                    frames = args.short_frames if length == 'short' else args.long_frames
                    print(f'{name}: pair {repeat + 1}/{args.repeats}, {frames} frames', flush=True)
                    result = run(f'{name}-{repeat + 1}-{length}', preset, frames)
                    pair[f'{length}_seconds'] = result['seconds']
                    pair['runs'][length] = result
                pairs.append(pair)
                report['workloads'][name] = {'validation': validation, 'pairs': pairs}
                (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
            summary = summarize(pairs, args.short_frames, args.long_frames)
            # A shader-free run should easily exceed the headless output's 60 Hz.
            # A low baseline can indicate compositor/driver pacing despite MAILBOX.
            if name == 'baseline' and summary['fps'] <= 65:
                raise BenchmarkError('Baseline is at or below 65 FPS; possible presentation cap. '
                                     'Results cannot establish uncapped shader throughput.')
            report['workloads'][name]['summary'] = summary
            print(f'{name}: {summary["median_ms_per_frame"]:.3f} ms/frame, '
                  f'{summary["fps"]:.1f} FPS', flush=True)
    report['status'] = 'complete'
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('preset', type=Path, help='Slang .slangp preset')
    parser.add_argument('image', type=Path, help='Image to render, at its native source resolution')
    parser.add_argument('--output-dir', type=Path,
                        help='New artifacts directory (default: a new directory under /tmp)')
    parser.add_argument('--short-frames', type=int, default=1000)
    parser.add_argument('--long-frames', type=int, default=7000)
    parser.add_argument('--warmup-frames', type=int, default=600)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--timeout', type=float, default=300, help='Seconds allowed per run')
    parser.add_argument('--gpu-index', type=int, default=0, help='RetroArch Vulkan GPU index')
    parser.add_argument('--output-resolution', type=int, nargs=2, default=[2880, 2160],
                        metavar=('WIDTH', 'HEIGHT'), help='Output size (default: 2880 2160)')
    args = parser.parse_args(argv)
    if any(size <= 0 for size in args.output_resolution):
        parser.error('output dimensions must be positive')
    if not 0 < args.short_frames < args.long_frames:
        parser.error('require 0 < --short-frames < --long-frames')
    if args.warmup_frames < 1 or args.repeats < 1 or args.gpu_index < 0:
        parser.error('warmup/repeats must be positive and GPU index must be nonnegative')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be finite and positive')
    for name in ('preset', 'image'):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file():
            parser.error(f'{name} is not a file: {path}')
        setattr(args, name, path)
    if args.preset.suffix.lower() != '.slangp':
        parser.error('preset must be a .slangp file')
    return args


def main(argv=None):
    global WIDTH, HEIGHT
    args = parse_args(argv)
    WIDTH, HEIGHT = args.output_resolution
    output = None
    try:
        for executable in ('flatpak', 'weston'):
            if not shutil.which(executable):
                raise BenchmarkError(f'{executable} is required on PATH. '
                                     'On Fedora, install Weston with: sudo dnf install weston')
        if args.output_dir:
            output = args.output_dir.expanduser().resolve()
            output.mkdir(parents=True, exist_ok=False)
        else:
            output = Path(tempfile.mkdtemp(prefix='crt-beans-benchmark-', dir='/tmp'))
        print(f'Artifacts: {output}', flush=True)
        report = benchmark(args, output)
        print(f'GPU: {report["workloads"]["shader"]["validation"]["gpu"]}')
        print(f'Results: {output / "results.json"}')
        return 0
    except (BenchmarkError, OSError, subprocess.TimeoutExpired) as exc:
        print(f'Benchmark failed: {exc}', file=sys.stderr)
        if output:
            print(f'Artifacts retained in {output}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f'Benchmark interrupted. Artifacts retained in {output}', file=sys.stderr)
        return 130


if __name__ == '__main__':
    sys.exit(main())
