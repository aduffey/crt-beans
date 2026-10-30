from dataclasses import dataclass
import numpy as np
import taichi as ti
import taichi.math as tm
from common import texelFetch, gamma_to_yuv, yuv_to_linear, gamma_to_linear


CHROMA_SUBCARRIER = (315.0 / 88.0) * 1e6
ACTIVE_LINE_TIME = 52.5e-6


@dataclass(frozen=True)
class ChromaInfo:
    periods_per_active_line: float
    periods_per_line: float
    offset_per_frame: float


def chroma_info_standard() -> ChromaInfo:
    periods_per_active_line = CHROMA_SUBCARRIER * ACTIVE_LINE_TIME
    return ChromaInfo(periods_per_active_line, 227.5, 0.5)


def chroma_info_nes(original_width: float, frame_count: int = 0) -> ChromaInfo:
    periods_per_active_line = (4.0 / 6.0) * original_width
    return ChromaInfo(periods_per_active_line, 682.0 / 3.0, 1.0 / 3.0)


def chroma_info_sms(original_width: float) -> ChromaInfo:
    periods_per_active_line = (10.0 / 15.0) * original_width
    return ChromaInfo(periods_per_active_line, 228.0, 0.0)


def chroma_info_pce(original_width: float) -> ChromaInfo:
    periods_per_active_line = (2.0 / 6.0) * original_width
    if original_width <= 280.0:
        periods_per_active_line = (4.0 / 6.0) * original_width
    elif original_width <= 375.0:
        periods_per_active_line = (3.0 / 6.0) * original_width
    return ChromaInfo(periods_per_active_line, 227.5, 0.5)


def chroma_info_genesis(original_width: float) -> ChromaInfo:
    if original_width > 300.0:
        periods_per_active_line = (8.0 / 15.0) * original_width
    else:
        periods_per_active_line = (10.0 / 15.0) * original_width
    return ChromaInfo(periods_per_active_line, 228.0, 0.0)


def chroma_info_snes(original_width: float, frame_count: int = 0) -> ChromaInfo:
    periods_per_active_line = (2.0 / 6.0) * original_width
    if original_width < 300.0:
        periods_per_active_line = (4.0 / 6.0) * original_width
    return ChromaInfo(periods_per_active_line, 682.0 / 3.0, 1.0 / 3.0)


def chroma_info_saturn(original_width: float) -> ChromaInfo:
    periods_per_active_line = original_width / 4.0
    if original_width < 340.0:
        periods_per_active_line = (65.0 / (2.0 * 61.0)) * original_width
    elif original_width < 400.0:
        periods_per_active_line = original_width / 2.0
    elif original_width < 680.0:
        periods_per_active_line = (65.0 / (4.0 * 61.0)) * original_width
    return ChromaInfo(periods_per_active_line, 227.5, 0.5)


def chroma_info_psx(original_width: float) -> ChromaInfo:
    # Pixel-clock divisors for 256/320/368/512/640 modes, with capture padding.
    # The NTSC video clock is 15 times the chroma subcarrier frequency.
    periods_per_active_line = original_width * 4.0 / 15.0
    if original_width <= 280.0:
        periods_per_active_line = original_width * 10.0 / 15.0
    elif original_width <= 350.0:
        periods_per_active_line = original_width * 8.0 / 15.0
    elif original_width <= 400.0:
        periods_per_active_line = original_width * 7.0 / 15.0
    elif original_width <= 560.0:
        periods_per_active_line = original_width * 5.0 / 15.0
    return ChromaInfo(periods_per_active_line, 227.5, 0.5)


def chroma_info_n64() -> ChromaInfo:
    periods_per_active_line = (640.0 * 227.5) / 773.5
    return ChromaInfo(periods_per_active_line, 227.5, 0.5)


# Platform aliases
chroma_info_tg16 = chroma_info_pce
chroma_info_megadrive = chroma_info_genesis
chroma_info_ps1 = chroma_info_psx


def get_chroma_info(system: str, original_width: float, frame_count: int = 0) -> ChromaInfo:
    """Derive system-specific NTSC subcarrier timing information.

    Matches timing derivations in slang-shaders/shaders/analog/chroma_carrier.inc
    and verified console hardware specifications.
    """
    sys_str = system.lower()
    if sys_str == 'nes':
        return chroma_info_nes(original_width, frame_count)
    elif sys_str == 'sms':
        return chroma_info_sms(original_width)
    elif sys_str in ('tg16', 'pce'):
        return chroma_info_pce(original_width)
    elif sys_str in ('genesis', 'megadrive'):
        return chroma_info_genesis(original_width)
    elif sys_str == 'snes':
        return chroma_info_snes(original_width, frame_count)
    elif sys_str == 'saturn':
        return chroma_info_saturn(original_width)
    elif sys_str in ('ps1', 'psx'):
        return chroma_info_psx(original_width)
    elif sys_str == 'n64':
        return chroma_info_n64()
    return chroma_info_standard()


def eval_chroma_carrier(
    t: float, line: int, frame_count: int,
    periods_per_active_line: float,
    periods_per_line: float,
    offset_per_frame: float,
    interlaced: int | bool = False,
    odd_field_first: int | bool = True,
) -> float:
    pos_val = periods_per_active_line * t
    position_offset = pos_val - np.floor(pos_val)
    line_fract = periods_per_line - np.floor(periods_per_line)
    if interlaced:
        is_even_field = line % 2
        field_line = line // 2
        phase_shift = 0 if odd_field_first else 1
        effective_frame = frame_count + phase_shift
        interlaced_frame = ((effective_frame + 4 - is_even_field) // 2) % 2
        raw_offset = 3 * interlaced_frame + 5 * is_even_field
        field_line_offset = raw_offset - 6 if raw_offset >= 6 else raw_offset
        line_offset = line_fract * (field_line + field_line_offset)
        frame_offset = 0.0
    else:
        line_offset = line_fract * line
        frame_offset = offset_per_frame * (frame_count % 2)
    return float(2.0 * np.pi * (position_offset + line_offset + frame_offset))


@ti.func
def chroma_carrier(t: float, line: int, frame_count: int,
                   periods_per_active_line: float,
                   periods_per_line: float,
                   offset_per_frame: float,
                   interlaced: int,
                   odd_field_first: int) -> float:
    pos_val = periods_per_active_line * t
    position_offset = pos_val - tm.floor(pos_val)
    line_fract = periods_per_line - tm.floor(periods_per_line)
    line_offset = 0.0
    frame_offset = 0.0
    if interlaced != 0:
        is_even_field = line % 2
        field_line = line // 2
        phase_shift = 1 if odd_field_first == 0 else 0
        effective_frame = frame_count + phase_shift
        interlaced_frame = ((effective_frame + 4 - is_even_field) // 2) % 2
        raw_offset = 3 * interlaced_frame + 5 * is_even_field
        field_line_offset = raw_offset - 6 if raw_offset >= 6 else raw_offset
        line_offset = line_fract * (field_line + field_line_offset)
    else:
        line_offset = line_fract * line
        frame_offset = offset_per_frame * (frame_count % 2)
    return 2.0 * np.pi * (position_offset + line_offset + frame_offset)


@ti.func
def carrier_exact_4tap(u: float, wc: float) -> tm.vec4:
    c1 = tm.cos(0.5 * wc)
    c3 = tm.cos(1.5 * wc)
    cu = tm.cos(u * wc)
    s1 = tm.sin(0.5 * wc)
    s3 = tm.sin(1.5 * wc)
    su = tm.sin(u * wc)

    e3 = 0.5 * (cu - c1) / (c3 - c1)
    e1 = 0.5 - e3
    o3 = (0.5 * su - u * s1) / (s3 - 3.0 * s1)
    o1 = u - 3.0 * o3

    return tm.vec4(e3 - o3, e1 - o1, e1 + o1, e3 + o3)


@ti.func
def interpolate_chroma(coord_x: float, target_y: int, Source: ti.template(), samples: int, height: int, wc: float) -> tm.vec2:
    result = tm.vec2(0.0, 0.0)
    if 0 <= target_y < height:
        texel_coord = coord_x * samples
        texel_floor = tm.floor(texel_coord - 0.5)
        d = (texel_coord - 0.5) - texel_floor
        u = d - 0.5
        w = carrier_exact_4tap(u, wc)
        base_x = int(texel_floor)
        for k in ti.static([-1, 0, 1, 2]):
            weight = 0.0
            if k == -1:
                weight = w.x
            elif k == 0:
                weight = w.y
            elif k == 1:
                weight = w.z
            else:
                weight = w.w
            sx = base_x + k
            if 0 <= sx < samples:
                s = Source[target_y, sx]
                result += weight * tm.vec2(s.y, s.z)
    return result


@ti.func
def yuv_to_linear_taichi(yuv: tm.vec3, gamma: float) -> tm.vec3:
    r = yuv.x + 1.13988303 * yuv.z
    g = yuv.x - 0.39464234 * yuv.y - 0.58062185 * yuv.z
    b = yuv.x + 2.03206187 * yuv.y
    rgb = tm.clamp(tm.vec3(r, g, b), 0.0, 1.0)
    return tm.pow(rgb, gamma)


def filter_fragment(image_in, output_dim, L: tm.vec3):
    (in_height, in_width, in_planes) = image_in.shape
    (out_height, out_width, out_planes) = output_dim
    field_in = ti.Vector.field(n=3, dtype=float, shape=(in_height, in_width))
    field_in.from_numpy(image_in)
    field_out = ti.Vector.field(n=3, dtype=float, shape=(out_height, out_width))
    taichi_filter_fragment(field_in, field_out, L)
    return field_out.to_numpy()


@ti.kernel
def taichi_filter_fragment(field_in: ti.template(), field_out: ti.template(), L: tm.vec3):
    (in_height, in_width) = field_in.shape
    (out_height, out_width) = field_out.shape
    SourceSize = tm.vec4(in_width, in_height, 1 / in_width, 1 / in_height)
    for y, x in field_out:
        vTexCoord = tm.vec2((x + 0.5) / out_width, (y + 0.5) / out_height)
        field_out[y, x] = filter_sim2(vTexCoord, field_in, SourceSize, L)
    return


@ti.func
def filter_sim(vTexCoord: tm.vec2, Source, SourceSize: tm.vec4, L: tm.vec3) -> tm.vec3:
    max_L = tm.max(tm.max(L.r, L.g), L.b)
    L_rcp = 1.0 / L

    filtered = tm.vec3(0.0)
    pix_y = int(tm.floor(vTexCoord.y * SourceSize.y))
    t = vTexCoord.x
    for pix_x in range(int(tm.floor(SourceSize.x * (vTexCoord.x - max_L))),
                       int(tm.floor(SourceSize.x * (vTexCoord.x + max_L))) + 1):
        s = texelFetch(Source, tm.ivec2(pix_x, pix_y))
        t0 = tm.vec3(pix_x * SourceSize.z)
        t1 = t0 + tm.vec3(SourceSize.z)
        t0 = tm.clamp(t0, t - L, t + L)
        t1 = tm.clamp(t1, t - L, t + L)
        # Integral of s * (1 / L) * (0.5 + 0.5 * cos(PI * (t - t_x) / L)) dt_x over t0 to t1
        filtered += 0.5 * s * L_rcp * (t1 - t0 + (L / np.pi) *
                                       (tm.sin(L_rcp * ((np.pi * t) - np.pi * t0)) - tm.sin(L_rcp * ((np.pi * t) - np.pi * t1))))
    return filtered


@ti.func
def filter_sim2(vTexCoord: tm.vec2, Source, SourceSize: tm.vec4, L: tm.vec3) -> tm.vec3:
    max_L = tm.max(tm.max(L.r, L.g), L.b)

    filtered = tm.vec3(0.0)
    pix_y = int(tm.floor(vTexCoord.y * SourceSize.y))
    t = vTexCoord.x

    # Set up loop bounds
    start = int(tm.floor(SourceSize.x * (vTexCoord.x - max_L)))
    end = int(tm.floor(SourceSize.x * (vTexCoord.x + max_L))) + 1

    # Set up the first common term for the left. The right side of the integral
    # for iteration i is the left side of the integral for iteration i + 1, so
    # we can avoid computing the term twice.
    t0 = start * SourceSize.z
    u0 = tm.clamp((t - t0) / L, -1.0, 1.0)
    left = np.pi * u0 + tm.sin(np.pi * u0)
    for pix_x in range(start, end):
        # Integral of s / L * (0.5 + 0.5 * cos(PI * (t - t_x) / L)) dt_x over t0 to t1
        s = texelFetch(Source, tm.ivec2(pix_x, pix_y))
        t1 = (pix_x + 1) * SourceSize.z
        u1 = tm.clamp((t - t1) / L, -1.0, 1.0)
        right = np.pi * u1 + tm.sin(np.pi * u1)
        filtered += s * (left - right)
        left = right
    return filtered / (np.pi * 2.0)


@ti.kernel
def taichi_ntsc_encode_modulate(
    field_yuv: ti.template(),
    field_composite: ti.template(),
    frame_count: int,
    periods_per_active_line: float,
    periods_per_line: float,
    offset_per_frame: float,
    interlaced: int,
    odd_field_first: int,
):
    (height, samples) = field_composite.shape
    for y, x in field_composite:
        t = (x + 0.5) / samples
        carrier = chroma_carrier(t, y, frame_count, periods_per_active_line, periods_per_line, offset_per_frame, interlaced, odd_field_first)
        yuv = field_yuv[y, x]
        field_composite[y, x] = yuv.x + yuv.y * tm.sin(carrier) + yuv.z * tm.cos(carrier)


@ti.kernel
def taichi_ntsc_decode_demodulate(
    field_composite: ti.template(),
    field_demod: ti.template(),
    frame_count: int,
    periods_per_active_line: float,
    periods_per_line: float,
    offset_per_frame: float,
    interlaced: int,
    odd_field_first: int,
):
    (height, samples) = field_composite.shape
    for y, x in field_demod:
        t = (x + 0.5) / samples
        carrier = chroma_carrier(t, y, frame_count, periods_per_active_line, periods_per_line, offset_per_frame, interlaced, odd_field_first)
        comp = field_composite[y, x]
        chroma_u = comp * 2.0 * tm.sin(carrier)
        chroma_v = comp * 2.0 * tm.cos(carrier)
        field_demod[y, x] = tm.vec3(comp, chroma_u, chroma_v)


@ti.kernel
def taichi_ntsc_decode_filter(
    field_in: ti.template(),
    field_out: ti.template(),
    chroma_filter_width: float,
):
    (height, samples) = field_in.shape
    width_rcp = 1.0 / chroma_filter_width
    i_bound = int(tm.floor(chroma_filter_width))

    for y, x in field_out:
        weight_sum = 0.0
        chroma = tm.vec2(0.0, 0.0)
        low_luma = 0.0
        for i in range(-i_bound, i_bound + 1):
            coeff = 0.5 + 0.5 * tm.cos(np.pi * i * width_rcp)
            weight_sum += coeff
            sample_x = x + i
            if 0 <= sample_x < samples:
                sample_val = field_in[y, sample_x]
                chroma += coeff * tm.vec2(sample_val.y, sample_val.z)
                low_luma += coeff * sample_val.x
        if weight_sum > 0.0:
            chroma = chroma / weight_sum
            low_luma = low_luma / weight_sum
        comp = field_in[y, x].x
        # Low-frequency signal-domain luma is detector evidence, not output Y.
        field_out[y, x] = tm.vec4(comp, chroma.x, chroma.y, low_luma)


@ti.func
def ntsc_comb_sample(source: ti.template(), y: int, x: int) -> tm.vec3:
    sample = tm.vec3(0.0)
    if 0 <= y < source.shape[0]:
        value = source[y, x]
        sample = tm.vec3(value.y, value.z, value.w)
    return sample


@ti.func
def ntsc_adaptive_phase_comb(
    middle: tm.vec3, upper: tm.vec3, lower: tm.vec3,
    upper2: tm.vec3, lower2: tm.vec3,
    upper_period: tm.vec3, lower_period: tm.vec3,
    left: tm.vec2, right: tm.vec2, baseline: tm.vec2,
    period: int, density: float, valid_upper: int, valid_lower: int,
) -> tm.vec2:
    """Phase-complete spatial candidates with a bounded correction to baseline.

    Samples hold (U, V, low-frequency Y) in the composite signal's gamma domain.
    Keep thresholds and the normalized 0.08 cap in sync with ntsc_decode_comb.slang.
    """
    uv_scale = tm.vec2(1.0 / 0.436, 1.0 / 0.615)
    m = tm.vec2(middle.x, middle.y)
    u = tm.vec2(upper.x, upper.y)
    d = tm.vec2(lower.x, lower.y)
    delta_left = (m - left) * uv_scale
    delta_right = (right - m) * uv_scale
    energy = density * tm.sqrt(0.5 * (tm.dot(delta_left, delta_left) + tm.dot(delta_right, delta_right)))
    horizontal = tm.smoothstep(0.02, 0.08, energy)

    periodic_u = 0.0
    periodic_d = 0.0
    if valid_upper != 0:
        periodic_u = 1.0 - tm.smoothstep(0.04, 0.16, tm.length((m - tm.vec2(upper_period.x, upper_period.y)) * uv_scale))
    if valid_lower != 0:
        periodic_d = 1.0 - tm.smoothstep(0.04, 0.16, tm.length((m - tm.vec2(lower_period.x, lower_period.y)) * uv_scale))

    luma_u = ti.abs(middle.z - upper.z)
    luma_d = ti.abs(middle.z - lower.z)
    center_luma = tm.max(luma_u, luma_d)
    center = 0.5 * m + 0.25 * (u + d)
    candidate_u = 0.5 * (m + u)
    candidate_d = 0.5 * (m + d)
    if period == 3:
        # The modeled NES/SNES carrier advances 120 degrees per field line.
        center = (m + u + d) / 3.0
        candidate_u = (m + u + tm.vec2(upper2.x, upper2.y)) / 3.0
        candidate_d = (m + d + tm.vec2(lower2.x, lower2.y)) / 3.0
        luma_u = tm.max(luma_u, ti.abs(middle.z - upper2.z))
        luma_d = tm.max(luma_d, ti.abs(middle.z - lower2.z))

    confidence = tm.vec3(
        tm.max(periodic_u, periodic_d) * (1.0 - tm.smoothstep(0.02, 0.06, center_luma)),
        periodic_u * (1.0 - tm.smoothstep(0.02, 0.06, luma_u)),
        periodic_d * (1.0 - tm.smoothstep(0.02, 0.06, luma_d)),
    ) * horizontal
    if valid_upper == 0 or valid_lower == 0:
        confidence.x = 0.0
    weights = confidence * confidence
    selected = (weights.x * center + weights.y * candidate_u + weights.z * candidate_d) / tm.max(weights.x + weights.y + weights.z, 1.0e-12)
    strength = tm.max(confidence.x, tm.max(confidence.y, confidence.z))
    correction = strength * (selected - baseline)
    correction *= tm.min(1.0, 0.08 / tm.max(tm.length(correction * uv_scale), 1.0e-12))
    return baseline + correction


@ti.kernel
def taichi_ntsc_decode_comb(
    field_in: ti.template(),
    field_out: ti.template(),
    filter_type: float,
    frame_count: int,
    periods_per_active_line: float,
    periods_per_line: float,
    offset_per_frame: float,
    gamma: float,
    interlaced: int,
    odd_field_first: int,
):
    (height, samples) = field_in.shape
    line_step = 2 if interlaced != 0 else 1
    wc = 2.0 * np.pi * (periods_per_active_line / samples)
    line_phase = periods_per_line - tm.floor(periods_per_line)
    period = 2 if ti.abs(line_phase - 0.5) < 1.0e-4 else 3
    density = (samples / periods_per_active_line) * (CHROMA_SUBCARRIER * ACTIVE_LINE_TIME / 900.0)
    for y, x in field_out:
        u = (x + 0.5) / samples
        middle = field_in[y, x]
        composite = middle.x
        middle_chroma = tm.vec2(middle.y, middle.z)
        chroma = middle_chroma

        if filter_type > 0.5:
            offset = (227.5 - periods_per_line) / periods_per_active_line
            upper_chroma = tm.vec2(0.0)
            lower_chroma = tm.vec2(0.0)
            upper = tm.vec3(0.0)
            lower = tm.vec3(0.0)
            if offset == 0.0:
                # Aligned scanlines need no fractional-sample interpolation.
                upper = ntsc_comb_sample(field_in, y - line_step, x)
                lower = ntsc_comb_sample(field_in, y + line_step, x)
                upper_chroma = tm.vec2(upper.x, upper.y)
                lower_chroma = tm.vec2(lower.x, lower.y)
            else:
                upper_chroma = interpolate_chroma(u - offset, y - line_step, field_in, samples, height, wc)
                lower_chroma = interpolate_chroma(u + offset, y + line_step, field_in, samples, height, wc)

            if filter_type < 1.5:
                # 3-line comb with boundary handling
                if y < line_step:
                    chroma = 0.5 * middle_chroma + 0.5 * lower_chroma
                elif y >= height - line_step:
                    chroma = 0.5 * middle_chroma + 0.5 * upper_chroma
                else:
                    chroma = 0.5 * middle_chroma + 0.25 * upper_chroma + 0.25 * lower_chroma
            else:
                # Adaptive directional comb with edge fallback
                uv_scale = tm.vec2(1.0 / 0.436, 1.0 / 0.615)
                m_scaled = middle_chroma * uv_scale
                u_scaled = upper_chroma * uv_scale
                d_scaled = lower_chroma * uv_scale

                diff_u = tm.length(m_scaled - u_scaled)
                diff_d = tm.length(m_scaled - d_scaled)

                if y < line_step:
                    diff_u = 999.0
                if y >= height - line_step:
                    diff_d = 999.0

                diff_min = tm.min(diff_u, diff_d)
                diff_sum = diff_u + diff_d + 1.0e-5
                k_dir = tm.clamp(((diff_u - diff_d) / diff_sum) * 3.0, -1.0, 1.0)

                combed_dir = 0.5 * middle_chroma + 0.25 * ((1.0 - k_dir) * upper_chroma + (1.0 + k_dir) * lower_chroma)
                if y < line_step:
                    combed_dir = 0.5 * (middle_chroma + lower_chroma)
                elif y >= height - line_step:
                    combed_dir = 0.5 * (middle_chroma + upper_chroma)

                w_comb = 1.0 - tm.smoothstep(0.06, 0.24, diff_min)

                chroma = tm.mix(middle_chroma, combed_dir, w_comb)

                # Zero-phase systems (Genesis/SMS) retain the existing fallback.
                # All new candidates use raw, unshifted samples from this field.
                if line_phase > 1.0e-4:
                    if offset != 0.0:
                        upper = ntsc_comb_sample(field_in, y - line_step, x)
                        lower = ntsc_comb_sample(field_in, y + line_step, x)
                    upper2 = ntsc_comb_sample(field_in, y - 2 * line_step, x)
                    lower2 = ntsc_comb_sample(field_in, y + 2 * line_step, x)
                    upper_period = upper2
                    lower_period = lower2
                    if period == 3:
                        upper_period = ntsc_comb_sample(field_in, y - 3 * line_step, x)
                        lower_period = ntsc_comb_sample(field_in, y + 3 * line_step, x)
                    left = ntsc_comb_sample(field_in, y, ti.max(x - 1, 0))
                    right = ntsc_comb_sample(field_in, y, ti.min(x + 1, samples - 1))
                    chroma = ntsc_adaptive_phase_comb(
                        tm.vec3(middle.y, middle.z, middle.w), upper, lower,
                        upper2, lower2, upper_period, lower_period,
                        tm.vec2(left.x, left.y), tm.vec2(right.x, right.y), chroma,
                        period, density, int(y >= period * line_step), int(y + period * line_step < height),
                    )

        carrier = chroma_carrier(u, y, frame_count, periods_per_active_line, periods_per_line, offset_per_frame, interlaced, odd_field_first)
        luma = composite - chroma.x * tm.sin(carrier) - chroma.y * tm.cos(carrier)
        field_out[y, x] = yuv_to_linear_taichi(tm.vec3(luma, chroma.x, chroma.y), gamma)


def simulate_linearize(image_in: np.ndarray, gamma: float = 2.4) -> np.ndarray:
    """Direct conversion of input gamma image to linear RGB without analog filtering."""
    return gamma_to_linear(image_in, gamma)


def simulate_rgb(
    image_in: np.ndarray,
    samples: int = 900,
    gamma: float = 2.4,
    rgb_bandwidth: float = 4.0e6,
) -> np.ndarray:
    """Simulate an analog RGB signal with per-channel horizontal low-pass filtering."""
    in_h, in_w, _ = image_in.shape
    L_val = 1.0 / (rgb_bandwidth * ACTIVE_LINE_TIME * 2.0)
    L = tm.vec3(L_val, L_val, L_val)

    field_in = ti.Vector.field(3, dtype=float, shape=(in_h, in_w))
    field_in.from_numpy(image_in)
    field_out = ti.Vector.field(3, dtype=float, shape=(in_h, samples))
    taichi_filter_fragment(field_in, field_out, L)

    return gamma_to_linear(field_out.to_numpy(), gamma)


def simulate_svideo(
    image_in: np.ndarray,
    samples: int = 900,
    gamma: float = 2.4,
    luma_bandwidth: float = 4.0e6,
    chroma_bandwidth: float = 1.2e6,
) -> np.ndarray:
    """Simulate an S-Video signal with separate luma and chroma horizontal filtering."""
    in_h, in_w, _ = image_in.shape
    yuv_in = gamma_to_yuv(image_in)

    L = tm.vec3(
        1.0 / (luma_bandwidth * ACTIVE_LINE_TIME * 2.0),
        1.0 / (chroma_bandwidth * ACTIVE_LINE_TIME * 2.0),
        1.0 / (chroma_bandwidth * ACTIVE_LINE_TIME * 2.0),
    )

    field_in = ti.Vector.field(3, dtype=float, shape=(in_h, in_w))
    field_in.from_numpy(yuv_in)
    field_out = ti.Vector.field(3, dtype=float, shape=(in_h, samples))
    taichi_filter_fragment(field_in, field_out, L)

    return yuv_to_linear(field_out.to_numpy(), gamma)


def simulate_ntsc(
    image_in: np.ndarray,
    samples: int = 900,
    gamma: float = 2.4,
    luma_bandwidth: float = 4.0e6,
    chroma_bandwidth: float = 1.2e6,
    ntsc_filter_type: str | float = 'adaptive',
    ntsc_system: str = 'standard',
    ntsc_frame_count: int = 0,
    interlacing_type: str | float = 'bob',
    odd_field_first: bool | float = True,
) -> np.ndarray:
    """Simulate NTSC composite video encoding and decoding."""
    in_h, in_w, _ = image_in.shape
    yuv_in = gamma_to_yuv(image_in)

    ci = get_chroma_info(ntsc_system, in_w, ntsc_frame_count)
    active_line_seconds = ci.periods_per_active_line / CHROMA_SUBCARRIER

    L = tm.vec3(
        1.0 / (luma_bandwidth * active_line_seconds * 2.0),
        1.0 / (chroma_bandwidth * active_line_seconds * 2.0),
        1.0 / (chroma_bandwidth * active_line_seconds * 2.0),
    )

    is_interlaced_flag = 1 if (in_h > 300 and str(interlacing_type).lower() in ('bob', 'weave', '0', '1', '0.0', '1.0')) else 0
    odd_field_first_flag = 1 if odd_field_first else 0

    # Pass 1: Horizontal low-pass filter on YUV
    f_yuv_in = ti.Vector.field(3, dtype=float, shape=(in_h, in_w))
    f_yuv_in.from_numpy(yuv_in)
    f_yuv_filtered = ti.Vector.field(3, dtype=float, shape=(in_h, samples))
    taichi_filter_fragment(f_yuv_in, f_yuv_filtered, L)

    # Pass 1b: Modulate to composite signal
    f_composite = ti.field(dtype=float, shape=(in_h, samples))
    taichi_ntsc_encode_modulate(
        f_yuv_filtered, f_composite, ntsc_frame_count,
        ci.periods_per_active_line, ci.periods_per_line, ci.offset_per_frame,
        is_interlaced_flag, odd_field_first_flag,
    )

    # Pass 2: Demodulate composite signal
    f_demod = ti.Vector.field(3, dtype=float, shape=(in_h, samples))
    taichi_ntsc_decode_demodulate(
        f_composite, f_demod, ntsc_frame_count,
        ci.periods_per_active_line, ci.periods_per_line, ci.offset_per_frame,
        is_interlaced_flag, odd_field_first_flag,
    )

    # Pass 3: Chroma low-pass filter (carrier-locked Hann window, -6 dB at fc / 3 ~= 1.2 MHz)
    chroma_filter_width = 1.5 * samples / ci.periods_per_active_line
    f_demod_filtered = ti.Vector.field(4, dtype=float, shape=(in_h, samples))
    taichi_ntsc_decode_filter(f_demod, f_demod_filtered, chroma_filter_width)

    # Pass 4: Y/C separation comb filter and conversion to linear RGB
    ft_val = 2.0
    if isinstance(ntsc_filter_type, str):
        ft_lower = ntsc_filter_type.lower()
        if ft_lower == 'notch':
            ft_val = 0.0
        elif ft_lower == 'comb':
            ft_val = 1.0
        elif ft_lower in ('adaptive', 'adaptive_comb'):
            ft_val = 2.0
    else:
        ft_val = float(ntsc_filter_type)

    f_linear_out = ti.Vector.field(3, dtype=float, shape=(in_h, samples))
    taichi_ntsc_decode_comb(
        f_demod_filtered, f_linear_out, ft_val, ntsc_frame_count,
        ci.periods_per_active_line, ci.periods_per_line, ci.offset_per_frame, gamma,
        is_interlaced_flag, odd_field_first_flag,
    )

    return f_linear_out.to_numpy()


def simulate_analog(
    image_in: np.ndarray,
    analog_type: str = 'rgb',
    samples: int = 900,
    gamma: float = 2.4,
    rgb_bandwidth: float = 4.0e6,
    luma_bandwidth: float = 4.0e6,
    chroma_bandwidth: float = 1.2e6,
    ntsc_filter_type: str = 'adaptive',
    ntsc_system: str = 'standard',
    ntsc_frame_count: int = 0,
    interlacing_type: str | float = 'bob',
    odd_field_first: bool | float = True,
) -> np.ndarray:
    """High-level dispatcher for analog simulation frontends."""
    if analog_type in (None, 'none', 'linearize', 'disabled', False):
        return simulate_linearize(image_in, gamma)

    atype = str(analog_type).lower()
    if atype == 'rgb':
        return simulate_rgb(image_in, samples, gamma, rgb_bandwidth)
    elif atype in ('svideo', 's-video'):
        return simulate_svideo(image_in, samples, gamma, luma_bandwidth, chroma_bandwidth)
    elif atype in ('ntsc', 'composite'):
        return simulate_ntsc(
            image_in,
            samples=samples,
            gamma=gamma,
            luma_bandwidth=luma_bandwidth,
            chroma_bandwidth=chroma_bandwidth,
            ntsc_filter_type=ntsc_filter_type,
            ntsc_system=ntsc_system,
            ntsc_frame_count=ntsc_frame_count,
            interlacing_type=interlacing_type,
            odd_field_first=odd_field_first,
        )
    else:
        raise ValueError(
            f"Unknown analog type: {analog_type}. "
            "Supported types: 'none', 'linearize', 'rgb', 'svideo', 'ntsc'."
        )


apply_analog = simulate_analog
