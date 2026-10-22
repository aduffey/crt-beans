// SPDX-License-Identifier: MIT

#define PI 3.141592653589793
#define CHROMA_SUBCARRIER ((315.0 / 88.0) * 1e6)

struct chroma_info {
    float periods_per_active_line;
    float periods_per_line;
    float offset_per_frame;
};

// NOTE: The chroma phases may not be accurate in an absolute sense. We can't
// we can't really tell when the chroma clock started, so we just set 0 to be
// the first input sample. So the phase may be off by some amount. This won't
// affect the patterns that are seen, but it may affect the actual colors.
// TODO better explanation.

chroma_info chroma_info_standard() {
    // We'll just assume that the active line time is 52.5 microseconds. We
    // don't have any way to figure this out from the input, but this should
    // be pretty close.
    const float periods_per_active_line = CHROMA_SUBCARRIER * 52.5e-6;
    return chroma_info(periods_per_active_line, 227.5, 0.5);
}

chroma_info chroma_info_nes(vec4 original_size, uint frame_count) {
    // The NES master clock (MCLK) is 21477270 Hz (6 * 3579545 Hz). There are 4
    // cycles per pixel.
    //
    // A full line is usually 1364 cycles long. A line is 227.33... chroma
    // periods long.
    //
    // A field is 262 lines. So a field is 1364 / 6 * 262 = 59561.33...
    // chroma periods long. Every other field has one line that is 1360 cycles
    // long, so every other field is 1364 / 6 * 261 + 1360 / 6 = 59560.66...
    // chroma periods long. It may be possible for a game to disable this, but
    // it seems unlikely that it happened commonly (Battletoads is one
    // exception that I have seen referenced).
    //
    // Note that this is essentially the same as the SNES.
    //
    // The NES is interesting in that it generates composite video directly. To
    // truly simulate this aspect, we would need a special encoder and to use
    // an emulator with a "raw" output.
    //
    // See: https://www.nesdev.org/wiki/NTSC_video

    float periods_per_active_line = 4.0 / 6.0 * original_size.x;
    return chroma_info(periods_per_active_line, 682.0 / 3.0, 1.0 / 3.0);
}

chroma_info chroma_info_sms(vec4 original_size) {
    // Same as Genesis in H32 mode.
    float periods_per_active_line = 10.0 / 15.0 * original_size.x;
    return chroma_info(periods_per_active_line, 228.0, 0.0);
}

chroma_info chroma_info_pce(vec4 original_size) {
    // The PC Engine/TurboGrafx-16 master clock (MCLK) is approximately
    // 21.47727 MHz, or 6 times the NTSC chroma subcarrier frequency.
    // The HuC6260 VCE divides MCLK by 4, 3, or 2 for dot clocks of
    // approximately 5.37, 7.16, or 10.74 MHz, respectively. This gives
    // 4/6, 3/6, or 2/6 chroma periods per pixel.
    //
    // A full line is 1365 MCLK cycles, including blanking, regardless
    // of the dot-clock setting. Therefore, there are 1365 / 6 = 227.5
    // chroma periods per line, giving a half-period phase shift between
    // successive lines.
    //
    // The VCE supports 262 or 263 lines per non-interlaced frame.
    // At 262 lines, 262 * 227.5 = 59605 chroma periods, so the phase
    // repeats each frame. At 263 lines, 263 * 227.5 = 59832.5 periods,
    // so the phase flips each frame. We can't tell from the core output
    // which line count a game is using, so we will assume 263, which seems
    // to be more common and will result in cleaner output.
    //
    // See:
    // https://raw.githubusercontent.com/mamedev/mame/master/src/devices/video/huc6260.cpp
    // https://www.pcengine-fx.com/forums/index.php?topic=11550.0
    // https://pcengine.proboards.com/thread/639/screen-resolution-setup-sprite-limits?page=2

    // Default to 512 px mode (2/6 cycles per dot)
    float periods_per_active_line = 2.0 / 6.0 * original_size.x;
    if (original_size.x <= 280.0) { // 256px mode (4/6 cycles per dot)
        periods_per_active_line = 4.0 / 6.0 * original_size.x;
    } else if (original_size.x <= 375.0) { // 320/352px mode (3/6 cycles per dot)
        periods_per_active_line = 3.0 / 6.0 * original_size.x;
    }

    return chroma_info(periods_per_active_line, 227.5, 0.5);
}

chroma_info chroma_info_genesis(vec4 original_size) {
    // The Genesis master clock (MCLK) is 53693175 Hz, and the chroma clock is
    // MCLK/15 (3579545 Hz). In H40 mode (320 pixel width), each pixel in the
    // active area is 8 cycles long[1]. In H32 mode (256 pixel width), each
    // pixel is 10 cycles long.
    //
    // In both modes, a full line is 3420 cycles long. Because 3420 % 15 == 0,
    // there is no phase offset between lines. With 262 lines per field[2],
    // there is also no phase offset between fields. When interlacing, there
    // are 262.5 lines per field. (3420 / 2) % 15 == 0, so there is no phase
    // offset between fields when interlacing, either.
    //
    // [1] Note that in the blanking area, some pixels in H40 mode are
    // different sizes. We can ignore that for our purposes.
    // [2] There are some half lines in the blanking area which add to full
    // lines.
    //
    // See: http://gendev.spritesmind.net/forum/viewtopic.php?f=2&t=3221

    float periods_per_active_line;
    // TODO handle if upscaled
    if (original_size.x > 300) { // H40
        periods_per_active_line = 8.0 / 15.0 * original_size.x;
    } else { // H32
        periods_per_active_line = 10.0 / 15.0 * original_size.x;
    }
    return chroma_info(periods_per_active_line, 228.0, 0.0);
}

chroma_info chroma_info_snes(vec4 original_size, uint frame_count) {
    // The SNES master clock (MCLK) is 21477270 Hz (6 * 3579545 Hz). When
    // outputting 256 pixels, there are 4 cycles per pixel[1].
    //
    // A full line is usually 1364 cycles long. A line is 227.33... chroma
    // periods long.
    //
    // A field is 262 lines. So a field is 1364 / 6 * 262 = 59561.33...
    // chroma periods long. Every other field has one line that is 1360 cycles
    // long, so every other field is 1364 / 6 * 261 + 1360 / 6 = 59560.66...
    // chroma periods long. It may be possible for a game to disable this, but
    // it seems unlikely that it happened commonly.
    //
    // [1] Note that two pixels (outside the active area) are actually 6 cycles
    // long.
    //
    // See: https://wiki.superfamicom.org/timing

    // Default to 512 px mode.
    float periods_per_active_line = 2.0 / 6.0 * original_size.x;
    if (original_size.x < 300) { // 256 px mode
        periods_per_active_line = 4.0 / 6.0 * original_size.x;
    }
    return chroma_info(periods_per_active_line, 682.0 / 3.0, 1.0 / 3.0);
}

chroma_info chroma_info_saturn(vec4 original_size) {
    // The Saturn has 427 dots per line in 320px mode, and 455 dots per line in
    // 352px mode. The dot clock in 352px mode is 2 * subcarrier, while the dot
    // clock in 320px mode is created with a 1708/1820 divider (i.e. 61/65).
    // 640px and 704px modes double the dot clock to 61/65 * 4 * subcarrier and
    // 4 * subcarrier, respectively.
    //
    // Therefore, there are 227.5 chroma periods per line.
    //
    // There are 263 lines per field in progressive mode, so the phase flips on
    // each field.
    //
    // It's possible that, in some cases, the higher clock rate may be used
    // 320 pixels per line. In this case the Saturn may add extra border
    // padding. If it is possible, it's unclear what emulators do in this case.
    //
    // See:
    // https://wiki.console5.com/wiki/315-5746
    // https://web.archive.org/web/20151218032848/http://dreamjam.co.uk/emuviews/sat/sattech.txt
    // https://junkerhq.net/xrgb/index.php?title=Optimal_timings

    // These bounds should work for most Saturn cores.
    // Default to 704 px mode.
    float periods_per_active_line = original_size.x / 4.0;
    if (original_size.x < 340) { // 320px mode
        periods_per_active_line = 65.0 / (2.0 * 61.0) * original_size.x;
    } else if (original_size.x < 400) { // 352px mode
        periods_per_active_line = original_size.x / 2.0;
    } else if (original_size.x < 680) { // 640px mode
        periods_per_active_line = 65.0 / (4.0 * 61.0) * original_size.x;
    }
    return chroma_info(periods_per_active_line, 227.5, 0.5);
}

chroma_info chroma_info_psx(vec4 original_size) {
    // The PSX timings are not entirely clear, and different hardware revisions
    // derived the clocks differently. What follows is some conjecture and
    // estimation that may result in something close enough.
    //
    // The earlier PSX seemed to have a video clock at 15x the subcarrier
    // frequency. To achieve 227.5 subcarrier periods per line, a line would be
    // 3412.5 cycles long. This is very close to the 3413 cycles listed on the
    // page below, which seems to have some uncertainty itself. If the correct
    // value was 3413, we'd have 227.5333... chroma periods per line, which
    // would complicate things significantly. For now, let's assume the correct
    // value is 3412.5 and we have the standard NTSC line length.
    //
    // There are 263 lines per frame, so we flip the phase each frame. This
    // might give us a hint regarding the line length as well, because if the
    // line length was 227.5333..., there would be no point using 263 lines
    // instead of 262. Either way, you'd get around a quarter period offset per
    // frame instead of the standard half a period.
    //
    // Each mode uses a different divider for the dot clock:
    //
    //     256 px: /10
    //     320 px: /8
    //     368 px: /7
    //     512 px: /5
    //     640 px: /4
    //
    // 256, 320, 512, and 640 modes all simplify to 512 / 3 periods per active
    // line. The 368 mode is slightly different.
    //
    // See: https://psx-spx.consoledev.net/dev/ps1/gpu/timings/

    // These bounds should work with SwanStation and Beetle PSX, at least.
    // It's not clear if other cores might add more padding.
    // Default to 640 px mode for high resolutions.
    float periods_per_active_line = original_size.x * 4.0 / 15.0;
    if (original_size.x <= 280.0) {
        periods_per_active_line = original_size.x * 10.0 / 15.0;
    } else if (original_size.x <= 350.0) {
        periods_per_active_line = original_size.x * 8.0 / 15.0;
    } else if (original_size.x <= 400.0) {
        periods_per_active_line = original_size.x * 7.0 / 15.0;
    } else if (original_size.x <= 560.0) {
        periods_per_active_line = original_size.x * 5.0 / 15.0;
    }
    return chroma_info(periods_per_active_line, 227.5, 0.5);
}

chroma_info chroma_info_n64() {
    // The N64 is mostly standard NTSC timings. There are 773.5 pixels per
    // line. There are 227.5 chroma periods per line. The progressive scan
    // output is 263 lines, which flips the phase every field.
    //
    // See: https://n64brew.dev/wiki/Video_Interface

    // We will assume that the input is the full 640 pixel width. The N64
    // framebuffer is always 640 pixels. This should be fine for upscaled
    // content as well, as long as the full framebuffer is used.
    float periods_per_active_line = (640 * 227.5) / 773.5;
    return chroma_info(periods_per_active_line, 227.5, 0.5);
}

chroma_info get_chroma_info(float system, vec4 original_size, uint frame_count) {
    if (system < 0.5) {
        // Standard
        return chroma_info_standard();
    } else if (system < 1.5) {
        // NES
        return chroma_info_nes(original_size, frame_count);
    } else if (system < 2.5) {
        // SMS
        return chroma_info_sms(original_size);
    } else if (system < 3.5) {
        // PCE/TG16
        return chroma_info_pce(original_size);
    } else if (system < 4.5) {
        // Genesis/MD
        return chroma_info_genesis(original_size);
    } else if (system < 5.5) {
        // SNES
        return chroma_info_snes(original_size, frame_count);
    } else if (system < 6.5) {
        // Saturn
        return chroma_info_saturn(original_size);
    } else if (system < 7.5) {
        // PS1
        return chroma_info_psx(original_size);
    } else if (system < 8.5) {
        // N64
        return chroma_info_n64();
    }
    // Default to standard
    return chroma_info_standard();
}

bool is_interlaced(float interlacing_type, vec4 source_size) {
    return interlacing_type < 1.5 && source_size.y > 300.0;
}

float chroma_carrier(float t, float line, uint frame_count, chroma_info info) {
    float position_offset = fract(info.periods_per_active_line * t);
    float line_offset = fract(info.periods_per_line) * line;
    float frame_offset = info.offset_per_frame * float(frame_count % 2u);
    return 2.0 * PI * (position_offset + line_offset + frame_offset);
}

float chroma_carrier_interlaced(float t, float line, uint frame_count, chroma_info info, float odd_field_first) {
    float position_offset = fract(info.periods_per_active_line * t);
    uint y = uint(line);
    uint is_even_field = y & 1u;
    float field_line = float(y >> 1u);
    uint phase_shift = (odd_field_first > 0.5) ? 1u : 0u;
    uint effective_frame = frame_count + phase_shift;

    // Computing interlaced_frame as this way evaluates the 30 Hz frame index
    // at which that specific field was rendered, ensuring the carrier phase
    // of the stale field remains invariant until the core updates it.
    uint interlaced_frame = ((effective_frame + 4u - is_even_field) >> 1u) & 1u;

    // Standard 525-line NTSC frames consist of an odd field followed by an even
    // field 263 lines later, followed by the next odd field 262 lines later
    // (263 + 262 = 525 lines per 2-field frame).
    // The subcarrier accumulates fract(periods_per_line) phase per raster line.
    // For all supported systems, this fraction has a denominator dividing 6:
    //   - Standard / PS1 / Saturn / N64: fract = 1/2 (denominator 2)
    //   - SNES: fract = 1/3 (denominator 3)
    //   - Genesis: fract = 0 (denominator 1)
    // Since 6 raster lines always contain an exact integer number of subcarrier
    // cycles (6 * 1/2 = 3, 6 * 1/3 = 2), line offsets can be evaluated modulo 6
    // without changing the phase:
    //   - Even field offset: 263 lines = 5 mod 6
    //   - Next frame offset: 525 lines = 3 mod 6
    // This produces the correct 4-field phase patterns at field start:
    //   - SNES:     [0, 2/3, 0, 2/3]
    //   - Standard: [0, 1/2, 1/2, 0]
    //   - Genesis:  [0, 0, 0, 0]
    uint raw_offset = 3u * interlaced_frame + 5u * is_even_field;
    uint field_line_offset = (raw_offset >= 6u) ? (raw_offset - 6u) : raw_offset;
    float line_offset = fract(info.periods_per_line) * (field_line + float(field_line_offset));
    return 2.0 * PI * (position_offset + line_offset);
}

float chroma_carrier(float t, float line, uint frame_count, chroma_info info, bool interlaced, float odd_field_first) {
    if (interlaced) {
        return chroma_carrier_interlaced(t, line, frame_count, info, odd_field_first);
    } else {
        return chroma_carrier(t, line, frame_count, info);
    }
}
