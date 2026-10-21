// SPDX-License-Identifier: MIT

#pragma parameter LumaBandwidth "Luma bandwidth (MHz)" 4.0 0.6 6.0 0.2
#pragma parameter ChromaBandwidth "Chroma bandwidth (MHz, each channel)" 1.2 0.6 6.0 0.2
#pragma parameter FilterType "Composite Y/C filter (0: notch, 1: comb, 2: adaptive comb)" 2.0 0.0 2.0 1.0
#pragma parameter System "NTSC system" 0.0 0.0 8.0 1.0
// System types: (0: Standard, 1: NES, 2: SMS, 3: TG16, 4: Genesis, 5: SNES, 6: Saturn, 7: PS1, 8: N64)
#pragma parameter InterlacingType "Interlacing type (0: bob, 1: weave, 2: VGA, 3: off)" 0.0 0.0 3.0 1.0
#pragma parameter OddFieldFirst "Interlacing phase (0: odd field first, 1: even field first)" 0.0 0.0 1.0 1.0
