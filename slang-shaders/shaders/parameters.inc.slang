#pragma parameter InputSpace "Input (0: System gamma, 1: 2.2 gamma, 2: sRGB)" 0.0 0.0 2.0 1.0
#pragma parameter YIQ "Composite (0: RGB, 1: YIQ/composite)" 0.0 0.0 1.0 1.0
#pragma parameter Cutoff "RGB/Y bandwidth (MHz)" 5.0 0.6 6.0 0.2
#pragma parameter ICutoff "I bandwidth (MHz)" 0.6 0.6 6.0 0.2
#pragma parameter QCutoff "Q bandwidth (MHz)" 0.6 0.6 6.0 0.2
#pragma parameter MaxSpotSize "Maximum spot size (proportion of scanline)" 0.90 0.6 1.0 0.05
#pragma parameter MinSpotSize "Minimum spot size (proportion of maximum)" 0.4 0.3 1.0 0.1
#pragma parameter OddFieldFirst "Interlacing (phase)" 0.0 0.0 1.0 1.0
#pragma parameter OverscanHorizontal "Horizontal overscan (proportion to crop)" 0.0 0.0 0.1 0.01
#pragma parameter OverscanVertical "Vertical overscan (proportion to crop)" 0.0 0.0 0.1 0.01
#pragma parameter GlowSigma "Glow width (proportion of screen height)" 0.03 0.01 0.05 0.01
#pragma parameter GlowAmount "Glow amount (mix ratio)" 0.10 0.0 1.0 0.05
