// SPDX-License-Identifier: MIT

// Antiderivative of k(u) = 1 - 3*u^2 + 2*|u|^3 on [-1, 1].
// F(1) - F(-1) = 1, so the horizontal spot has unit total energy.
vec3 cubic_primitive(vec3 u) {
    return u * (1.0 + u * u * (0.5 * abs(u) - 1.0));
}

// Convolve s/w * k((x - t)/w) with a constant texel over [x0, x1].
// dx0 = x - x0 and dx1 = x - x1 are the signed distances from the evaluation point.
// All distances are in scanline heights.
vec3 spot_integral(vec3 sample_value, vec3 width, float dx0, float dx1) {
    vec3 w_rcp = 1.0 / width;
    vec3 u0 = clamp(w_rcp * dx0, -1.0, 1.0);
    vec3 u1 = clamp(w_rcp * dx1, -1.0, 1.0);
    return sample_value * (cubic_primitive(u0) - cubic_primitive(u1));
}

uvec3 render_scanline_horizontal(vec2 tex_coord, float delta) {
    vec3 energy = vec3(0.0);
    vec3 moment = vec3(0.0);

    float min_width = params.MinSpotSize * params.MaxSpotSize;
    float width_range = params.MaxSpotSize - min_width;
    float radius = params.MaxSpotSize / delta;
    float tex_x = tex_coord.x * params.SourceSize.x;
    float energy_scale = 1.0;

#ifndef FORCE_ZOH
    if (min_width < 1.5 * delta) {
#endif
        // Zero-order hold integration: used for un-filtered inputs and when
        // the narrowest spot covers too few samples for point quadrature.
        int start_i = int(floor(tex_x - radius));
        int stop_i  = int(floor(tex_x + radius));

        for (int i = start_i; i <= stop_i; ++i) {
            float sample_x = (float(i) + 0.5) * params.SourceSize.z;
            vec3 sample_value = textureLod(Source, vec2(sample_x, tex_coord.y), 0.0).rgb;
            float dx0 = (tex_x - float(i)) * delta;
            float dx1 = dx0 - delta;
            vec3 piece_width = min_width + sqrt(sample_value) * width_range;
            vec3 piece_energy = spot_integral(sample_value, piece_width, dx0, dx1);
            energy += piece_energy;
            moment += piece_width * piece_energy;
        }
#ifndef FORCE_ZOH
    } else {
        int start_i = int(floor(tex_x - radius + 0.5));
        int stop_i  = int(floor(tex_x + radius - 0.5));

        for (int i = start_i; i <= stop_i; ++i) {
            float sample_x = (float(i) + 0.5) * params.SourceSize.z;
            vec3 sample_value = textureLod(Source, vec2(sample_x, tex_coord.y), 0.0).rgb;
            float abs_dx = abs(tex_x - (float(i) + 0.5)) * delta;
            vec3 piece_width = min_width + sqrt(sample_value) * width_range;
            vec3 piece_w_rcp = 1.0 / piece_width;
            vec3 u = min(abs_dx * piece_w_rcp, vec3(1.0));
            vec3 weight = sample_value * ((u * u) * (2.0 * u - 3.0) + 1.0);
            energy += weight * piece_w_rcp;
            moment += weight;
        }
        energy_scale = delta;
    }
#endif

    vec3 width = min(mix(vec3(min_width), moment / energy, notEqual(energy, vec3(0.0))), vec3(params.MaxSpotSize));
    vec3 width_rcp = 1.0 / width;

    // Pre-multiply energy with MaxSpotSize * width_rcp to save vec3 multiplications
    // per viewport pixel in the vertical scanline pass.
    vec3 scaled_energy = energy * (energy_scale * params.MaxSpotSize * width_rcp);

    // Pack both scaled_energy and width_rcp into one texture. This avoids an extra
    // pass (since we don't have multiple render target support) and allows
    // fewer texture samples in the next pass.
    return uvec3(
        packHalf2x16(vec2(scaled_energy.r, width_rcp.r)),
        packHalf2x16(vec2(scaled_energy.g, width_rcp.g)),
        packHalf2x16(vec2(scaled_energy.b, width_rcp.b))
    );
}
