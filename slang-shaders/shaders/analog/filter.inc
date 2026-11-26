// SPDX-License-Identifier: MIT

#define PI 3.141592653589793

vec3 low_pass(vec3 L, vec2 pos, vec4 source_size, sampler2D Source) {
    float L_max = max(L.r, max(L.g, L.b));
    vec3 L_rcp = 1.0 / L;

    vec3 filtered = vec3(0.0);
    int pix_y = int(floor(pos.y * source_size.y));
    float t = pos.x;

    // Bound the start and end pixel so we don't fetch outside our texture
    // (which is undefined behavior). There's no need to fetch there anyway,
    // it would just be zeros.
    int start = max(int(floor(source_size.x * (pos.x - L_max))), 0);
    int end = min(int(floor(source_size.x * (pos.x + L_max))), int(source_size.x) - 1);

    // Set up the first common term for the left. The right side of the integral
    // for iteration i is the left side of the integral for iteration i + 1, so
    // we can avoid computing the term twice.
    float t0 = start * source_size.z;
    vec3 u0 = clamp((t - t0) * L_rcp, -1.0, 1.0);
    vec3 left = PI * u0 + sin(PI * u0);
    for (int pix_x = start; pix_x <= end; pix_x++) {
        // Integral of s / L * (0.5 + 0.5 * cos(PI * (t - t_x) / L)) dt_x over t0 to t1
        vec3 s = texelFetch(Source, ivec2(pix_x, pix_y), 0).rgb;
        float t1 = (pix_x + 1) * source_size.z;
        vec3 u1 = clamp((t - t1) * L_rcp, -1.0, 1.0);
        vec3 right = PI * u1 + sin(PI * u1);
        filtered += s * (left - right);
        left = right;
    }
    // We factored a common term of 1 / (2 * PI) out of the loop, so apply
    // that now.
    return filtered / (2.0 * PI);
}
