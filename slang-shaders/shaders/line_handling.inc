// SPDX-License-Identifier: MIT

struct lines_t {
    // Source texel row indices (may be out of bounds at the top/bottom edges).
    int upper_sample_y;
    int lower_sample_y;
    // Distance units are *scanlines heights*.
    float upper_distance_y;
    float lower_distance_y;
};

// Get the two closest lines to the given position.
lines_t get_closest_lines(float v, vec4 sourceSize) {
    float y = v * sourceSize.y;
    int upper_sample_y = int(round(y));
    int lower_sample_y = upper_sample_y - 1;
    float upper_distance_y = (float(upper_sample_y) + 0.5) - y;
    float lower_distance_y = upper_distance_y - 1.0;

    return lines_t(upper_sample_y, lower_sample_y, upper_distance_y, lower_distance_y);
}

// Get the two closest lines from the *odd* field (top field: rows 0, 2, 4..., no
// half-scanline offset at start) to the given position.
lines_t get_odd_lines(float v, vec4 sourceSize) {
    float y = v * sourceSize.y;
    int upper_sample_y = int(ceil(0.5 * y - 0.25)) * 2;
    int lower_sample_y = upper_sample_y - 2;
    float upper_distance_y = 0.5 * ((float(upper_sample_y) + 0.5) - y);
    float lower_distance_y = upper_distance_y - 1.0;

    return lines_t(upper_sample_y, lower_sample_y, upper_distance_y, lower_distance_y);
}

// Get the two closest lines from the *even* field (bottom field: rows 1, 3, 5...,
// displaced half a field scanline below the odd field) to the given position.
lines_t get_even_lines(float v, vec4 sourceSize) {
    float y = v * sourceSize.y;
    int upper_sample_y = int(ceil(0.5 * y + 0.25)) * 2 - 1;
    int lower_sample_y = upper_sample_y - 2;
    float upper_distance_y = 0.5 * ((float(upper_sample_y) + 0.5) - y);
    float lower_distance_y = upper_distance_y - 1.0;

    return lines_t(upper_sample_y, lower_sample_y, upper_distance_y, lower_distance_y);
}

// Get the two closest lines as if the source was line-doubled. Note that this
// means that the sample coordinates for the upper and lower lines may be the
// same, even though the distances are different.
lines_t get_doubled_lines(float v, vec4 sourceSize) {
    float y = v * sourceSize.y;
    float upper_distance_y = ceil(2.0 * y - 0.5) + 0.5 - 2.0 * y;
    float lower_distance_y = upper_distance_y - 1.0;
    int upper_sample_y = int(ceil(y + 0.25)) - 1;
    int lower_sample_y = int(ceil(y - 0.25)) - 1;

    return lines_t(upper_sample_y, lower_sample_y, upper_distance_y, lower_distance_y);
}
