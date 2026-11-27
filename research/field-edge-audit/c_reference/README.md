# Portable signed INT8 classification head

`packed_head.c` reads a released EQ01 binary head and a sequence of signed INT8 feature rows. The implementation uses the C99 standard library and `libm`. An IEEE binary32 float representation and eight-bit bytes are required. The decoder reads little-endian fields explicitly. The first four head bytes are `EQ01`, followed by a two-byte unsigned dimension, a four-byte scale, a four-byte bias and exactly one signed byte per weight. Input vectors are consecutive rows of the same dimension with symmetric codes from −127 through 127. Each row emits four tab-separated fields in order: integer dot product, integer squared norm, signed final score, and binary prediction.

```sh
cc -std=c99 -O2 -Wall -Wextra -Werror -pedantic c_reference/packed_head.c -o /tmp/packed_head -lm
/tmp/packed_head /path/to/local_int8.bin /path/to/rows.i8 ROW_COUNT
python c_reference/verify.py
```

The Python audit independently quantizes the saved 1,280-dimensional activations, supplies the exact packed head and vector bytes, and compares all integer accumulations against a 64-bit reference. It compares the final score and binary class against the exported experimental predictions. The signed score equals `dot * weight_scale / sqrt(sum_squares) + bias`, with a denominator of one for a zero vector. A zero score maps to class one. Each summand has magnitude at most 16,129. At 1,280 features, both the absolute dot product and squared norm are bounded by 20,645,120. The largest dimension representable by the two-byte header gives a bound of 1,057,014,015, within signed 32-bit range. The audit includes saturated and all-zero vectors, reversed encoded dimension bytes, malformed headers, invalid codes, truncation and trailing bytes.

`results/c99_head_audit.json` records input hashes, decision counts by budget, floating-point differences, observed accumulator magnitudes and compiler information. The test covers final-head execution on the host using frozen encoder outputs. Feature extraction and on-device resource measurements require separate validation.
