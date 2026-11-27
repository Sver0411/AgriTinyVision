/* C99 host reference for an EQ01 binary head and a batch of packed int8 rows. */
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float read_f32le(const unsigned char *p) {
    uint32_t bits = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                    ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    float value;
    memcpy(&value, &bits, sizeof bits);
    return value;
}

static int unpack_symmetric(unsigned char byte) {
    return byte < 128 ? (int)byte : (int)byte - 256;
}

static int fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    return EXIT_FAILURE;
}

int main(int argc, char **argv) {
    if (argc != 4) return fail("usage: packed_head HEAD.bin ROWS.i8 ROW_COUNT");
    if (CHAR_BIT != 8 || sizeof(float) != 4 || FLT_RADIX != 2 ||
        FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128) return fail("IEEE binary32 required");
    char *end;
    errno = 0;
    unsigned long rows = strtoul(argv[3], &end, 10);
    if (errno || argv[3][0] < '0' || argv[3][0] > '9' || *end || !rows) return fail("invalid row count");
    FILE *head = fopen(argv[1], "rb");
    if (!head) return fail("cannot open head");
    unsigned char header[14];
    if (fread(header, 1, sizeof header, head) != sizeof header || memcmp(header, "EQ01", 4))
        return fail("invalid EQ01 header");
    size_t d = (size_t)header[4] | ((size_t)header[5] << 8);
    float scale = read_f32le(header + 6), bias = read_f32le(header + 10);
    if (!d || !isfinite(scale) || !(scale > 0.0f) || !isfinite(bias)) return fail("invalid head parameters");
    unsigned char *codes = malloc(d), *row = malloc(d);
    if (!codes || !row) return fail("insufficient memory");
    if (fread(codes, 1, d, head) != d || fgetc(head) != EOF || ferror(head)) return fail("invalid head length");
    fclose(head);
    if (memchr(codes, 128, d)) return fail("head contains code outside [-127,127]");
    FILE *input = fopen(argv[2], "rb");
    if (!input) return fail("cannot open packed rows");
    for (unsigned long r = 0; r < rows; ++r) {
        if (fread(row, 1, d, input) != d) return fail("truncated rows");
        if (memchr(row, 128, d)) return fail("row contains code outside [-127,127]");
        int32_t dot = 0, sum_squares = 0;
        for (size_t i = 0; i < d; ++i) {
            int32_t x = unpack_symmetric(row[i]), w = unpack_symmetric(codes[i]);
            dot += x * w;
            sum_squares += x * x;
        }
        double norm = sum_squares ? sqrt((double)sum_squares) : 1.0;
        double score = (double)dot * (double)scale / norm + (double)bias;
        if (printf("%" PRId32 "\t%" PRId32 "\t%.17g\t%d\n", dot, sum_squares, score, score >= 0.0) < 0)
            return fail("output error");
    }
    if (fgetc(input) != EOF || ferror(input)) return fail("unexpected trailing row data");
    fclose(input);
    free(codes); free(row);
    return fflush(stdout) == 0 ? EXIT_SUCCESS : fail("output flush error");
}
