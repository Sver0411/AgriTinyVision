/*
 * Turnkey ESP32-S3 benchmark for BASA on-device inference & online streaming gate.
 * Compatible with ESP-IDF v5.4+ (same toolchain as AgriTinyVision v0.2 on ESP32-S3 N16R8).
 *
 * Measures:
 *   1. Bit-exact parity of EQ01 1,294-byte INT8 classification head against Python reference.
 *   2. Real microsecond latency (P50 / P95 / P99) for:
 *        - FP32->INT8 symmetric activation quantization (1,280-D)
 *        - EQ01 INT8 head inference (1,280-D dot + L2 norm + bias)
 *        - Causal 64-frame rolling percentile streaming gate step
 *   3. Exact internal SRAM & PSRAM footprint before/after state allocation.
 *   4. Prints a single copy-pasteable BENCH_JSON line for direct ingestion into the paper.
 */

#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef ESP_PLATFORM
#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_idf_version.h"
#include "esp_private/esp_clk.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#else
#include <time.h>
static int64_t esp_timer_get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000LL + (int64_t)(ts.tv_nsec / 1000L);
}
#endif

#include "basa_fixture.h"

#define BENCH_ITERS 1000
#define RING_CAP 64

typedef struct {
    uint16_t dim;
    float scale;
    float bias;
    int8_t weights[BASA_DIM];
} eq01_head_t;

typedef struct {
    double ring[RING_CAP];
    int count;
    int cursor;
    int64_t grant;
    int64_t spent;
    double mean_jpeg;
} stream_gate_t;

static float read_f32le(const uint8_t *p) {
    uint32_t bits = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                    ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    float v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

static bool eq01_parse(const uint8_t *blob, size_t len, eq01_head_t *out) {
    if (len != 14 + BASA_DIM || memcmp(blob, "EQ01", 4) != 0) return false;
    uint16_t d = (uint16_t)blob[4] | ((uint16_t)blob[5] << 8);
    float scale = read_f32le(blob + 6);
    float bias = read_f32le(blob + 10);
    if (d != BASA_DIM || !isfinite(scale) || !(scale > 0.0f) || !isfinite(bias)) return false;
    for (size_t i = 0; i < BASA_DIM; ++i) {
        int8_t w = (int8_t)blob[14 + i];
        if (w == -128) return false;
        out->weights[i] = w;
    }
    out->dim = d;
    out->scale = scale;
    out->bias = bias;
    return true;
}

static double eq01_infer(const eq01_head_t *h, const int8_t *x, int32_t *dot_out, int32_t *sq_out) {
    int32_t dot = 0;
    int32_t sum_sq = 0;
    for (size_t i = 0; i < BASA_DIM; ++i) {
        int32_t xv = (int32_t)x[i];
        int32_t wv = (int32_t)h->weights[i];
        dot += xv * wv;
        sum_sq += xv * xv;
    }
    if (dot_out) *dot_out = dot;
    if (sq_out) *sq_out = sum_sq;
    double norm = sum_sq ? sqrt((double)sum_sq) : 1.0;
    return (double)dot * (double)h->scale / norm + (double)h->bias;
}

static float quantize_activation(const float *x, int8_t *q_out) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < BASA_DIM; ++i) {
        float a = fabsf(x[i]);
        if (a > max_abs) max_abs = a;
    }
    float scale = max_abs > 0.0f ? (max_abs / 127.0f) : 1.0f;
    float inv = 1.0f / scale;
    for (size_t i = 0; i < BASA_DIM; ++i) {
        float scaled = nearbyintf(x[i] * inv);
        if (scaled > 127.0f) scaled = 127.0f;
        if (scaled < -127.0f) scaled = -127.0f;
        q_out[i] = (int8_t)scaled;
    }
    return scale;
}

static int stream_gate_step(stream_gate_t *g, double score, int32_t jpeg_bytes, int t, int n_total) {
    int win = g->count < RING_CAP ? g->count : RING_CAP;
    int less = 0, eq = 0;
    for (int i = 0; i < win; ++i) {
        if (g->ring[i] < score) less++;
        else if (g->ring[i] == score) eq++;
    }
    double prio = win == 0 ? 0.5 : (less + 0.5 * eq) / (double)win;
    g->ring[g->cursor] = score;
    g->cursor = (g->cursor + 1) % RING_CAP;
    if (g->count < RING_CAP) g->count++;

    double frac = (double)(g->grant - g->spent) / ((double)(n_total - t) * g->mean_jpeg);
    if (frac > 1.0) frac = 1.0;
    int ask = (win >= 16 && prio >= 1.0 - frac && jpeg_bytes <= g->grant - g->spent) ? 1 : 0;
    if (ask) g->spent += jpeg_bytes;
    g->mean_jpeg += 0.05 * ((double)jpeg_bytes - g->mean_jpeg);
    return ask;
}

static int cmp_i64(const void *a, const void *b) {
    int64_t va = *(const int64_t *)a, vb = *(const int64_t *)b;
    return (va > vb) - (va < vb);
}

static void stats_us(int64_t *samples, int n, int64_t *p50, int64_t *p95, int64_t *p99) {
    qsort(samples, (size_t)n, sizeof(int64_t), cmp_i64);
    *p50 = samples[n / 2];
    *p95 = samples[(n * 95) / 100];
    *p99 = samples[(n * 99) / 100];
}

static void run_benchmark(void) {
#ifdef ESP_PLATFORM
    size_t heap_before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t psram_before = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
#else
    size_t heap_before = 0, psram_before = 0;
#endif

    eq01_head_t *head = (eq01_head_t *)malloc(sizeof(eq01_head_t));
    stream_gate_t *gate = (stream_gate_t *)calloc(1, sizeof(stream_gate_t));
    int8_t *q_buf = (int8_t *)malloc(BASA_DIM);
    float *fp_buf = (float *)malloc(BASA_DIM * sizeof(float));
    int64_t *t_us = (int64_t *)malloc(BENCH_ITERS * sizeof(int64_t));

#ifdef ESP_PLATFORM
    size_t heap_after = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t psram_after = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
#else
    size_t heap_after = 0, psram_after = 0;
#endif

    /* 1. Bit-exact fixture parity check */
    bool parsed = eq01_parse(BASA_HEAD_BLOB, sizeof(BASA_HEAD_BLOB), head);
    int mismatches = parsed ? 0 : BASA_FIXTURE_ROWS;
    double max_abs_err = 0.0;
    if (parsed) {
        for (int r = 0; r < BASA_FIXTURE_ROWS; ++r) {
            int32_t dot = 0, sq = 0;
            double score = eq01_infer(head, BASA_FIXTURE_X[r], &dot, &sq);
            int pred = score >= 0.0 ? 1 : 0;
            double err = fabs(score - BASA_FIXTURE_SCORE[r]);
            if (err > max_abs_err) max_abs_err = err;
            if (dot != BASA_FIXTURE_DOT[r] || sq != BASA_FIXTURE_SQ[r] ||
                pred != BASA_FIXTURE_PRED[r] || err > 1e-11) {
                mismatches++;
            }
        }
    }

    /* 2. Latency: EQ01 parse */
    for (int i = 0; i < BENCH_ITERS; ++i) {
        int64_t t0 = esp_timer_get_time();
        parsed = eq01_parse(BASA_HEAD_BLOB, sizeof(BASA_HEAD_BLOB), head);
        t_us[i] = esp_timer_get_time() - t0;
    }
    int64_t parse_p50, parse_p95, parse_p99;
    stats_us(t_us, BENCH_ITERS, &parse_p50, &parse_p95, &parse_p99);

    /* 3. Latency: EQ01 INT8 head inference */
    volatile double sink = 0.0;
    for (int i = 0; i < BENCH_ITERS; ++i) {
        const int8_t *row = BASA_FIXTURE_X[i % BASA_FIXTURE_ROWS];
        int64_t t0 = esp_timer_get_time();
        double s = eq01_infer(head, row, NULL, NULL);
        t_us[i] = esp_timer_get_time() - t0;
        sink += s;
    }
    int64_t infer_p50, infer_p95, infer_p99;
    stats_us(t_us, BENCH_ITERS, &infer_p50, &infer_p95, &infer_p99);

    /* 4. Latency: FP32->INT8 symmetric activation quantization */
    for (size_t j = 0; j < BASA_DIM; ++j) {
        fp_buf[j] = (float)BASA_FIXTURE_X[0][j] * 0.03125f;
    }
    volatile float scale_sink = 0.0f;
    for (int i = 0; i < BENCH_ITERS; ++i) {
        int64_t t0 = esp_timer_get_time();
        float sc = quantize_activation(fp_buf, q_buf);
        t_us[i] = esp_timer_get_time() - t0;
        scale_sink += sc;
    }
    int64_t quant_p50, quant_p95, quant_p99;
    stats_us(t_us, BENCH_ITERS, &quant_p50, &quant_p95, &quant_p99);

    /* 5. Latency: Causal 64-frame rolling percentile streaming gate */
    gate->grant = 1353LL * 1700LL;
    gate->spent = 260000LL;
    gate->mean_jpeg = 6800.0;
    volatile int ask_sink = 0;
    for (int i = 0; i < BENCH_ITERS; ++i) {
        double urgency = -fabs(BASA_FIXTURE_SCORE[i % BASA_FIXTURE_ROWS]);
        int64_t t0 = esp_timer_get_time();
        int a = stream_gate_step(gate, urgency, 6750, i, BENCH_ITERS + 353);
        t_us[i] = esp_timer_get_time() - t0;
        ask_sink += a;
    }
    int64_t gate_p50, gate_p95, gate_p99;
    stats_us(t_us, BENCH_ITERS, &gate_p50, &gate_p95, &gate_p99);

#ifdef ESP_PLATFORM
    esp_chip_info_t info;
    esp_chip_info(&info);
    int cpu_mhz = esp_clk_cpu_freq() / 1000000;
    const char *idf_ver = esp_get_idf_version();
    int chip_model = (int)info.model;
    int cores = info.cores;
#else
    int cpu_mhz = 0, chip_model = 0, cores = 0;
    const char *idf_ver = "host-c99";
#endif

    printf("\nBENCH_JSON: {"
           "\"idf_version\":\"%s\",\"chip_model\":%d,\"cores\":%d,\"cpu_mhz\":%d,"
           "\"dim\":%d,\"head_wire_bytes\":%u,\"fixture_rows\":%d,\"mismatches\":%d,"
           "\"max_abs_err\":%.3e,\"iters\":%d,"
           "\"eq01_head_struct_bytes\":%u,\"stream_gate_struct_bytes\":%u,"
           "\"internal_heap_before\":%u,\"internal_heap_after\":%u,"
           "\"psram_before\":%u,\"psram_after\":%u,"
           "\"parse_us\":[%" PRId64 ",%" PRId64 ",%" PRId64 "],"
           "\"infer_us\":[%" PRId64 ",%" PRId64 ",%" PRId64 "],"
           "\"quant_us\":[%" PRId64 ",%" PRId64 ",%" PRId64 "],"
           "\"gate_us\":[%" PRId64 ",%" PRId64 ",%" PRId64 "]}\n",
           idf_ver, chip_model, cores, cpu_mhz, BASA_DIM,
           (unsigned)sizeof(BASA_HEAD_BLOB), BASA_FIXTURE_ROWS, mismatches,
           max_abs_err, BENCH_ITERS,
           (unsigned)sizeof(eq01_head_t), (unsigned)sizeof(stream_gate_t),
           (unsigned)heap_before, (unsigned)heap_after,
           (unsigned)psram_before, (unsigned)psram_after,
           parse_p50, parse_p95, parse_p99,
           infer_p50, infer_p95, infer_p99,
           quant_p50, quant_p95, quant_p99,
           gate_p50, gate_p95, gate_p99);

    (void)sink; (void)scale_sink; (void)ask_sink;
    free(t_us); free(fp_buf); free(q_buf); free(gate); free(head);
}

#ifdef ESP_PLATFORM
void app_main(void) {
    vTaskDelay(pdMS_TO_TICKS(1000));
    run_benchmark();
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}
#else
int main(void) {
    run_benchmark();
    return 0;
}
#endif
