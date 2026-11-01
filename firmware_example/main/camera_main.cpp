#include "camera_core.hpp"
#include "camera_model_config.h"
#include "dl_model_base.hpp"
#include "driver/uart.h"
#include "driver/usb_serial_jtag.h"
#include "esp_app_desc.h"
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_image_format.h"
#include "esp_log.h"
#include "esp_ota_ops.h"
#include "esp_random.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "cJSON.h"
#include "mbedtls/sha256.h"
#include <cstdarg>
#include <cstdio>
#include <memory>
#include <new>
#include <vector>

static constexpr const char *FW = "0.2.0";
extern const uint8_t model_data[] asm("_binary_tinyvision_espdl_start");
extern const uint8_t model_end[] asm("_binary_tinyvision_espdl_end");
static char boot_id[17], tx_json[4096];
static std::unique_ptr<dl::Model> model;
static dl::TensorBase *input = nullptr, *output = nullptr;
static const char *model_error = "not_loaded";
static int64_t load_us = 0;
static camera::SequenceWindow sequences;
static bool session = false;
static uint8_t *rgb = nullptr;
static int discard_log(const char *, va_list) { return 0; }

static bool transport_init()
{
#if CONFIG_ATV_TRANSPORT_USB_JTAG
    if (usb_serial_jtag_is_driver_installed()) return true;
    // IDF's USJ ISR drops bytes when its ring is full. A complete single-frame
    // burst must fit even while the task is yielding; large malloc uses PSRAM.
    usb_serial_jtag_driver_config_t config = {4096, 96 * 1024};
    return usb_serial_jtag_driver_install(&config) == ESP_OK;
#else
    uart_config_t config = {};
    config.baud_rate = CONFIG_ATV_UART_BAUDRATE;
    config.data_bits = UART_DATA_8_BITS;
    config.parity = UART_PARITY_DISABLE;
    config.stop_bits = UART_STOP_BITS_1;
    config.flow_ctrl = UART_HW_FLOWCTRL_DISABLE;
    config.source_clk = UART_SCLK_DEFAULT;
    if (uart_param_config(UART_NUM_0, &config) != ESP_OK) return false;
    return uart_is_driver_installed(UART_NUM_0) || uart_driver_install(UART_NUM_0, 8192, 4096, 0, nullptr, 0) == ESP_OK;
#endif
}
static int read_bytes(uint8_t *data, size_t size)
{
#if CONFIG_ATV_TRANSPORT_USB_JTAG
    return usb_serial_jtag_read_bytes(data, size, pdMS_TO_TICKS(50));
#else
    return uart_read_bytes(UART_NUM_0, data, size, pdMS_TO_TICKS(50));
#endif
}
static bool write_bytes(const uint8_t *data, size_t size)
{
    size_t sent = 0;
    int64_t deadline = esp_timer_get_time() + 2000000;
    while (sent < size && esp_timer_get_time() < deadline) {
#if CONFIG_ATV_TRANSPORT_USB_JTAG
        int n = usb_serial_jtag_write_bytes(data + sent, size - sent, pdMS_TO_TICKS(100));
#else
        int n = uart_write_bytes(UART_NUM_0, data + sent, size - sent);
#endif
        if (n < 0) return false;
        sent += n;
    }
    return sent == size;
}
static bool send(camera::Type type, uint32_t seq, const uint8_t *data = nullptr, size_t size = 0)
{
    if (!seq || size > camera::CONTROL_BYTES) return false;
    uint8_t header[32];
    camera::header(header, type, seq, data, size);
    return write_bytes(header, sizeof(header)) && write_bytes(data, size);
}
static bool send_json(camera::Type type, uint32_t seq, cJSON *o)
{
    if (!o) return false;
    bool ok = cJSON_PrintPreallocated(o, tx_json, sizeof(tx_json), false);
    cJSON_Delete(o);
    return ok && send(type, seq, reinterpret_cast<const uint8_t *>(tx_json), std::strlen(tx_json));
}
static void text(cJSON *o, const char *key, const char *value) { cJSON_AddStringToObject(o, key, value); }
static void number(cJSON *o, const char *key, double value) { cJSON_AddNumberToObject(o, key, value); }
static void nullable(cJSON *o, const char *key, int64_t value)
{
    if (value < 0) cJSON_AddNullToObject(o, key); else number(o, key, value);
}
static void memory(cJSON *o)
{
    number(o, "free_internal", heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    number(o, "free_psram", heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    number(o, "min_free_internal", heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    number(o, "min_free_psram", heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
}
static void error(uint32_t seq, const char *code, const char *message)
{
    cJSON *o = cJSON_CreateObject();
    if (!o) return;
    text(o, "state", "UNKNOWN"); text(o, "code", code); text(o, "message", message); text(o, "boot_id", boot_id);
    send_json(camera::ERROR, seq, o);
}
static bool load_model()
{
    int64_t start = esp_timer_get_time();
    size_t bytes = model_end - model_data;
    if (bytes != CAMERA_MODEL_BYTES) { model_error = "model_size"; return false; }
    uint8_t digest[32];
    if (mbedtls_sha256(model_data, bytes, digest, 0) != 0) { model_error = "sha256_failed"; return false; }
    char hash[65];
    for (int i = 0; i < 32; ++i) std::snprintf(hash + i * 2, 3, "%02x", digest[i]);
    if (std::strcmp(hash, CAMERA_MODEL_SHA256)) { model_error = "model_hash"; return false; }
    rgb = static_cast<uint8_t *>(heap_caps_malloc(camera::FRAME_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (!rgb) { model_error = "rgb_psram_allocation"; return false; }
    // Same constructor as the fixture: creates context, loads graph, then allocates tensors.
    model.reset(new (std::nothrow) dl::Model(reinterpret_cast<const char *>(model_data),
                                          fbs::MODEL_LOCATION_IN_FLASH_RODATA));
    if (!model) { model_error = "model_allocation"; return false; }
    auto ins = model->get_inputs(), outs = model->get_outputs();
    if (ins.size() != 1 || outs.size() != 1) { model_error = "tensor_count"; return false; }
    input = ins.begin()->second; output = outs.begin()->second;
    if (!input || !output || !input->data || !output->data || input->shape != std::vector<int>({1, 160, 160, 3}) ||
        output->shape != std::vector<int>({1, 2}) || input->dtype != dl::DATA_TYPE_INT8 || output->dtype != dl::DATA_TYPE_INT8 ||
        int(input->exponent) != CAMERA_INPUT_EXPONENT || int(output->exponent) != CAMERA_OUTPUT_EXPONENT) {
        model_error = "tensor_contract"; return false;
    }
    load_us = esp_timer_get_time() - start;
    model_error = nullptr;
    return true;
}
static void hello(uint32_t seq)
{
    cJSON *o = cJSON_CreateObject();
    if (!o) return;
    text(o, "chip", "ESP32-S3"); text(o, "board", "ESP32-S3 N16R8"); text(o, "boot_id", boot_id);
    text(o, "firmware_version", FW); text(o, "model_version", CAMERA_MODEL_VERSION);
    text(o, "model_sha256", CAMERA_MODEL_SHA256); text(o, "idf_version", esp_get_idf_version());
    text(o, "deployment_sha256", CAMERA_DEPLOYMENT_SHA256);
    text(o, "esp_dl_version", "3.3.13");
#if CONFIG_ATV_TRANSPORT_USB_JTAG
    text(o, "transport", "USB Serial/JTAG");
    number(o, "rx_buffer_bytes", 96 * 1024);
#else
    text(o, "transport", "UART0 bridge");
    number(o, "rx_buffer_bytes", 8192);
#endif
    number(o, "uart_baudrate", CONFIG_ATV_UART_BAUDRATE);
    cJSON_AddBoolToObject(o, "model_loaded", model_error == nullptr);
    if (model_error) text(o, "model_error", model_error); else cJSON_AddNullToObject(o, "model_error");
    nullable(o, "model_load_us", model_error ? -1 : load_us); number(o, "model_bytes", CAMERA_MODEL_BYTES);
    text(o, "dtype", "int8");
    const int ishape[] = {1, 160, 160, 3}, oshape[] = {1, 2};
    cJSON_AddItemToObject(o, "input_shape", cJSON_CreateIntArray(ishape, 4));
    cJSON_AddItemToObject(o, "output_shape", cJSON_CreateIntArray(oshape, 2));
    number(o, "input_exponent", CAMERA_INPUT_EXPONENT); number(o, "output_exponent", CAMERA_OUTPUT_EXPONENT);
    number(o, "decision_threshold", CAMERA_THRESHOLD); number(o, "normal_upper_bound", CAMERA_NORMAL_BOUND);
    number(o, "suspect_lower_bound", CAMERA_SUSPECT_BOUND);
    text(o, "abstention_policy", "threshold-centered-v1"); text(o, "quality_gate_version", "laplacian-gray-v1");
    text(o, "confidence_definition", "assigned_class_probability; UNKNOWN=null");
    number(o, "reset_reason", esp_reset_reason());
    uint32_t flash_size;
    if (esp_flash_get_size(nullptr, &flash_size) == ESP_OK) number(o, "flash_chip_bytes", flash_size);
    else cJSON_AddNullToObject(o, "flash_chip_bytes");
    const esp_partition_t *part = esp_ota_get_running_partition();
    if (part) {
        number(o, "app_partition_bytes", part->size);
        esp_partition_pos_t pos = {part->address, part->size};
        esp_image_metadata_t image = {};
        if (esp_image_get_metadata(&pos, &image) == ESP_OK) number(o, "app_image_bytes", image.image_len);
        else cJSON_AddNullToObject(o, "app_image_bytes");
    }
    const esp_app_desc_t *desc = esp_app_get_description();
    char hash[65];
    for (int i = 0; i < 32; ++i) std::snprintf(hash + i * 2, 3, "%02x", desc->app_elf_sha256[i]);
    text(o, "firmware_sha256", hash); text(o, "firmware_hash_kind", "app_elf_sha256");
    memory(o);
    session = send_json(camera::HELLO_ACK, seq, o);
    sequences.restart(seq);
}
static void infer(uint32_t seq, size_t size)
{
    if (!session) { error(seq, "handshake_required", "Send HELLO after connect or reboot"); return; }
    if (!sequences.accept(seq)) { error(seq, "duplicate_or_old_sequence", "Frame already consumed"); return; }
    if (model_error) { error(seq, "model_not_loaded", model_error); return; }
    if (!send(camera::FRAME_ACK, seq)) { session = false; return; }
    int64_t begin = esp_timer_get_time();
    camera::Quality q;
    const camera::QualityConfig qc = {CAMERA_BLUR_MIN, CAMERA_BRIGHTNESS_MIN, CAMERA_BRIGHTNESS_MAX, CAMERA_CONTRAST_MIN};
    bool passed = camera::quality(rgb, size, qc, q);
    int64_t quality_us = esp_timer_get_time() - begin, preprocess_us = -1, inference_us = -1;
    float p = NAN;
    size_t saturated = 0;
    const char *state = "UNKNOWN";
    char reason[96] = {};
    if (!passed) {
        if (q.reasons & 1) std::strcat(reason, "blur;");
        if (q.reasons & 2) std::strcat(reason, "exposure;");
        if (q.reasons & 4) std::strcat(reason, "low_contrast;");
        if (q.reasons & 8) std::strcat(reason, "invalid_input;");
    } else {
        int64_t prestart = esp_timer_get_time();
        int8_t *quantized = static_cast<int8_t *>(input->data);
        float inverse_scale = DL_RESCALE(input->exponent);
        for (size_t i = 0; i < camera::FRAME_BYTES; ++i) {
            float value = camera::normalized(rgb[i], CAMERA_MEAN[i % 3], CAMERA_STD[i % 3]);
            quantized[i] = dl::quantize<int8_t>(value, inverse_scale);
            int rounded = dl::tool::round(value * inverse_scale);
            if (rounded < -128 || rounded > 127) ++saturated;
        }
        preprocess_us = esp_timer_get_time() - prestart;
        int64_t start = esp_timer_get_time();
        model->run();
        inference_us = esp_timer_get_time() - start;
        int8_t *logits = static_cast<int8_t *>(output->data);
        float a = dl::dequantize(logits[0], DL_SCALE(output->exponent));
        float b = dl::dequantize(logits[1], DL_SCALE(output->exponent));
        float maximum = std::max(a, b), ea = std::exp(a - maximum), eb = std::exp(b - maximum);
        p = eb / (ea + eb);
        state = camera::decide(p, CAMERA_NORMAL_BOUND, CAMERA_SUSPECT_BOUND);
        std::strcpy(reason, !std::isfinite(p) ? "invalid_model_output" :
                           std::strcmp(state, "UNKNOWN") == 0 ? "decision_boundary" : "threshold_interval");
    }
    cJSON *o = cJSON_CreateObject();
    if (!o) { error(seq, "memory", "Result allocation failed"); return; }
    number(o, "sequence", seq); text(o, "boot_id", boot_id); text(o, "state", state); text(o, "reason", reason);
    if (std::isfinite(p) && p >= 0 && p <= 1) {
        number(o, "normal_probability", 1.0f - p); number(o, "suspect_probability", p);
        if (!std::strcmp(state, "UNKNOWN")) cJSON_AddNullToObject(o, "confidence");
        else number(o, "confidence", !std::strcmp(state, "NORMAL") ? 1.0f - p : p);
    } else {
        cJSON_AddNullToObject(o, "normal_probability"); cJSON_AddNullToObject(o, "suspect_probability");
        cJSON_AddNullToObject(o, "confidence");
    }
    number(o, "decision_threshold", CAMERA_THRESHOLD); cJSON_AddBoolToObject(o, "quality_passed", passed);
    number(o, "brightness", q.brightness); number(o, "contrast", q.contrast);
    number(o, "laplacian_variance", q.laplacian_variance); number(o, "quality_us", quality_us);
    nullable(o, "preprocess_us", preprocess_us); nullable(o, "inference_us", inference_us);
    if (preprocess_us < 0) cJSON_AddNullToObject(o, "saturated_values"); else number(o, "saturated_values", saturated);
    memory(o); text(o, "model_version", CAMERA_MODEL_VERSION); text(o, "firmware_version", FW);
    number(o, "total_us", esp_timer_get_time() - begin);
    if (!send_json(camera::INFER_RESULT, seq, o)) session = false;
}
extern "C" void app_main(void)
{
    std::snprintf(boot_id, sizeof(boot_id), "%08lx%08lx", (unsigned long)esp_random(), (unsigned long)esp_random());
    esp_log_set_vprintf(discard_log); esp_log_level_set("*", ESP_LOG_NONE);
    if (!transport_init()) return;
    if (!load_model()) { model.reset(); input = output = nullptr; }
    static uint8_t fallback[camera::CONTROL_BYTES];
    camera::Decoder decoder(rgb ? rgb : fallback, rgb ? camera::FRAME_BYTES : sizeof(fallback));
    uint8_t chunk[4096];
    for (;;) {
        if (decoder.expire(esp_timer_get_time()) == camera::Event::TIMEOUT) {
            session = false; error(decoder.sequence, "timeout", "Partial frame expired; send HELLO");
        }
        int count = read_bytes(chunk, sizeof(chunk));
        for (int i = 0; i < count; ++i) {
            auto event = decoder.push(chunk[i], esp_timer_get_time());
            if (event == camera::Event::PACKET) {
                switch (decoder.type) {
                    case camera::HELLO: hello(decoder.sequence); break;
                    case camera::FRAME: infer(decoder.sequence, decoder.size()); break;
                    case camera::PING: send(camera::PING, decoder.sequence); break;
                    default: error(decoder.sequence, "unexpected_type", "Expected HELLO, FRAME or PING"); break;
                }
            } else if (event == camera::Event::BAD_CRC) {
                error(decoder.sequence, "payload_crc", "UNKNOWN: integrity check failed");
            } else if (event == camera::Event::BAD_LENGTH) {
                error(decoder.sequence, "frame_format", "UNKNOWN: dimensions, format or length rejected");
            }
        }
        vTaskDelay(1);
    }
}
