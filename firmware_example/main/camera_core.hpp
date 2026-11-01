#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace camera {
constexpr size_t FRAME_BYTES = 160 * 160 * 3;
constexpr size_t HEADER_BYTES = 32;
constexpr size_t CONTROL_BYTES = 4096;
enum Type : uint8_t { HELLO = 1, HELLO_ACK, FRAME, FRAME_ACK, INFER_RESULT, ERROR, PING };

inline uint32_t crc32(const uint8_t *data, size_t size)
{
    uint32_t crc = ~0u;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int j = 0; j < 8; ++j) crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}
inline uint16_t u16(const uint8_t *p) { return uint16_t(p[0]) | (uint16_t(p[1]) << 8); }
inline uint32_t u32(const uint8_t *p) { return uint32_t(u16(p)) | (uint32_t(u16(p + 2)) << 16); }
inline void put16(uint8_t *p, uint16_t n) { p[0] = n; p[1] = n >> 8; }
inline void put32(uint8_t *p, uint32_t n) { put16(p, n); put16(p + 2, n >> 16); }

inline void header(uint8_t *h, Type type, uint32_t sequence, const uint8_t *data, size_t size)
{
    std::memset(h, 0, HEADER_BYTES);
    std::memcpy(h, "ATV2", 4);
    h[4] = 1;
    h[5] = type;
    put32(h + 8, sequence);
    if (type == FRAME) { put16(h + 12, 160); put16(h + 14, 160); h[16] = 1; }
    put32(h + 20, size);
    put32(h + 24, crc32(data, size));
    put32(h + 28, crc32(h, 28));
}

enum class Event { NONE, PACKET, BAD_HEADER, BAD_LENGTH, BAD_CRC, TIMEOUT };

struct SequenceWindow {
    uint32_t last = 0;
    void restart(uint32_t hello_sequence) { last = hello_sequence; }
    bool accept(uint32_t sequence)
    {
        if (!sequence || sequence <= last) return false;
        last = sequence;
        return true;
    }
};

class Decoder {
    uint8_t header_[HEADER_BYTES] = {};
    uint8_t *payload_;
    size_t capacity_, used_ = 0, received_ = 0, length_ = 0;
    int64_t started_ = 0, last_ = 0;

    void resync_header()
    {
        // Retain any possible magic prefix/suffix within a damaged header.
        for (size_t i = 1; i < used_; ++i) {
            size_t n = std::min(size_t(4), used_ - i);
            if (std::memcmp(header_ + i, "ATV2", n) == 0) {
                used_ -= i;
                std::memmove(header_, header_ + i, used_);
                return;
            }
        }
        used_ = 0;
    }

public:
    uint32_t sequence = 0;
    Type type = HELLO;
    explicit Decoder(uint8_t *payload, size_t capacity) : payload_(payload), capacity_(capacity) {}
    void reset() { used_ = received_ = length_ = 0; started_ = last_ = 0; }
    Event expire(int64_t now)
    {
        if (used_ && (now - started_ > 15000000 || now - last_ > 2000000)) {
            reset();
            return Event::TIMEOUT;
        }
        return Event::NONE;
    }
    size_t size() const { return length_; }
    Event push(uint8_t byte, int64_t now)
    {
        if (!used_) started_ = now;
        last_ = now;
        if (used_ < 4) {
            if (byte == uint8_t("ATV2"[used_])) header_[used_++] = byte;
            else { used_ = byte == 'A' ? 1 : 0; header_[0] = 'A'; }
            return Event::NONE;
        }
        if (used_ < HEADER_BYTES) {
            header_[used_++] = byte;
            if (used_ < HEADER_BYTES) return Event::NONE;
            if (header_[4] != 1 || u16(header_ + 6) || header_[17] || header_[18] || header_[19] ||
                crc32(header_, 28) != u32(header_ + 28) || header_[5] < HELLO || header_[5] > PING) {
                resync_header();
                return Event::BAD_HEADER;
            }
            sequence = u32(header_ + 8);
            type = Type(header_[5]);
            length_ = u32(header_ + 20);
            const bool dims = u16(header_ + 12) == 160 && u16(header_ + 14) == 160 && header_[16] == 1;
            const bool control_dims = !u16(header_ + 12) && !u16(header_ + 14) && !header_[16];
            if (!sequence || length_ > capacity_ || (type == FRAME ? (!dims || length_ != FRAME_BYTES) :
                (!control_dims || length_ > CONTROL_BYTES)) ||
                ((type == HELLO || type == FRAME_ACK || type == PING) && length_)) {
                resync_header();
                return Event::BAD_LENGTH;
            }
            received_ = 0;
        } else {
            // Header validation above bounds both length_ and every write.
            payload_[received_++] = byte;
        }
        if (received_ == length_) {
            const bool good = crc32(payload_, length_) == u32(header_ + 24);
            used_ = received_ = 0;
            return good ? Event::PACKET : Event::BAD_CRC;
        }
        return Event::NONE;
    }
};

struct QualityConfig { float blur_min, brightness_min, brightness_max, contrast_min; };
struct Quality { double brightness = 0, contrast = 0, laplacian_variance = 0; uint8_t reasons = 0; };
inline float gray(const uint8_t *rgb, int x, int y)
{
    const uint8_t *p = rgb + (y * 160 + x) * 3;
    return p[0] * 0.299f + p[1] * 0.587f + p[2] * 0.114f;
}

struct Moments {
    double sum = 0, squares = 0;
    size_t count = 0;
    void add(double v) { sum += v; squares += v * v; ++count; }
    double mean() const { return sum / count; }
    double variance() const { return std::max(0.0, squares / count - mean() * mean()); }
};
inline bool quality(const uint8_t *rgb, size_t size, const QualityConfig &config, Quality &q)
{
    if (!rgb || size != FRAME_BYTES) { q.reasons = 8; return false; }
    Moments intensity, lap;
    for (int y = 0; y < 160; ++y) for (int x = 0; x < 160; ++x) {
        float g = gray(rgb, x, y);
        intensity.add(g);
        if (x > 0 && x < 159 && y > 0 && y < 159) {
            lap.add(gray(rgb, x, y - 1) + gray(rgb, x, y + 1) + gray(rgb, x - 1, y) +
                    gray(rgb, x + 1, y) - 4 * g);
        }
    }
    q.brightness = intensity.mean();
    q.contrast = std::sqrt(intensity.variance());
    q.laplacian_variance = lap.variance();
    q.reasons = (q.laplacian_variance < config.blur_min ? 1 : 0) |
                (q.brightness < config.brightness_min || q.brightness > config.brightness_max ? 2 : 0) |
                (q.contrast < config.contrast_min ? 4 : 0);
    return q.reasons == 0;
}

inline float normalized(uint8_t pixel, float mean, float stddev)
{
    return (float(pixel) / 255.0f - mean) / stddev;
}
inline const char *decide(float p, float lower, float upper)
{
    return !std::isfinite(p) || p < 0 || p > 1 ? "UNKNOWN" :
           p < lower ? "NORMAL" : p < upper ? "UNKNOWN" : "SUSPECT";
}
} // namespace camera
