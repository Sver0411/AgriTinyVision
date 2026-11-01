#include "camera_core.hpp"
#include <array>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
    assert(argc == 3);
    std::ifstream file(argv[2], std::ios::binary);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), {});
    if (std::string(argv[1]) == "quality") {
        camera::Quality q;
        bool passed = camera::quality(data.data(), data.size(), {20, 25, 235, 12}, q);
        std::printf("%.12g %.12g %.12g %u %d\n", q.brightness, q.contrast, q.laplacian_variance, q.reasons, passed);
    } else if (std::string(argv[1]) == "normalize") {
        const float mean[] = {.485f, .456f, .406f}, stddev[] = {.229f, .224f, .225f};
        for (size_t i = 0; i < data.size(); ++i) {
            float value = camera::normalized(data[i], mean[i % 3], stddev[i % 3]);
            std::printf("%.9g\n", value);
        }
    } else {
        // Guard bytes detect writes beyond the one permitted RGB buffer.
        std::array<uint8_t, camera::FRAME_BYTES + 2> guarded;
        guarded.fill(0xa5);
        camera::Decoder decoder(guarded.data() + 1, camera::FRAME_BYTES);
        int64_t clock = 1;
        for (uint8_t byte : data) {
            auto event = decoder.push(byte, clock++);
            if (event == camera::Event::PACKET) {
                std::printf("packet %u %u %zu\n", unsigned(decoder.type), decoder.sequence, decoder.size());
            } else if (event != camera::Event::NONE) std::printf("error %u\n", unsigned(event));
            assert(guarded.front() == 0xa5 && guarded.back() == 0xa5);
        }
        if (decoder.expire(clock + 16000000) == camera::Event::TIMEOUT) std::puts("timeout");
        assert(camera::crc32(reinterpret_cast<const uint8_t *>("123456789"), 9) == 0xcbf43926);
        assert(std::string(camera::decide(.2f, .2f, .4f)) == "UNKNOWN");
        assert(std::string(camera::decide(.4f, .2f, .4f)) == "SUSPECT");
        assert(std::string(camera::decide(NAN, .2f, .4f)) == "UNKNOWN");
        camera::SequenceWindow window;
        window.restart(1);
        assert(window.accept(2));
        assert(!window.accept(2) && !window.accept(1) && !window.accept(0));
        window.restart(1);
        assert(window.accept(2));
    }
}
