#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace hermes_person {
// Update one rectangular block of packed grayscale pixels. Keep the original
// float expression and lround semantics; no fixed-point or fast-math rounding.
inline void updateBackgroundPixels(uint8_t* bg, const uint8_t* now, int width,
                                   int x1, int y1, int x2, int y2,
                                   bool valid, float alpha) {
    const int count = x2 - x1;
    if (count <= 0 || y2 <= y1) return;
    if (!valid) {
        for (int y = y1; y < y2; ++y) {
            const size_t offset = size_t(y) * width + x1;
            // memmove also preserves the old copy behavior for identical buffers.
            std::memmove(bg + offset, now + offset, size_t(count));
        }
        return;
    }
    const float keep = 1 - alpha;
    for (int y = y1; y < y2; ++y) {
        const size_t offset = size_t(y) * width + x1;
        uint8_t* dst = bg + offset;
        const uint8_t* src = now + offset;
        for (int x = 0; x < count; ++x) {
            dst[x] = uint8_t(std::lround(keep * dst[x] + alpha * src[x]));
        }
    }
}
}
