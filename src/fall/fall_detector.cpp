#include "fall_detector.h"
#include "../tracking/kalman_filter.h"
#include "../tracking/hungarian.h"
#include <cmath> // sqrt, abs
#include <algorithm> // min, max
#include <limits>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <deque>
#include <map>
#include <set>
#include <queue>

#include <cstdio>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstring>
#include <climits>
#define ENABLE_PERF_PROFILING 1

using namespace std;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace VisionSDK {

namespace {

// =========================================================
// Visualization Utilities (RGB)
// =========================================================

// Save RGB buffer (3 bytes per pixel) to BMP
bool saveBMP_RGB(const std::string& filename, const uint8_t* rgbData, int width, int height) {
    FILE* f = fopen(filename.c_str(), "wb");
    if (!f) return false;

    int filesize = 54 + 3 * width * height;
    uint8_t header[54] = {
        0x42, 0x4D, 0,0,0,0, 0,0,0,0, 54,0,0,0, 40,0,0,0,
        0,0,0,0, 0,0,0,0, 1,0, 24,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0
    };

    header[2] = (uint8_t)(filesize);
    header[3] = (uint8_t)(filesize >> 8);
    header[4] = (uint8_t)(filesize >> 16);
    header[5] = (uint8_t)(filesize >> 24);
    header[18] = (uint8_t)(width);
    header[19] = (uint8_t)(width >> 8);
    header[20] = (uint8_t)(width >> 16);
    header[21] = (uint8_t)(width >> 24);
    header[22] = (uint8_t)(height);
    header[23] = (uint8_t)(height >> 8);
    header[24] = (uint8_t)(height >> 16);
    header[25] = (uint8_t)(height >> 24);

    fwrite(header, 1, 54, f);

    int padSize = (4 - (width * 3) % 4) % 4;
    uint8_t pad[3] = {0, 0, 0};

    // BMP is stored bottom-to-top, BGR format
    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;
            uint8_t r = rgbData[idx];
            uint8_t g = rgbData[idx + 1];
            uint8_t b = rgbData[idx + 2];
            uint8_t bgr[3] = {b, g, r}; // Swap for BMP
            fwrite(bgr, 1, 3, f);
        }
        fwrite(pad, 1, padSize, f);
    }
    fclose(f);
    return true;
}

void drawPixelRGB(std::vector<uint8_t>& img, int w, int h, int x, int y, uint8_t r, uint8_t g, uint8_t b) {
    if (x >= 0 && x < w && y >= 0 && y < h) {
        int idx = (y * w + x) * 3;
        img[idx] = r;
        img[idx + 1] = g;
        img[idx + 2] = b;
    }
}

void drawLineRGB(std::vector<uint8_t>& img, int w, int h, int x1, int y1, int x2, int y2, uint8_t r, uint8_t g, uint8_t b, int thickness = 1) {
    int dx = std::abs(x2 - x1), sx = x1 < x2 ? 1 : -1;
    int dy = -std::abs(y2 - y1), sy = y1 < y2 ? 1 : -1;
    int err = dx + dy, e2; 
    
    // Helper to draw brush
    auto drawBrush = [&](int cx, int cy) {
        for (int ty = -thickness/2; ty <= thickness/2; ty++) {
            for (int tx = -thickness/2; tx <= thickness/2; tx++) {
                drawPixelRGB(img, w, h, cx + tx, cy + ty, r, g, b);
            }
        }
    };

    while (true) {
        drawBrush(x1, y1);
        if (x1 == x2 && y1 == y2) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x1 += sx; }
        if (e2 <= dx) { err += dx; y1 += sy; }
    }
}

void drawRectRGB(std::vector<uint8_t>& img, int w, int h, int x, int y, int rw, int rh, uint8_t r, uint8_t g, uint8_t b, int thickness = 2) {
    int x2 = x + rw - 1;
    int y2 = y + rh - 1;
    drawLineRGB(img, w, h, x, y, x2, y, r, g, b, thickness);
    drawLineRGB(img, w, h, x2, y, x2, y2, r, g, b, thickness);
    drawLineRGB(img, w, h, x2, y2, x, y2, r, g, b, thickness);
    drawLineRGB(img, w, h, x, y2, x, y, r, g, b, thickness);
}

void drawFilledRectRGB(std::vector<uint8_t>& img, int w, int h, int x, int y, int rw, int rh, uint8_t r, uint8_t g, uint8_t b) {
    for (int j = y; j < y + rh; j++) {
        for (int i = x; i < x + rw; i++) {
            drawPixelRGB(img, w, h, i, j, r, g, b);
        }
    }
}

void drawArrowRGB(std::vector<uint8_t>& img, int w, int h, int x, int y, int dx, int dy, uint8_t r, uint8_t g, uint8_t b, int thickness = 2) {
    int x2 = x + dx;
    int y2 = y + dy;
    drawLineRGB(img, w, h, x, y, x2, y2, r, g, b, thickness);
    
    // Draw Arrow Head
    // Simple 30 degree wings
    if (std::abs(dx) + std::abs(dy) > 5) {
        float angle = atan2((float)dy, (float)dx);
        float headLen = 15.0f; // Longer head
        float angle1 = angle + M_PI * 0.85; // Backwards angle
        float angle2 = angle - M_PI * 0.85;

        int x3 = x2 + (int)(cos(angle1) * headLen);
        int y3 = y2 + (int)(sin(angle1) * headLen);
        int x4 = x2 + (int)(cos(angle2) * headLen);
        int y4 = y2 + (int)(sin(angle2) * headLen);

        drawLineRGB(img, w, h, x2, y2, x3, y3, r, g, b, thickness);
        drawLineRGB(img, w, h, x2, y2, x4, y4, r, g, b, thickness);
    }
}


#if 0
struct ObjectFeatures {
    int area;          // 面積 (pixel count)
    float cx, cy;      // 重心
    float angle;       // 角度
    float major, minor;// 長短軸
};
#endif
// 假設 mask 已經是上面 NEON 算出來的 0/255 陣列
// width, height: 影像尺寸
// minArea: 過濾噪點用
std::vector<::VisionSDK::ObjectFeatures> find_objects_optimized(const uint8_t* mask, int width, int height, int minArea) {
    std::vector<::VisionSDK::ObjectFeatures> results;
    
    static std::vector<uint8_t> visited;
    if (visited.size() != (size_t)(width * height)) {
        visited.assign(width * height, 0);
    } else {
        std::fill(visited.begin(), visited.end(), 0);
    }
    
    int dx[] = {1, -1, 0, 0};
    int dy[] = {0, 0, 1, -1};

    struct BlobData {
        long long sum_x, sum_y, sum_xx, sum_yy, sum_xy;
        int count;
        int min_x, max_x, min_y, max_y;
        std::vector<int> pixels;
        bool merged;
        int parent;
    };
    std::vector<BlobData> blobs;

    // 1. Initial Blob Extraction (Flood Fill)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            if (mask[idx] == 255 && visited[idx] == 0) {
                BlobData blob;
                blob.sum_x = blob.sum_y = blob.sum_xx = blob.sum_yy = blob.sum_xy = 0;
                blob.count = 0;
                blob.min_x = x; blob.max_x = x;
                blob.min_y = y; blob.max_y = y;
                blob.merged = false;
                blob.parent = (int)blobs.size();

                std::queue<int> q;
                q.push(idx);
                visited[idx] = 1;

                while (!q.empty()) {
                    int curr = q.front();
                    q.pop();

                    int cy_curr = curr / width;
                    int cx_curr = curr % width;

                    blob.sum_x += cx_curr;
                    blob.sum_y += cy_curr;
                    blob.sum_xx += (long long)cx_curr * cx_curr;
                    blob.sum_yy += (long long)cy_curr * cy_curr;
                    blob.sum_xy += (long long)cx_curr * cy_curr;
                    blob.count++;
                    blob.pixels.push_back(curr);

                    if (cx_curr < blob.min_x) blob.min_x = cx_curr;
                    if (cx_curr > blob.max_x) blob.max_x = cx_curr;
                    if (cy_curr < blob.min_y) blob.min_y = cy_curr;
                    if (cy_curr > blob.max_y) blob.max_y = cy_curr;

                    for (int i = 0; i < 4; ++i) {
                        int nx = cx_curr + dx[i];
                        int ny = cy_curr + dy[i];
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            int nidx = ny * width + nx;
                            if (mask[nidx] == 255 && visited[nidx] == 0) {
                                visited[nidx] = 1;
                                q.push(nidx);
                            }
                        }
                    }
                }
                blobs.push_back(std::move(blob));
            }
        }
    }

    // 2. BBox Expansion Merging (DSU-like logic)
    int expansion = 1;  // Reduced from 10 to 1 pixel for tighter merging
    bool changed = true;
    while (changed) {
        changed = false;
        for (int i = 0; i < (int)blobs.size(); ++i) {
            if (blobs[i].merged) continue;
            for (int j = i + 1; j < (int)blobs.size(); ++j) {
                if (blobs[j].merged) continue;

                // Intersection of expanded BBoxes
                bool overlap = !(blobs[i].max_x + expansion < blobs[j].min_x - expansion ||
                                blobs[i].min_x - expansion > blobs[j].max_x + expansion ||
                                blobs[i].max_y + expansion < blobs[j].min_y - expansion ||
                                blobs[i].min_y - expansion > blobs[j].max_y + expansion);

                if (overlap) {
                    // Merge j into i
                    blobs[i].sum_x += blobs[j].sum_x;
                    blobs[i].sum_y += blobs[j].sum_y;
                    blobs[i].sum_xx += blobs[j].sum_xx;
                    blobs[i].sum_yy += blobs[j].sum_yy;
                    blobs[i].sum_xy += blobs[j].sum_xy;
                    blobs[i].count += blobs[j].count;
                    blobs[i].pixels.insert(blobs[i].pixels.end(), blobs[j].pixels.begin(), blobs[j].pixels.end());
                    
                    if (blobs[j].min_x < blobs[i].min_x) blobs[i].min_x = blobs[j].min_x;
                    if (blobs[j].max_x > blobs[i].max_x) blobs[i].max_x = blobs[j].max_x;
                    if (blobs[j].min_y < blobs[i].min_y) blobs[i].min_y = blobs[j].min_y;
                    if (blobs[j].max_y > blobs[i].max_y) blobs[i].max_y = blobs[j].max_y;

                    blobs[j].merged = true;
                    changed = true;
                }
            }
        }
    }

    // 3. Re-calculating Features for final objects
    for (auto& blob : blobs) {
        if (!blob.merged && blob.count > minArea) {
            float mean_x = (float)blob.sum_x / blob.count;
            float mean_y = (float)blob.sum_y / blob.count;

            float u20 = (float)blob.sum_xx / blob.count - mean_x * mean_x;
            float u02 = (float)blob.sum_yy / blob.count - mean_y * mean_y;
            float u11 = (float)blob.sum_xy / blob.count - mean_x * mean_y;

            float delta = std::sqrt(4 * u11 * u11 + (u20 - u02) * (u20 - u02));
            float lambda1 = (u20 + u02 + delta) / 2;
            float lambda2 = (u20 + u02 - delta) / 2;

            ::VisionSDK::ObjectFeatures obj;
            obj.id = (int)results.size();
            obj.area = blob.count;
            obj.cx = mean_x;
            obj.cy = mean_y;
            obj.major = 4.0f * std::sqrt(std::max(0.0f, lambda1));
            obj.minor = 4.0f * std::sqrt(std::max(0.0f, lambda2));
            obj.angle = 0.5f * std::atan2(2 * u11, u20 - u02);
            obj.pixels = std::move(blob.pixels);
            
            results.push_back(obj);
        }
    }

    return results;
}


// Minimal 5x7 bitmap font
const uint8_t font5x7[] = {
    // 0-9
    0x1F,0x11,0x1F, 0x00,0x1F,0x00, 0x1D,0x15,0x17, 0x15,0x15,0x1F, 0x07,0x04,0x1F,
    0x17,0x15,0x1D, 0x1F,0x15,0x1D, 0x01,0x01,0x1F, 0x1F,0x15,0x1F, 0x17,0x15,0x1F,
    // .
    0x10,0x00,0x00,
    // F (11)
    0x1F,0x05,0x00,
    // A (12)
    0x1F,0x05,0x1F, 
    // L (13)
    0x1F,0x10,0x10,
    // T (14)
    0x01,0x1F,0x01,
    // R (15)
    0x1F,0x05,0x1A,
    // U (16)
    0x1F,0x10,0x1F,
    // E (17)
    0x1F,0x15,0x11,
    // S (18)
    0x1D,0x15,0x17,
     // M (19)
    0x1F,0x02,0x1F,
    // I (20)
    0x00,0x1F,0x00,
    // B (21)
    0x1F,0x15,0x0A,
    // D (22)
    0x1F,0x11,0x0E,
    // X (23)
    0x11,0x04,0x11
};

void drawCharRGB(std::vector<uint8_t>& img, int w, int h, int cx, int cy, int charArgs, uint8_t r, uint8_t g, uint8_t b, int scale = 1) {
    if (charArgs < 0 || charArgs > 23) return;
    const uint8_t* ptr = font5x7 + charArgs * 3;
    for (int col = 0; col < 3; col++) {
        uint8_t colData = ptr[col];
        for (int row = 0; row < 5; row++) {
            if ((colData >> row) & 1) { 
                // Draw Scaled Pixel
                for (int sy = 0; sy < scale; sy++) {
                    for (int sx = 0; sx < scale; sx++) {
                        int px = cx + (col * 2) * scale + sx;
                        int py = cy + (row * 2) * scale + sy;
                         drawPixelRGB(img, w, h, px, py, r, g, b);
                    }
                }
            }
        }
    }
}

void drawStringRGB(std::vector<uint8_t>& img, int w, int h, int x, int y, const std::string& s, uint8_t r, uint8_t g, uint8_t b, int scale = 1) {
    int cx = x;
    for (char c : s) {
        int idx = -1;
        if (c >= '0' && c <= '9') idx = c - '0';
        else if (c == '.') idx = 10;
        else if (c == 'F') idx = 11;
        else if (c == 'A') idx = 12;
        else if (c == 'L') idx = 13;
        else if (c == 'T') idx = 14;
        else if (c == 'R') idx = 15;
        else if (c == 'U') idx = 16;
        else if (c == 'E') idx = 17;
        else if (c == 'S') idx = 18;
        else if (c == 'M') idx = 19;
        else if (c == 'I') idx = 20;
        else if (c == 'B') idx = 21;
        else if (c == 'D') idx = 22;
        else if (c == 'X') idx = 23;
        
        if (idx != -1) {
            drawCharRGB(img, w, h, cx, y, idx, r, g, b, scale);
            cx += 8 * scale; 
        } else {
             cx += 4 * scale; 
        }
    }
}

// Helper: Get Static Foreground BBox using Background Subtraction (Raw Data)
static bool getStaticBoundingBoxRaw(const unsigned char* currData, int currW, int currH, int currC,
                                    const unsigned char* bgData, int bgW, int bgH,
                                    int center_x, int center_y, int search_radius, 
                                    int& out_w, int& out_h) {
     if (currW != bgW || currH != bgH) return false;
     
     int h = currH;
     int w = currW;
     int c = currC;
     
     int min_x = w, max_x = 0;
     int min_y = h, max_y = 0;
     int count = 0;
     
     int start_x = std::max(0, center_x - search_radius);
     int end_x = std::min(w, center_x + search_radius);
     int start_y = std::max(0, center_y - search_radius);
     int end_y = std::min(h, center_y + search_radius);
     
     int thresh = 30; // FG Threshold
     
     for(int y=start_y; y<end_y; y+=2) { // Skip lines for speed
         for(int x=start_x; x<end_x; x+=2) {
             int idx = (y * w + x) * c;
             int diff = 0;
             if (c == 1) {
                  diff = std::abs((int)currData[idx] - (int)bgData[idx]);
             } else {
                  diff = std::abs((int)currData[idx] - (int)bgData[idx]) +
                         std::abs((int)currData[idx+1] - (int)bgData[idx+1]) + 
                         std::abs((int)currData[idx+2] - (int)bgData[idx+2]);
                  diff /= 3;
             }
             
             if (diff > thresh) {
                 if(x < min_x) min_x = x;
                 if(x > max_x) max_x = x;
                 if(y < min_y) min_y = y;
                 if(y > max_y) max_y = y;
                 count++;
             }
         }
     }
     
     if (count < 50) return false; // Too small / noise
     
     out_w = max_x - min_x;
     out_h = max_y - min_y;
     return true;
 }

 static bool drawFloatRGB(std::vector<uint8_t>& img, int w, int h, int x, int y, float v, uint8_t r, uint8_t g, uint8_t b, int scale = 1) {
    char buf[16];
    sprintf(buf, "%.1f", v);
    drawStringRGB(img, w, h, x, y, buf, r, g, b, scale);
    return true;
}

// Calculate bounding box size (in pixels) from object's block list
// Returns the width and height of the bounding box enclosing all blocks
// block_width and block_height are the size of each block in pixels
static void getObjectBoundingBoxPixels(const MotionObject& obj, int grid_cols, int grid_rows, 
                                       int frame_width, int frame_height,
                                       int& out_bbox_width, int& out_bbox_height, 
                                       int& out_min_x, int& out_min_y,
                                       int& out_max_x, int& out_max_y) {
    // Calculate block dimensions
    int block_width = frame_width / grid_cols;
    int block_height = frame_height / grid_rows;
    
    // Find min/max block indices
    int min_block_x = INT_MAX; int max_block_x = INT_MIN;
    int min_block_y = INT_MAX; int max_block_y = INT_MIN;
    
    for (int block_idx : obj.blocks) {
        int block_row = block_idx / grid_cols;
        int block_col = block_idx % grid_cols;
        
        min_block_x = std::min(min_block_x, block_col);
        max_block_x = std::max(max_block_x, block_col);
        min_block_y = std::min(min_block_y, block_row);
        max_block_y = std::max(max_block_y, block_row);
    }
    
    if (obj.blocks.empty()) {
        out_bbox_width = 0;
        out_bbox_height = 0;
        out_min_x = out_min_y = out_max_x = out_max_y = 0;
        return;
    }
    
    // Convert to pixel coordinates
    out_min_x = min_block_x * block_width;
    out_min_y = min_block_y * block_height;
    out_max_x = (max_block_x + 1) * block_width;  // +1 to get right edge of block
    out_max_y = (max_block_y + 1) * block_height; // +1 to get bottom edge of block
    
    out_bbox_width = out_max_x - out_min_x;
    out_bbox_height = out_max_y - out_min_y;
}

} // anonymous namespace

// ==================================================================================
//  Helpers & Profiler (from C_V2_EDGE.cpp)
// ==================================================================================

struct FunctionTimer {
    std::map<std::string, double> total_ms;
    std::map<std::string, uint64_t> calls;

    inline void add(const std::string& name, double ms) {
        total_ms[name] += ms;
        calls[name] += 1ULL;
    }
};

struct TimerGuard {
    FunctionTimer* prof;
    std::string name;
    std::chrono::high_resolution_clock::time_point t0;
    TimerGuard(FunctionTimer& p, const std::string& n) : prof(&p), name(n), t0(std::chrono::high_resolution_clock::now()) {}
    ~TimerGuard() {
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
        prof->add(name, ms);
    }
};

// ==================================================================================
//  OptimizedBlockMotionEstimator (from C_V2_EDGE.cpp)
// ==================================================================================

class OptimizedBlockMotionEstimator {
public:
    OptimizedBlockMotionEstimator(int h_blocks, int v_blocks, int block_sz, int sr)
    : horizontal_blocks(h_blocks), vertical_blocks(v_blocks),
      block_size(block_sz), search_range(sr), diff_check_range(5) 
    {
        block_active_counters.assign(h_blocks * v_blocks, 0);
    }

    void setDiffCheckRange(int n) { diff_check_range = n; }
    void setSearchMode(int mode) { search_algo_mode = mode; }
    void setBlockDecay(bool enable, int frames) { 
        enable_block_decay = enable; 
        block_decay_max_frames = frames; 
        printf("DEBUG: setBlockDecay enable=%d frames=%d\n", enable, frames);
    }

    void setBlockDilation(bool enable) {
        enable_block_dilation = enable;
    }

    void blockBasedMotionEstimation(const ::Image& curr_frame_in,
                                    std::vector<MotionVector>& motion_vectors,
                                    std::vector<BlockPosition>& positions,            
                                    std::vector<bool>& changed_blocks_mask,
                                    std::vector<::Image>& active_blocks,
                                    std::vector<int>& active_indices,
                                    int dilation_threshold = 2)
    {
        // TimerGuard total_t(profiler, "blockBasedMotionEstimation_total");

        motion_vectors.clear();
        positions.clear();
        changed_blocks_mask.clear();
        active_blocks.clear();
        active_indices.clear();

        const int total_blocks = horizontal_blocks * vertical_blocks;
        
        if (prev_frames.empty()) 
        {
            buildPositionsOnce(curr_frame_in, positions_cache);
            positions.assign(positions_cache.begin(), positions_cache.end());
            motion_vectors.assign(total_blocks, MotionVector(0, 0));
            changed_blocks_mask.assign(total_blocks, false);
            prev_frames.push_back(curr_frame_in.clone());
            return;
        }

        const ::Image& prev_frame_ref = prev_frames.back(); // For motion estimation (t-1)

        const ::Image& curr_frame = curr_frame_in;

        if ((int)positions_cache.size() != total_blocks) {
             buildPositionsOnce(curr_frame, positions_cache);
        }
        positions = positions_cache;

        // --- Change Detection ---
        // --- Change Detection ---
        changed_blocks_mask.assign(total_blocks, false);
        detectChangedBlocks(prev_frames, curr_frame, changed_blocks_mask, 0.05, dilation_threshold);

        // --- Block Decay Logic ---
        if (enable_block_decay) {
            // printf("DEBUG: Applying Block Decay...\n");
            if ((int)block_active_counters.size() != total_blocks) {
                block_active_counters.assign(total_blocks, 0);
            }

            for (int i = 0; i < total_blocks; ++i) {
                if (changed_blocks_mask[i]) {
                    // Reset counter if block is naturally active
                    block_active_counters[i] = block_decay_max_frames;
                } else if (block_active_counters[i] > 0) {
                    // Decrease counter and FORCE active
                    block_active_counters[i]--;
                    changed_blocks_mask[i] = true;
                }
            }
        }

        // --- ROI Creation & Motion Estimation ---
        // Zero-copy: Avoid creating 'active_blocks' which are deep copies.
        // Instead, we just iterate indices and pass pointers.
        
        // divideActiveBlocks logic is simple: if changed_mask[i] is true, it's active.
        // We can just iterate directly.
        motion_vectors.assign(total_blocks, MotionVector(0, 0));

        const uint8_t* curr_data = curr_frame.getData();
        int curr_stride = curr_frame.width();

        for (int i = 0; i < total_blocks; ++i) {
             if (!changed_blocks_mask[i]) continue;
             
             const BlockPosition& pos = positions_cache[i];
             int bw = pos.x_end - pos.x_start;
             int bh = pos.y_end - pos.y_start;
             
             // Point to ROI in current frame
             const uint8_t* block_ptr = curr_data + pos.y_start * curr_stride + pos.x_start;

             int dx = 0, dy = 0, sad = 0;
             optimizedMotionEstimation(block_ptr, curr_stride, prev_frame_ref,
                                       pos.x_start, pos.y_start,
                                       bh, bw, dx, dy, sad, i);

             motion_vectors[i] = MotionVector(dx, dy);
        }

        updateMotionHistoryFast(motion_vectors);
        prev_frames.push_back(curr_frame.clone());
        if((int)prev_frames.size() > diff_check_range) prev_frames.pop_front();
    }

    FunctionTimer profiler;

    void PrintTimings() {
         std::cout << "\n=== AVG TIMINGS ===\n";
         for(auto& kv : profiler.total_ms) {
             std::cout << kv.first << ": " << kv.second << " ms\n";
         }
    }

private:
    int horizontal_blocks;
    int vertical_blocks;
    int block_size;
    int search_range;
    int diff_check_range;
    int search_algo_mode = 1; // Default to LDSP

    std::deque<::Image> prev_frames;
    std::vector<BlockPosition> positions_cache;
    std::vector<std::vector<MotionVector>> motion_history;
    
    // Block Decay Logic
    std::vector<int> block_active_counters;
    bool enable_block_decay = false;
    int block_decay_max_frames = 0;

    // Block Dilation Logic
    bool enable_block_dilation = false;

    void buildPositionsOnce(const ::Image& img, std::vector<BlockPosition>& out) {
        out.clear();
        int h = img.height(), w = img.width();
        int bh = h / vertical_blocks;
        int bw = w / horizontal_blocks;

        out.reserve(horizontal_blocks * vertical_blocks);
        for (int by = 0; by < vertical_blocks; ++by) {
            for (int bx = 0; bx < horizontal_blocks; ++bx) {
                int xs = bx * bw;
                int ys = by * bh;
                int xe = (bx == horizontal_blocks - 1) ? w : xs + bw;
                int ye = (by == vertical_blocks - 1) ? h : ys + bh;
                BlockPosition p; 
                p.x_start = xs; p.x_end = xe; 
                p.y_start = ys; p.y_end = ye;
                p.i=by; p.j=bx;
                out.push_back(p);
            }
        }
    }

    // NEON optimized SAD row
    static inline int sad_u8_row(const uint8_t* a, const uint8_t* b, int len) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        int sad = 0;
        int n = len;
        while (n >= 16) {
            uint8x16_t va = vld1q_u8(a);
            uint8x16_t vb = vld1q_u8(b);
            uint8x16_t vdiff = vabdq_u8(va, vb);
            uint16x8_t vpad1 = vpaddlq_u8(vdiff);
            uint32x4_t vpad2 = vpaddlq_u16(vpad1);
            uint64x2_t vpad3 = vpaddlq_u32(vpad2);
            sad += (int)vgetq_lane_u64(vpad3, 0);
            sad += (int)vgetq_lane_u64(vpad3, 1);
            a += 16; b += 16; n -= 16;
        }
        while (n--) sad += std::abs((int)(*a++) - (int)(*b++));
        return sad;
#else
        int sad = 0;
        for (int i=0;i<len;++i) sad += std::abs((int)a[i] - (int)b[i]);
        return sad;
#endif
    }

    int computeSAD(const uint8_t* block_data, int block_stride, int bw, int bh, const ::Image& prev_frame, int ref_x, int ref_y) {
        const int stride_prev = prev_frame.width();
        const uint8_t* pprev  = prev_frame.getData() + ref_y * stride_prev + ref_x;

        int sad = 0;
        for (int y=0; y<bh; ++y) {
            sad += sad_u8_row(block_data + y * block_stride, pprev + y * stride_prev, bw);
        }
        return sad;
    }


    // Optimized NEON Change Ratio Calculation (Fuses binarize + count)
    double computeChangeRatio(const ::Image& prev, const ::Image& curr, const BlockPosition& pos, int threshold = 30) {
        const uint8_t* pprev = prev.getData();
        const uint8_t* pcurr = curr.getData();
        const int stride = prev.width();
        int count = 0;
        int roi_w = pos.x_end - pos.x_start;
        int roi_h = pos.y_end - pos.y_start;
        
        const uint8_t* r_prev = pprev + pos.y_start * stride + pos.x_start;
        const uint8_t* r_curr = pcurr + pos.y_start * stride + pos.x_start;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        const uint8x16_t vthr = vdupq_n_u8((uint8_t)threshold);
        for(int y=0; y<roi_h; ++y) {
            int x = 0;
            const uint8_t* rp = r_prev + y * stride;
            const uint8_t* rc = r_curr + y * stride;
            
            // Vectorized loop
            for (; x <= roi_w - 16; x += 16) {
                uint8x16_t va = vld1q_u8(rp + x);
                uint8x16_t vb = vld1q_u8(rc + x);
                uint8x16_t vdiff = vabdq_u8(va, vb);
                uint8x16_t vmask = vcgtq_u8(vdiff, vthr);
                // Count bits in vmask? 
                // vmask has 0xFF for set, 0x00 for unset.
                // Population count is expensive in NEON v7 without special instruction.
                // Alternative: accumulate mask values? 0xFF = -1 in int8.
                // Let's use simple check: if vmask is all zero, skip.
                // Or just spill to stack? No.
                // Accumulate to count:
                // cnt += population_count(vmask).
                // Efficient way:
                // vcntq_u8 counts bits. 0xFF has 8 bits. We want bytes.
                // simple: shift right and add?
                // Actually, vcnt is available in AArch64 or newer.
                // Cortex-A7 supports vcnt? Yes (NEON VFPv4).
                uint8x16_t vones = vcntq_u8(vmask); // Each byte becomes 8 (for 0xFF) or 0.
                // Sum bytes.
                // Pairwise add chain.
                uint16x8_t vp = vpaddlq_u8(vones);
                uint32x4_t vp2 = vpaddlq_u16(vp);
                uint64x2_t vp3 = vpaddlq_u32(vp2);
                uint64_t sum = vgetq_lane_u64(vp3, 0) + vgetq_lane_u64(vp3, 1);
                count += (int)(sum / 8); 
            }
            // Scalar tail
            for (; x < roi_w; ++x) {
                if (std::abs((int)rp[x] - (int)rc[x]) > threshold) count++;
            }
        }
#else
        for(int y=0; y<roi_h; ++y) {
            const uint8_t* rp = r_prev + y * stride;
            const uint8_t* rc = r_curr + y * stride;
            for(int x=0; x<roi_w; ++x) {
                if (std::abs((int)rp[x] - (int)rc[x]) > threshold) count++;
            }
        }
#endif
        return (double)count / (roi_w * roi_h);
    }

    int detectChangedBlocks(const std::deque<::Image>& history, const ::Image& curr,
                            std::vector<bool>& changed_blocks_mask,
                            double threshold_ratio,
                            int dilation_threshold = 2)
    {
        int total_changed = 0;
        
        // Loop through recent history to find ANY change
        // We iterate positions first to potentially break early? 
        // No, current logic is "OR" across history frames.
        // If a block changed in ANY frame, it's changed.
        
        // Cache optimization: Swap loops?
        // If we iterate history inside, we read image multiple times.
        // History is small (5 frames).
        // Positions are many (12x16 = 192).
        // Iterating blocks efficiently is key.
        // Current logic: For each history frame, create diff, check all blocks.
        // Optimized logic: For each block, check history frames until change found.
        
        for(size_t i=0; i<positions_cache.size(); ++i) {
             if (changed_blocks_mask[i]) continue; // Already marked changed

             const auto& pos = positions_cache[i];
             
             // Check against history
             for (const auto& past_frame : history) {
                 double ratio = computeChangeRatio(past_frame, curr, pos, 30);
                 if(ratio > threshold_ratio) {
                     changed_blocks_mask[i] = true;
                     total_changed++;
                     break; // Found change, move to next block
                 }
             }
        }

        // Post-Processing: Hole Filling (Dilation) at Block Level
        // If a block is unchanged but has >= 2 changed neighbors, mark it changed.
        if (enable_block_dilation) {
            std::vector<bool> dilation_mask = changed_blocks_mask;
            int added_by_dilation = 0;
            
            for (int by = 0; by < vertical_blocks; ++by) {
                for (int bx = 0; bx < horizontal_blocks; ++bx) {
                    int idx = by * horizontal_blocks + bx;
                    if (changed_blocks_mask[idx]) continue;

                    int neighbors = 0;
                    if (bx > 0 && changed_blocks_mask[idx - 1]) neighbors++; // Left
                    if (bx < horizontal_blocks - 1 && changed_blocks_mask[idx + 1]) neighbors++; // Right
                    if (by > 0 && changed_blocks_mask[idx - horizontal_blocks]) neighbors++; // Up
                    if (by < vertical_blocks - 1 && changed_blocks_mask[idx + horizontal_blocks]) neighbors++; // Down
                    
                    if (neighbors >= dilation_threshold) {
                        dilation_mask[idx] = true;
                        added_by_dilation++;
                    }
                }
            }
            
            if (added_by_dilation > 0) {
                changed_blocks_mask = dilation_mask;
                total_changed += added_by_dilation;
            }
        }

        return total_changed;
    }


    void predictMotionVectorFast(int block_idx, int& pdx, int& pdy) {
        if(!motion_history.empty() && block_idx < (int)motion_history.back().size()) {
             pdx = motion_history.back()[block_idx].dx;
             pdy = motion_history.back()[block_idx].dy;
        } else {
             pdx = 0; pdy = 0;
        }
    }

    void updateMotionHistoryFast(const std::vector<MotionVector>& mvs) {
        motion_history.push_back(mvs);
        if(motion_history.size() > 5) motion_history.erase(motion_history.begin());
    }

    void optimizedMotionEstimation(const uint8_t* block_data, int block_stride, const ::Image& prev,
                                   int x_start, int y_start, int bh, int bw,
                                   int& best_dx, int& best_dy, int& best_sad, int idx)
    {
        int pdx=0, pdy=0;
        predictMotionVectorFast(idx, pdx, pdy);
        if (search_algo_mode == 0) {
            fastDiamondSearch(block_data, block_stride, prev, x_start, y_start, bh, bw, pdx, pdy, best_dx, best_dy, best_sad);
        } else {
            LDSP_SDSP_Search(block_data, block_stride, prev, x_start, y_start, bh, bw, pdx, pdy, best_dx, best_dy, best_sad);
        }
    }

    void fastDiamondSearch(const uint8_t* block_data, int block_stride, const ::Image& prev,
                           int x_start, int y_start, int bh, int bw,
                           int init_dx, int init_dy,
                           int& best_dx, int& best_dy, int& best_sad)
    {
        best_dx = init_dx; best_dy = init_dy;
        
        auto safe_in = [&](int rx, int ry) {
            return (rx>=0 && ry>=0 && rx+bw<=prev.width() && ry+bh<=prev.height());
        };

        best_sad = std::numeric_limits<int>::max();
        if(safe_in(x_start+best_dx, y_start+best_dy)) {
             best_sad = computeSAD(block_data, block_stride, bw, bh, prev, x_start+best_dx, y_start+best_dy);
        }

        static const int small_diamond[5][2] = { {0,0},{-1,0},{1,0},{0,-1},{0,1} };
        bool improved = true;
        int iter = 0;
        while(improved && iter < 3) {
            improved = false;
            int cur_dx = best_dx, cur_dy = best_dy;
            for(int k=0; k<5; ++k) {
                int cdx = cur_dx + small_diamond[k][0];
                int cdy = cur_dy + small_diamond[k][1];
                if(std::abs(cdx) > search_range || std::abs(cdy) > search_range) continue;
                int rx = x_start + cdx;
                int ry = y_start + cdy;
                if(!safe_in(rx, ry)) continue;
                
                int sad = computeSAD(block_data, block_stride, bw, bh, prev, rx, ry);
                if(sad < best_sad) {
                    best_sad = sad; best_dx = cdx; best_dy = cdy;
                    improved = true;
                }
            }
            iter++;
        }
    }
    void LDSP_SDSP_Search(const uint8_t* block_data, int block_stride, const ::Image& prev_frame,
        int x_start, int y_start, int block_h, int block_w,
        int predicted_dx, int predicted_dy,
        int& best_dx, int& best_dy, int& best_sad) {

        // --- 1. 初始化 ---
        // 先計算起始預測點 (Center) 的 SAD，作為比較的基準
        best_dx = predicted_dx;
        best_dy = predicted_dy;
        best_sad = std::numeric_limits<int>::max();

        auto safe_in = [&](int rx, int ry) {
            return (rx>=0 && ry>=0 && rx+block_w<=prev_frame.width() && ry+block_h<=prev_frame.height());
        };

        // 初始位置邊界檢查與 SAD 計算 (Center Point)
        {
            int ref_x = x_start + best_dx;
            int ref_y = y_start + best_dy;

            // 確保預測點本身沒有出界
            if (abs(best_dx) <= search_range && abs(best_dy) <= search_range && safe_in(ref_x, ref_y)) {
                best_sad = computeSAD(block_data, block_stride, block_w, block_h, prev_frame, ref_x, ref_y);
            }
        }

        // 定義 LDSP (大鑽石) 的 8 個周圍點 (不含中心，半徑=2)
        const int ldsp_offsets[][2] = {
            {0, -2}, {1, -1}, {2, 0}, {1, 1},
            {0, 2}, {-1, 1}, {-2, 0}, {-1, -1}
        };
        const int ldsp_count = 8;

        // 定義 SDSP (小鑽石) 的 4 個周圍點 (不含中心，半徑=1)
        const int sdsp_offsets[][2] = {
            {0, -1}, {1, 0}, {0, 1}, {-1, 0}
        };
        const int sdsp_count = 4;

        // --- 2. 階段一：LDSP (Large Diamond Search Pattern) ---
        int search_iterations = 0;
        const int max_iterations = 10;
        bool center_is_best = false;

        while (!center_is_best && search_iterations < max_iterations) {
            center_is_best = true;

            int next_center_dx = best_dx;
            int next_center_dy = best_dy;
            int local_min_sad = best_sad;

            for (int k = 0; k < ldsp_count; k++) {
                int candidate_dx = best_dx + ldsp_offsets[k][0];
                int candidate_dy = best_dy + ldsp_offsets[k][1];

                if (abs(candidate_dx) > search_range || abs(candidate_dy) > search_range)
                    continue;

                int ref_x = x_start + candidate_dx;
                int ref_y = y_start + candidate_dy;

                if (!safe_in(ref_x, ref_y))
                    continue;

                int sad = computeSAD(block_data, block_stride, block_w, block_h, prev_frame, ref_x, ref_y);

                if (sad < local_min_sad) {
                    local_min_sad = sad;
                    next_center_dx = candidate_dx;
                    next_center_dy = candidate_dy;
                    center_is_best = false; 
                }
            }

            if (!center_is_best) {
                best_dx = next_center_dx;
                best_dy = next_center_dy;
                best_sad = local_min_sad;
            }
            search_iterations++;
        }

        // --- 3. 階段二：SDSP (Small Diamond Search Pattern) ---
        for (int k = 0; k < sdsp_count; k++) {
            int candidate_dx = best_dx + sdsp_offsets[k][0];
            int candidate_dy = best_dy + sdsp_offsets[k][1];

            if (abs(candidate_dx) > search_range || abs(candidate_dy) > search_range)
                continue;

            int ref_x = x_start + candidate_dx;
            int ref_y = y_start + candidate_dy;

            if (!safe_in(ref_x, ref_y))
                continue;

            int sad = computeSAD(block_data, block_stride, block_w, block_h, prev_frame, ref_x, ref_y);

            if (sad < best_sad) {
                best_sad = sad;
                best_dx = candidate_dx;
                best_dy = candidate_dy;
            }
        }
    }
};

// ==================================================================================
//  Object Detection Logic (from C_V2.cpp)
// ==================================================================================

std::vector<MotionObject> extractMotionObjects(
    const std::vector<MotionVector>& blocks,
    const std::vector<bool>& changed_mask,
    int rows, int cols,
    float threshold,
    int searchRadius)
{
    std::vector<MotionObject> objs;
    std::vector<char> visited(rows * cols, 0);

    auto idx = [&](int r, int c) { return r * cols + c; };
    int objId = 0;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int start = idx(r, c);
            if(start >= (int)blocks.size()) continue;

            float mag = std::sqrt((float)blocks[start].dx * blocks[start].dx +
                                  (float)blocks[start].dy * blocks[start].dy);

            // CHANGED: Allow if Magnitude High OR in Changed Mask (Dilation)
            bool is_changed = false;
            if (start < (int)changed_mask.size()) is_changed = changed_mask[start];
            
            if (mag < threshold && !is_changed) continue;
            if (visited[start]) continue;

            MotionObject obj;
            obj.id = objId++;

            std::vector<int> stack = { start };
            visited[start] = 1;

            float sumDx = 0, sumDy = 0;
            int count = 0;
            int motion_count = 0; // NEW: Count only blocks with motion
            int sumR = 0, sumC = 0;

            while (!stack.empty()) {
                int cur = stack.back(); stack.pop_back();

                int cr = cur / cols;
                int cc = cur % cols;

                obj.blocks.push_back(cur);
                obj.block_motion_vectors.push_back(blocks[cur]); // NEW: store per-block vector
                
                // Track geometric stats (all blocks)
                sumR += cr;
                sumC += cc;
                count++;
                
                // Track motion stats (only significant blocks)
                if (std::abs(blocks[cur].dx) > 0.1f || std::abs(blocks[cur].dy) > 0.1f) {
                    sumDx += blocks[cur].dx;
                    sumDy += blocks[cur].dy;
                    motion_count++;
                }
                
                // Dynamic Search Radius
                for (int dy = -searchRadius; dy <= searchRadius; dy++) {
                    for (int dx = -searchRadius; dx <= searchRadius; dx++) {
                        if (dy == 0 && dx == 0) continue;

                        int nr = cr + dy, nc = cc + dx;
                        if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
                        
                        int ni = idx(nr, nc);
                        if(ni >= (int)visited.size()) continue; 
                        if (visited[ni]) continue;

                        float mag2 = std::sqrt((float)blocks[ni].dx * blocks[ni].dx +
                                             (float)blocks[ni].dy * blocks[ni].dy);
                        
                        bool is_changed2 = false;
                        if (ni < (int)changed_mask.size()) is_changed2 = changed_mask[ni];

                        if (mag2 < threshold && !is_changed2) continue;

                        visited[ni] = 1;
                        stack.push_back(ni);
                    }
                }
            }

            if(count > 0) {
                // Motion Average: Based on motion blocks only (avoid dilution by static blocks)
                if (motion_count > 0) {
                    obj.avgDx = sumDx / motion_count;
                    obj.avgDy = sumDy / motion_count;
                } else {
                    obj.avgDx = 0;
                    obj.avgDy = 0;
                }
                
                // Geometric Center: Based on ALL blocks (ROI)
                obj.centerX = sumC / (float)count;
                obj.centerY = sumR / (float)count;
                
                obj.strength = std::sqrt(obj.avgDx * obj.avgDx + obj.avgDy * obj.avgDy);
                
                // =========================================================
                // NEW: Variance Calculation (Stability Features)
                // =========================================================
                if (motion_count > 0) {
                    // 1. Magnitude Variance
                    float sum_mag_sq_diff = 0.0f;
                    float mean_mag = obj.strength; // Close enough to mean of magnitudes? 
                    // Actually obj.strength is magnitude of Mean Vector. Mean of Magnitudes is different.
                    // Let's recompute proper Mean Magnitude first.
                    float sum_mag = 0.0f;
                    // For Direction (Circular Stats)
                    float sum_sin = 0.0f;
                    float sum_cos = 0.0f;
                    int valid_vecs = 0;

                    for (const auto& mv : obj.block_motion_vectors) {
                        float dx = (float)mv.dx;
                        float dy = (float)mv.dy;
                        if (std::abs(dx) < 0.1f && std::abs(dy) < 0.1f) continue; // Skip static blocks for variance? 
                        // Or should we include them? If object is partly static, variance is high.
                        // Let's stick to "Motion Blocks" logic (motion_count).
                        
                        float mag = std::sqrt(dx*dx + dy*dy);
                        sum_mag += mag;
                        
                        // Angle
                        float ang = std::atan2(dy, dx);
                        
                        // Weighted by Magnitude
                        sum_sin += std::sin(ang) * mag;
                        sum_cos += std::cos(ang) * mag;
                        valid_vecs++;
                    }
                    
                    if (valid_vecs > 0) {
                        // Magnitude Variance
                        float avg_mag = sum_mag / valid_vecs;
                        for (const auto& mv : obj.block_motion_vectors) {
                            float dx = (float)mv.dx;
                            float dy = (float)mv.dy;
                            if (std::abs(dx) < 0.1f && std::abs(dy) < 0.1f) continue;
                            
                            float mag = std::sqrt(dx*dx + dy*dy);
                            sum_mag_sq_diff += (mag - avg_mag) * (mag - avg_mag);
                        }
                        obj.magnitude_variance = std::sqrt(sum_mag_sq_diff / valid_vecs);
                        
                        // Direction Variance (Weighted Circular)
                        // R_weighted = |Sum(mag * vec)| / Sum(mag)
                        // We accumulated sum_sin = sum(mag * sin), sum_cos = sum(mag * cos)
                        float R_weighted = 0.0f;
                        if (sum_mag > 0.0001f) { // Avoid div by zero
                             R_weighted = std::sqrt(sum_sin*sum_sin + sum_cos*sum_cos) / sum_mag;
                        }

                        // Circular Std Dev = sqrt( -2 * ln(R) )
                        if (R_weighted < 1.0f && R_weighted > 0.0f) {
                            obj.direction_variance = std::sqrt(-2.0f * std::log(R_weighted));
                        } else if (R_weighted >= 1.0f) {
                             obj.direction_variance = 0.0f; // Perfectly aligned
                        } else {
                             obj.direction_variance = 10.0f; // Max/Undefined
                        }
                    } else {
                         obj.magnitude_variance = 0.0f;
                         obj.direction_variance = 0.0f;
                    }
                } else {
                    obj.magnitude_variance = 0.0f;
                    obj.direction_variance = 0.0f; // Lower Analysis threshold to 0.0
                }

                obj.safe_area_ratio = 0.0f; // Calculated later
                objs.push_back(obj);
            }
        }
    }
    return objs;
}

// Helper to check point in polygon
inline float cross_product(float ax, float ay, float bx, float by, float cx, float cy) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

// Check if point p is inside convex quad defined by poly
bool isPointInConvexQuad(const std::vector<std::pair<float, float>>& poly, float px, float py) {
    if (poly.size() != 4) return false;
    bool has_pos = false, has_neg = false;
    for (int i = 0; i < 4; i++) {
        float cp = cross_product(poly[i].first, poly[i].second, 
                                 poly[(i+1)%4].first, poly[(i+1)%4].second, 
                                 px, py);
        if (cp > 0) has_pos = true;
        if (cp < 0) has_neg = true;
        if (has_pos && has_neg) return false;
    }
    return true;
}

void detectObjectTemporalMotion(
    const std::vector<std::vector<MotionObject>>& history,
    float movementThreshold,
    float strongThreshold,
    float accelerationThreshold,
    float safeAreaThreshold,
    int M, int N,
    float accelUpperThreshold,
    float accelLowerThreshold,
    const std::vector<std::pair<int, int>>& bed_region,
    std::vector<int>& outTriggeredIds,
    long long currentFrameIdx,
    std::string& outWarning,
    int frame_width, int frame_height,
    int grid_cols, int grid_rows,
    bool enable_bed_exit_verification, // NEW
    bool enable_block_shrink_verification // NEW
    )
{
    if (!enable_block_shrink_verification || !enable_bed_exit_verification)
    {
        //printf("[FallDetector] Block Shrink Verification and Bed Exit Verification are disabled.\n");
    }
    int T = (int)history.size();
    if (T < N) return;
    int start = (T > M) ? (T - M) : 0;

    // Pre-process Bed Region Poly (float) for efficiency
    std::vector<std::pair<float, float>> bed_poly;
    bool has_bed_poly = false;
    if (bed_region.size() == 4) {
        for(auto& p : bed_region) bed_poly.push_back({(float)p.first, (float)p.second});
        has_bed_poly = true;
        //printf("has_bed_poly = True\n");
    }

    // Track how many times each object triggers a "Fall Signal" in the window
    std::map<int, int> object_signal_counts;

    // =================================================================================
    // STEP 1: Traverse Window to Count Fall Signals (N-out-of-M)
    // =================================================================================
    for (int f = start; f < T; f++) 
    {
        long long actual_frame_num = currentFrameIdx - (T - 1 - f);
        for (const auto& obj : history[f]) 
        {
            if (f == (T-1))
            {
                printf("[Debug] actual_frame_num %d, obj.id %d, obj.strength %f \n", actual_frame_num, obj.id, obj.strength);
            }
            // Basic Conditions
            bool is_strong_fall = (obj.strength >= strongThreshold && obj.blocks.size() > 0);
            bool is_accel_fall = (obj.acceleration > accelerationThreshold);
            bool is_larger_safe_area = (obj.safe_area_ratio > safeAreaThreshold);
            
            // Accel Change
            bool is_accel_change_fall = false;
            float prev_accel = 0.0f;
            if (f > 0) {
                 for(const auto& prev : history[f-1]) {
                     if(prev.id == obj.id) {
                         if (prev.acceleration > accelUpperThreshold && obj.acceleration < accelLowerThreshold) 
                             is_accel_change_fall = true;
                         break;
                     }
                 }
            }

            // PRIMARY TRIGGER: Any strong momentum or acceleration event
            bool is_primary_signal = false;
            
            // We maintain the filter that the fall must be valid (Vertical Velocity > 1.5, Outside Safe Area)
            // But we allow "Bed Exit" which transitions from Safe->Unsafe.
            // Let's stick to the core requirement: 
            // "Is there N frames of FALL EVIDENCE?"
            // Evidence = (Strong || HighAccel || AccelChange) AND (VerticalVelocity > 1.5)
            
            // Note: obj.avgDy is the vertical velocity. Defaults checking > 1.5
            // if (obj.avgDy > 1.5f) { // Re-enabled filter
            
            if (!is_larger_safe_area || is_strong_fall) { // Allow Strong Fall even if seemingly safely inside? Or NO?
                 // Usually falls happen outside.
                 if (!is_larger_safe_area) {
                     if (is_strong_fall || is_accel_fall || is_accel_change_fall) 
                     {
                        //printf("frame id %d, obj.strength %f, obj.acceleration %f, obj.avgDy %f\n", actual_frame_num, obj.strength, obj.acceleration, obj.avgDy);
                        is_primary_signal = true;
                     }
                 }
            }
            //}

            if (is_primary_signal) {
                object_signal_counts[obj.id]++;
                // printf("DEBUG: Frame %lld Obj %d Signal Count %d\n", actual_frame_num, obj.id, object_signal_counts[obj.id]);
            }
        }
    }

    // =================================================================================
    // STEP 2: Secondary Verification (History Check) for Candidates with Count >= N
    // =================================================================================
    for (auto const& [id, count] : object_signal_counts) 
    {
        if (count >= N) 
        {
            if (!enable_block_shrink_verification || !enable_bed_exit_verification) 
            {
                outTriggeredIds.push_back(id);
                //outWarning += "Fall Confirmed: N/M + Verified. ";
                //printf("[FallDetector] CONFIRMED FALL Obj %d (Count %d/%d). BedExit:%d Shrink:%d (Frame %lld)\n", 
                //        id, count, N, condition_bed_exit, condition_block_shrink, currentFrameIdx);
                continue;
            }


             // Met Primary N-out-of-M condition.
             bool condition_bed_exit = false;
             bool condition_block_shrink = false;
             
             // Scan this object's history in the window to find evidence
             bool was_inside = false;
             bool is_outside_now = false; // "Now" implies end of window or specific transition
             
             float max_blocks = 0;
             int max_blocks_frame = 0;
             float current_blocks = 0;
             float max_avg_dy = -999.0f; // Track max downward velocity
             
             // To be robust, let's find the Object's state at the END (T-1) and ANY START point
             // Actually, Bed Exit is a transition.
             
             // Scan Pass
             for(int f = start; f < T; f++) 
             {
                long long actual_frame_num = currentFrameIdx - (T - 1 - f);
                 for(const auto& obj : history[f]) 
                 {
                     printf("[Debug] Obj %d Strength %.2f Center (%.1f, %.1f) Size %zu\n", obj.id, obj.strength, obj.centerX, obj.centerY, obj.blocks.size());
                     if (obj.id == id) 
                     {
                         // Track Max Dy (Downward is Positive)
                         if (obj.avgDy > max_avg_dy) max_avg_dy = obj.avgDy;

                         float px = (obj.centerX + 0.5f) * (frame_width / (float)grid_cols);
                         float py = (obj.centerY + 0.5f) * (frame_height / (float)grid_rows);
                         //printf("[Algo] Frame %d, obj.id : %d, obj.Center(Blk): %.2f,%.2f -> (Pix): %.2f,%.2f\n", currentFrameIdx - (T-1-f), obj.id, obj.centerX, obj.centerY, px, py);
                         if (f==(T-1))
                         {
                            if (has_bed_poly) 
                            {
                                // Use Centroid check
                                if (isPointInConvexQuad(bed_poly, px, py)) 
                                {
                                    //was_inside = true;
                                } 
                                else 
                                {
                                    is_outside_now = true;
                                    //printf("[%d] is_outside_now: %d (Frame %lld)\n", f, is_outside_now, currentFrameIdx);
                                }
                            } 
                            else 
                            {
                                // Fallback to Safe Area Ratio
                                //if (obj.safe_area_ratio > 0.5f) was_inside = true;
                                if (obj.safe_area_ratio < 0.3f) is_outside_now = true; 
                            }
                        }
                        else
                        {
                            if (has_bed_poly) 
                            {
                                // Use Centroid check
                                if (isPointInConvexQuad(bed_poly, px, py)) 
                                {
                                    was_inside = true;
                                    //printf("[%d] was_inside: %d (Frame %lld)\n", f, was_inside, currentFrameIdx);
                                } 
                                else 
                                {
                                    //is_outside_now = true;
                                }
                            } 
                            else 
                            {
                                // Fallback to Safe Area Ratio
                                if (obj.safe_area_ratio > 0.5f) was_inside = true;
                                //if (obj.safe_area_ratio < 0.3f) is_outside_now = true; 
                            }
                        }
                        if ((float)obj.blocks.size() > max_blocks) 
                        {
                            max_blocks = (float)obj.blocks.size();
                            max_blocks_frame = actual_frame_num;
                            //printf("[Debug] max_blocks %f\n", max_blocks);
                        }
                        current_blocks = (float)obj.blocks.size(); // Keeps updating to latest
                        //printf("[Debug] current_blocks %f\n", current_blocks); 
                     }
                     
                 }
             }
             
             // Check Bed Exit (Must have been inside, and is now effectively outside)
             // We can check if "is_outside_now" (at some point or end) AND "was_inside"
             // AND check if there was significant downward motion (Fall) vs just walking (Horizontal)
             // Check Bed Exit (Must have been inside, and is now effectively outside)
             // AND check if there was significant downward motion (Fall) vs just walking (Horizontal)
             // Check Bed Exit (Must have been inside, and is now effectively outside)
             if (was_inside && is_outside_now)
             {
                 bool confirmed = false;
                 if (enable_bed_exit_verification) {
                    if (max_avg_dy > 2.0f) {
                        confirmed = true;
                        printf("[FallDetector] Bed Exit CONFIRMED for Obj %d (MaxDy: %.2f). Frame %lld\n", id, max_avg_dy, currentFrameIdx);
                    } else {
                        printf("[FallDetector] Bed Exit IGNORED for Obj %d (MaxDy: %.2f < 2.0). Frame %lld\n", id, max_avg_dy, currentFrameIdx);
                    }
                 } else {
                     // Verification Disabled: Trust the primary signal + region transition
                     confirmed = true;
                     printf("[FallDetector] Bed Exit (No Verify) for Obj %d. Frame %lld\n", id, currentFrameIdx);
                 }
                 
                 if (confirmed) condition_bed_exit = true;
             }
             
             // Check Block Shrink
             // Check Block Shrink
             if (max_blocks > 0 && current_blocks > 0) 
             {
                 float ratio = current_blocks / max_blocks;
                 if (ratio < 0.4f) 
                 {
                    bool confirmed = false;
                    if (enable_block_shrink_verification) {
                        if (max_avg_dy > 2.0f) {
                            confirmed = true;
                            printf("[FallDetector] Block Shrink CONFIRMED (Ratio %.2f, MaxDy %.2f)\n", ratio, max_avg_dy);
                        } else {
                            printf("[FallDetector] Block Shrink IGNORED (Ratio %.2f, MaxDy %.2f < 2.0)\n", ratio, max_avg_dy);
                        }
                    } else {
                        // Verification Disabled
                        confirmed = true;
                        printf("[FallDetector] Block Shrink (No Verify) (Ratio %.2f)\n", ratio);
                    }
                    if (confirmed) condition_block_shrink = true;
                 }
             }
             
             // FINAL DECISION
             if (condition_bed_exit || condition_block_shrink) {
                 outTriggeredIds.push_back(id);
                 outWarning += "Fall Confirmed: N/M + Verified. ";
                 printf("[FallDetector] CONFIRMED FALL Obj %d (Count %d/%d). BedExit:%d Shrink:%d (Frame %lld)\n", 
                        id, count, N, condition_bed_exit, condition_block_shrink, currentFrameIdx);
             }
        }
    }
}

// NEW Function for Pixel-Based Fall Detection (User Request)
void detectFallPixelStats(
    const std::vector<std::vector<MotionObject>>& history,
    int N, 
    long long currentFrameIdx,
    std::vector<int>& outTriggeredIds,
    std::string& outWarning
) 
{
    // Window size Check
    int T = (int)history.size();
    if (T < N) return;
    int start = (T > N) ? (T - N) : 0;

    // Check each object present in current frame (or very recent)
    // We iterate backwards from T-1
    if (history.empty()) return;
    
    // Collect stats for objects in the window
    std::map<int, float> max_pixels;
    std::map<int, float> max_brightness;
    
    std::map<int, int> drop_streak;
    std::map<int, int> static_streak;
    std::map<int, bool> fall_phase_confirmed;

    // 1. Traverse Window (Old -> New) to find trends
    printf("[Debug] detectFallPixelStats Loop Start T=%d\n", T);
    for (int f = start; f < T; f++) 
    {
        printf("[Debug] detectFallPixelStats - Frame %d\n", f);
        for (const auto& obj : history[f]) 
        {
            int id = obj.id;
            
            // Calculate Moving Averages for Trend Analysis
            // Pre-Window: [f-5, f-1] (5 frames)
            // Post-Window: [f, f+4] (5 frames)
            int win_pre = 5;
            int win_post = 5;
            
            float pre_sum_pix = 0;
            float pre_sum_bri = 0;
            int pre_count = 0;
            
            if ((f == T-1) && (obj.pixel_count > 1200))
            {
                printf("[Debug] currentFrameIdx %lld, id %d, obj.pixel_count %d, obj.avg_brightness %f\n", currentFrameIdx, id, obj.pixel_count, obj.avg_brightness);
                printf("[DDD] obj.acceleration %f, obj.strength %f, obj.centerX %f\n", obj.acceleration, obj.strength, obj.centerX);
            }

            for (int k = 1; k <= win_pre; ++k) {
                int pre_idx = f - k;
                if (pre_idx >= 0 && pre_idx < (int)history.size()) {
                    for (const auto& o : history[pre_idx]) {
                        if (o.id == id) {
                            pre_sum_pix += o.pixel_count;
                            pre_sum_bri += o.avg_brightness;
                            pre_count++;
                            break;
                        }
                    }
                }
            }
            
            float post_sum_pix = 0;
            float post_sum_bri = 0;
            int post_count = 0;
            
            for (int k = 0; k < win_post; ++k) {
                int post_idx = f + k;
                if (post_idx < (int)history.size()) {
                     for (const auto& o : history[post_idx]) {
                        if (o.id == id) {
                            post_sum_pix += o.pixel_count;
                            post_sum_bri += o.avg_brightness;
                            post_count++;
                            break;
                        }
                    }
                }
            }
            
            bool is_drop = false;
            
            if (pre_count >= 3 && post_count >= 1) { // Require sufficient baseline
                float pre_avg_pix = pre_sum_pix / pre_count;
                float pre_avg_bri = pre_sum_bri / pre_count;
                float post_avg_pix = post_sum_pix / post_count;
                float post_avg_bri = post_sum_bri / post_count;
                
                // Debug Printing (Optional - Remove later)
                //printf("currentFrameIdx %d, ID %d F %d: Pre(%.1f, %.1f) Post(%.1f, %.1f)\n", currentFrameIdx, id, f, pre_avg_pix, pre_avg_bri, post_avg_pix, post_avg_bri);

                if (pre_avg_pix > 50 && pre_avg_bri > 20.0f) {
                    if (post_avg_pix < (pre_avg_pix * 0.6f) 
                    //&& post_avg_bri < (pre_avg_bri * 0.8f)
                    ) 
                    {
                        is_drop = true;
                        //printf("[FallDetector!!!] Trend Drop Detected ID %d Frame %lld, F %d, pre_avg_pix %.1f, post_avg_pix %.1f, pre_avg_bri %.1f, post_avg_bri %.1f\n", id, currentFrameIdx, f, pre_avg_pix, post_avg_pix, pre_avg_bri, post_avg_bri);
                    }
                }
            }
            
            if (is_drop) {
                drop_streak[id]++;
            } 
            else 
            {
                 // If not dropping (values high), reset streak? 
                 // Yes, we want a *sustained* drop.
                 drop_streak[id]--;
            }
            
            // Confirm Fall Phase if drop persists ~10 frames
            if (drop_streak[id] >= 10) {
                fall_phase_confirmed[id] = true;
                printf("[Fall Detect!!!] Fall Phase Confirmed ID %d Frame %lld\n", id, currentFrameIdx);
            }
            
            // Check Static Phase (Post-Fall)
            // Condition: Low Strength AND Low Acceleration
            // Only counts if we have confirmed a fall phase (or are currently in it)
            bool is_static = true;//(obj.strength < 2.0f && std::abs(obj.acceleration) < 1.0f);
            
            if (fall_phase_confirmed[id] && is_static) {
                 static_streak[id]++;
            } 
            //else {
            //     static_streak[id] = 0;
           // }
        }
    }
    printf("[Debug] detectFallPixelStats Loop End.\n");
    
    // 2. Final Decision (Trigger if Static Phase persists > 10 frames)
    for (auto const& [id, count] : static_streak) {
        if (count >= 1) {
             // Check if it's "Current" object (must be in last frame or close)
             bool is_current = false;
             if(!history.empty()) 
             {
                for(auto& o : history.back()) 
                {
                    printf("[Debug static_streak] Obj %d in history Frame %lld\n", o.id, currentFrameIdx);
                    if(o.id == id) is_current = true;
                }
             }
             
             if(is_current) {
                 outTriggeredIds.push_back(id);
                 outWarning += "Fall Confirmed (Pixel Stats: 10+10). ";
                 printf("[FallDetector] CONFIRMED FALL (Pixel Stats 10+10) Obj %d. MaxPix:%.0f MaxBri:%.1f (Frame %lld)\n", 
                        id, max_pixels[id], max_brightness[id], currentFrameIdx);
             }
        }
    }

    // 3. NEW: Pixel Decay Analysis (Static Fall Detection)
    if (!history.empty()) {
        const auto& current_objs = history.back();
        int history_sz = history.size();
        
        for (const auto& curr_obj : current_objs) {
            int id = curr_obj.id;
            
            // Skip if already triggered by other logic
            bool already_triggered = false;
            for(int tid : outTriggeredIds) if(tid == id) already_triggered = true;
            if(already_triggered) continue;

            // 3.1 Gather History for this ID
            // 3.1 Gather History & Find Peak
            float max_pix = 0;
            int max_pix_idx = -1; // Index in history
            float peak_cx = 0, peak_cy = 0; // Centroid at Peak
            
            // float min_pix = 9999999; // Unused
            
            float sum_cx = 0, sum_cy = 0;
            int count = 0;
            
            // Check last 40 frames
            int lookback = 40;
            int start_idx = std::max(0, history_sz - lookback);

            for (int i = start_idx; i < history_sz; ++i) {
                for (const auto& o : history[i]) {
                    if (o.id == id) {
                        if (o.pixel_count > max_pix) {
                            max_pix = (float)o.pixel_count;
                            max_pix_idx = i;
                            peak_cx = o.centerX;
                            peak_cy = o.centerY;
                        }
                        
                        sum_cx += o.centerX;
                        sum_cy += o.centerY;
                        count++;
                        break;
                    }
                }
            }
            
            if (count < 10) continue; // Need some history
            
            // float avg_cx = sum_cx / count;
            // float avg_cy = sum_cy / count;
            float curr_pix = (float)curr_obj.pixel_count;
            
            // 3.2 Criteria
            // A. Significant Decay: Current < 50% of Peak
            bool is_decay = (curr_pix < max_pix * 0.5f);
            
            // B. Interval Speed (Movement DURING Decay)
            // Calculate avg speed from Peak Frame to Current Frame
            float interval_speed = 0.0f;
            if (is_decay && max_pix_idx >= 0 && max_pix_idx < history_sz - 1) {
                float dx = curr_obj.centerX - peak_cx;
                float dy = curr_obj.centerY - peak_cy;
                float dist = std::sqrt(dx*dx + dy*dy);
                int frames_diff = (history_sz - 1) - max_pix_idx; 
                // frames_diff is basically valid frames count since peek. 
                // history.back() is current frame.
                
                if (frames_diff > 0) {
                    interval_speed = dist / (float)frames_diff;
                }
            }
            
            // Threshold: 0.3 grid blocks per frame?
            // If static fading, speed should be very low (< 0.1).
            // If falling/moving away, speed > 0.3.
            bool is_moving_shrink = (interval_speed > 0.3f); 

            // C. Not At Edge (Safe Margin)
            bool not_at_edge = (curr_obj.centerX > 1.0f && curr_obj.centerX < (12.0f - 1.0f) &&
                                curr_obj.centerY > 1.0f && curr_obj.centerY < (16.0f - 1.0f)); 
             
             // Combined Trigger
             if (is_decay && is_moving_shrink && not_at_edge && max_pix > 2000) {
                 outTriggeredIds.push_back(id);
                 outWarning += "Pixel Decay Detected (Structural Fall). ";
                 printf("[FallDetector] CONFIRMED FALL (Structural Decay) Obj %d. Peak:%.0f Curr:%.0f IntSpd:%.3f (Frame %lld)\n", 
                        id, max_pix, curr_pix, interval_speed, currentFrameIdx);
             }
        }
    }

}


// ==================================================================================
//  FallDetector::Impl
// ==================================================================================




// Create Trapezoid Mask
::Image createTrapezoidMask(int width, int height, const std::vector<std::pair<float, float>>& points) {
    ::Image mask(width, height, 1);
    // Fill with 255 (Background / Safe Area?) -> Logic from C_V2.cpp says:
    // "createTrapezoidMask... mask.at(x,y) = 0" for inside trapezoid?
    // But we don't want to break signature too much.
    // Actually, let's just make it a method of Impl or pass it.
    // For now, I will skip detailed pixel check inside this loop and do it AFTER.
    // Or better, update function signature.
    // C_V2.cpp: inside trapezoid => 0 (mask). 
    // Wait, let's check usage in C_V2.cpp:
    // binary_diff.at(x,y) = (...) & bedRegion.at(x, y);
    // If inside bed is SAFER/IGNORED, then it should be 0.
    // If inside bed is where we CARE, it should be 255.
    // C_V2.cpp comment: "在床的區域內會被刪掉" (Deleted in bed region).
    // So bed region = 0, Outside = 255.
    // This means we detect motion OUTSIDE the bed.
    
    // Initialize 255
    uint8_t* data = mask.getData();
    std::fill(data, data + width * height, 255);

    if (points.size() != 4) return mask;

    int minX = width, maxX = 0, minY = height, maxY = 0;
    for (const auto& p : points) {
        if (p.first < minX) minX = std::max(0, (int)p.first);
        if (p.first > maxX) maxX = std::min(width - 1, (int)p.first);
        if (p.second < minY) minY = std::max(0, (int)p.second);
        if (p.second > maxY) maxY = std::min(height - 1, (int)p.second);
    }

    for (int y = minY; y <= maxY; ++y) {
        for (int x = minX; x <= maxX; ++x) {
            if (isPointInConvexQuad(points, x, y)) {
                // Inside bed -> 0
                mask.at(x, y) = 0;
            }
        }
    }
    return mask;
}

// ==================================================================================
//  Pyramid LK Helper Structures & Functions
// ==================================================================================

namespace LK_Utils {

    // Simple image for signed 16-bit gradients
    struct ShortImage {
        int w, h;
        std::vector<int16_t> data;
        
        ShortImage() : w(0), h(0) {}
        ShortImage(int _w, int _h) : w(_w), h(_h) {
            data.resize(w * h, 0);
        }
    };

    struct ImagePyramid {
        std::vector<::Image> levels;
        std::vector<ShortImage> Ix_levels;
        std::vector<ShortImage> Iy_levels;
    };

    // NEON-optimized Sobel Gradient
    void computeGradients(const ::Image& img, ShortImage& Ix, ShortImage& Iy) {
        int w = img.width();
        int h = img.height();
        
        if (Ix.w != w || Ix.h != h) Ix = ShortImage(w, h);
        if (Iy.w != w || Iy.h != h) Iy = ShortImage(w, h);
        
        const uint8_t* src = img.getData();
        int16_t* dx = Ix.data.data();
        int16_t* dy = Iy.data.data();
        
        // Zero borders
        // (Simplified: just zero out for now, can be optimized)
        std::fill(dx, dx + w, 0); 
        std::fill(dx + (h - 1) * w, dx + h * w, 0);
        std::fill(dy, dy + w, 0);
        std::fill(dy + (h - 1) * w, dy + h * w, 0);

        for (int y = 1; y < h - 1; ++y) {
            int x = 1;
            
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
            // NEON Implementation
            for (; x <= w - 1 - 16; x += 16) {
                // Load 3 rows: top, mid, bot
                // We need 16+2 pixels for the window (x-1 to x+16)
                // Load 2 v128 (16 bytes) + overlap? 
                // Using vld1q_u8 loads 16 bytes.
                // To get neighbors, we can load unaligned or use tbl?
                // Unaligned load is easiest on modern ARM.
                
                int top_idx = (y - 1) * w + x;
                int mid_idx = (y) * w + x;
                int bot_idx = (y + 1) * w + x;
                
                // Load p(x-1) .. p(x+16)
                // We need [x-1, x+15] ? No, for 16 results at x..x+15, we need inputs x-1..x+16
                // So 18 pixels.
                
                uint8x16_t t_m1 = vld1q_u8(src + top_idx - 1);
                uint8x16_t t_p1 = vld1q_u8(src + top_idx + 1);
                
                uint8x16_t m_m1 = vld1q_u8(src + mid_idx - 1); // For Ix: -2*src(x-1)
                uint8x16_t m_p1 = vld1q_u8(src + mid_idx + 1); // For Ix: +2*src(x+1)
                
                uint8x16_t b_m1 = vld1q_u8(src + bot_idx - 1);
                uint8x16_t b_p1 = vld1q_u8(src + bot_idx + 1);
                
                // Also need center col for Iy
                uint8x16_t t_0 = vld1q_u8(src + top_idx);
                uint8x16_t b_0 = vld1q_u8(src + bot_idx);

                // Convert to u16
                uint16x8_t t_m1_L = vmovl_u8(vget_low_u8(t_m1));
                uint16x8_t t_m1_H = vmovl_u8(vget_high_u8(t_m1));
                uint16x8_t t_p1_L = vmovl_u8(vget_low_u8(t_p1));
                uint16x8_t t_p1_H = vmovl_u8(vget_high_u8(t_p1));
                
                uint16x8_t m_m1_L = vmovl_u8(vget_low_u8(m_m1));
                uint16x8_t m_m1_H = vmovl_u8(vget_high_u8(m_m1));
                uint16x8_t m_p1_L = vmovl_u8(vget_low_u8(m_p1));
                uint16x8_t m_p1_H = vmovl_u8(vget_high_u8(m_p1));

                uint16x8_t b_m1_L = vmovl_u8(vget_low_u8(b_m1));
                uint16x8_t b_m1_H = vmovl_u8(vget_high_u8(b_m1));
                uint16x8_t b_p1_L = vmovl_u8(vget_low_u8(b_p1));
                uint16x8_t b_p1_H = vmovl_u8(vget_high_u8(b_p1));
                
                // Ix = (t_p1 + 2*m_p1 + b_p1) - (t_m1 + 2*m_m1 + b_m1)
                // Low 8
                uint16x8_t sum_p_L = vaddq_u16(t_p1_L, vaddq_u16(vshlq_n_u16(m_p1_L, 1), b_p1_L));
                uint16x8_t sum_m_L = vaddq_u16(t_m1_L, vaddq_u16(vshlq_n_u16(m_m1_L, 1), b_m1_L));
                int16x8_t ix_L = vsubq_s16(vreinterpretq_s16_u16(sum_p_L), vreinterpretq_s16_u16(sum_m_L));

                // High 8
                uint16x8_t sum_p_H = vaddq_u16(t_p1_H, vaddq_u16(vshlq_n_u16(m_p1_H, 1), b_p1_H));
                uint16x8_t sum_m_H = vaddq_u16(t_m1_H, vaddq_u16(vshlq_n_u16(m_m1_H, 1), b_m1_H));
                int16x8_t ix_H = vsubq_s16(vreinterpretq_s16_u16(sum_p_H), vreinterpretq_s16_u16(sum_m_H));
                
                vst1q_s16(dx + (y * w + x), ix_L);
                vst1q_s16(dx + (y * w + x) + 8, ix_H);

                // Iy = (b_m1 + 2*b_0 + b_p1) - (t_m1 + 2*t_0 + t_p1)
                uint16x8_t t_0_L = vmovl_u8(vget_low_u8(t_0));
                uint16x8_t t_0_H = vmovl_u8(vget_high_u8(t_0));
                uint16x8_t b_0_L = vmovl_u8(vget_low_u8(b_0));
                uint16x8_t b_0_H = vmovl_u8(vget_high_u8(b_0));

                uint16x8_t sum_b_L = vaddq_u16(b_m1_L, vaddq_u16(vshlq_n_u16(b_0_L, 1), b_p1_L));
                uint16x8_t sum_t_L = vaddq_u16(t_m1_L, vaddq_u16(vshlq_n_u16(t_0_L, 1), t_p1_L));
                int16x8_t iy_L = vsubq_s16(vreinterpretq_s16_u16(sum_b_L), vreinterpretq_s16_u16(sum_t_L));

                uint16x8_t sum_b_H = vaddq_u16(b_m1_H, vaddq_u16(vshlq_n_u16(b_0_H, 1), b_p1_H));
                uint16x8_t sum_t_H = vaddq_u16(t_m1_H, vaddq_u16(vshlq_n_u16(t_0_H, 1), t_p1_H));
                int16x8_t iy_H = vsubq_s16(vreinterpretq_s16_u16(sum_b_H), vreinterpretq_s16_u16(sum_t_H));

                vst1q_s16(dy + (y * w + x), iy_L);
                vst1q_s16(dy + (y * w + x) + 8, iy_H);
            }
#endif
            // Scalar fallback/cleanup
            for (; x < w - 1; ++x) {
                int cur = y * w + x;
                int ix = (int)src[cur + 1 - w] - (int)src[cur - 1 - w] +
                         2 * ((int)src[cur + 1] - (int)src[cur - 1]) +
                         (int)src[cur + 1 + w] - (int)src[cur - 1 + w];
                         
                int iy = (int)src[cur - 1 + w] + 2 * (int)src[cur + w] + (int)src[cur + 1 + w] -
                        ((int)src[cur - 1 - w] + 2 * (int)src[cur - w] + (int)src[cur + 1 - w]);
                
                dx[cur] = (int16_t)ix;
                dy[cur] = (int16_t)iy;
            }
            
            // Fix Right Border
            dx[y * w + w - 1] = 0;
            dy[y * w + w - 1] = 0;
        }
    }
    
    // Downsample 2x2
    void downsample(const ::Image& src, ::Image& dst) {
        int w = src.width();
        int h = src.height();
        int nw = w / 2;
        int nh = h / 2;
        if (dst.width() != nw || dst.height() != nh || dst.getChannels() != 1) {
            // Need a way to resize/create. Image class doesn't have realloc in public?
            // "Image result(new_w, new_h, channels)" - we can assign.
            // But passed as reference.
            // Just assume dst is correct size or we can't resize it in place efficiently without assignment.
            // Actually Image has assignments.
             dst = ::Image(nw, nh, 1);
        }
        
        const uint8_t* val = src.getData();
        uint8_t* dval = dst.getData();
        
        for(int y=0; y<nh; ++y) {
            for(int x=0; x<nw; ++x) {
                int sx = x*2;
                int sy = y*2;
                // Simple block average
                int sum = val[sy*w + sx] + val[sy*w + sx+1] + 
                          val[(sy+1)*w + sx] + val[(sy+1)*w + sx+1];
                dval[y*nw + x] = (uint8_t)((sum + 2) >> 2);
            }
        }
    }

    void buildPyramid(const ::Image& img, ImagePyramid& pyr, int levels) {
        pyr.levels.clear();
        pyr.Ix_levels.clear();
        pyr.Iy_levels.clear();
        
        // Base level
        pyr.levels.push_back(img.clone()); // Assuming we might want to keep original or just ref? clone safe.
        pyr.Ix_levels.resize(levels);
        pyr.Iy_levels.resize(levels);
        
        computeGradients(pyr.levels[0], pyr.Ix_levels[0], pyr.Iy_levels[0]);
        
        for(int l=1; l<levels; ++l) {
            ::Image nextL;
            downsample(pyr.levels.back(), nextL);
            pyr.levels.push_back(nextL);
            computeGradients(nextL, pyr.Ix_levels[l], pyr.Iy_levels[l]);
        }
    }
    void computeLK_Level(const ImagePyramid& prev_pyr, const ImagePyramid& curr_pyr, 
                         int level, 
                         float prev_x, float prev_y, 
                         float& curr_x, float& curr_y) 
    {
        const ::Image& prev_img = prev_pyr.levels[level];
        const ::Image& curr_img = curr_pyr.levels[level];
        const ShortImage& Ix_img = prev_pyr.Ix_levels[level];
        const ShortImage& Iy_img = prev_pyr.Iy_levels[level];
        
        int w = prev_img.width();
        int h = prev_img.height();
        
        int ix = (int)prev_x;
        int iy = (int)prev_y;
        
        // Win 3 -> 7x7. Boundary Check: Need +3/-3 for win, +1 for bilinear => 4
        if (ix < 4 || iy < 4 || ix >= w - 4 || iy >= h - 4) return;

        // 1. Compute Spatial Gradient Matrix G (Sum(Ix^2)...) AND mismatch setup
        // We can precompute this since the window on Prev is fixed.
        int32_t Gxx = 0, Gxy = 0, Gyy = 0;
        
        const int win_r = 3;
        
        // NEON Acc
        
        for (int dy = -win_r; dy <= win_r; ++dy) {
            int row_offset = (iy + dy) * w + ix;
            const int16_t* ptr_Ix = Ix_img.data.data() + row_offset;
            const int16_t* ptr_Iy = Iy_img.data.data() + row_offset;
            
            // Process 7 pixels. Vector 8 (last one masked or unused?)
            // Just use scalar loop for 7 pixels or vector logic? 
            // 7 is small. But we can load 8 and mask.
            // Or just scalar for G matrix since it's only once per point.
            // Let's optimize robustly.
            
            for (int dx = -win_r; dx <= win_r; ++dx) {
                int16_t valIx = ptr_Ix[dx];
                int16_t valIy = ptr_Iy[dx];
                Gxx += valIx * valIx;
                Gxy += valIx * valIy;
                Gyy += valIy * valIy;
            }
        }
        
        // Invert G
        // det = Gxx*Gyy - Gxy*Gxy
        float det = (float)((double)Gxx * Gyy - (double)Gxy * Gxy);
        if (det < 1.0f) return; // Singular
        
        float inv_det = 1.0f / det;
        // invG = [Gyy -Gxy; -Gxy Gxx] * inv_det
        
        // 2. Iterations
        for(int k=0; k<10; ++k) {
            // Compute b vector: Sum(Ix * dt), Sum(Iy * dt)
            // dt = Curr(interp) - Prev(exact)
            // We need to interpolate Curr at (curr_x + dx, curr_y + dy)
            // Iterate window again.
            
            float bx = 0, by = 0;
            
            if (curr_x < 3.0f || curr_y < 3.0f || curr_x >= w - 4.0f || curr_y >= h - 4.0f) break;

            int cx_i = (int)curr_x;
            int cy_i = (int)curr_y;
            float fx = curr_x - cx_i;
            float fy = curr_y - cy_i;
            
            // Bilinear weights
            // For NEON: w00, w01, w10, w11
            // We iterate dy from -3 to 3
            // For each row in window:
            //   Prev Row is (iy+dy, ix-3..ix+3)
            //   Curr Row is (cy_i+dy, cx_i-3..cx_i+3) PLUS bilinear (+1 row, +1 col)
            //   i.e. we access Curr at Y_row and Y_row+1.
            
            // Optimization: Load 8 pixels from Curr Row, interpolating X using NEON (vdup fx)
            // Then interpolate Y.
            
            // Let's allow SCALAR for now inside iteration to guarantee correctness first, 
            // as this is complex to vectorize without mistakes in blind coding.
            // The G-matrix vectorization above was skipped, so performance gain comes mainly from Pyramid (less iterations needed at fine scale usually)
            // But user asked for optimized NEON.
            // I will implement a simplified NEON loop for the 7 pixels row mismatch.
            
            // Precompute weights for current iteration
            float w00 = (1.0f - fy) * (1.0f - fx);
            float w01 = (1.0f - fy) * fx;
            float w10 = fy * (1.0f - fx);
            float w11 = fy * fx;
            
            // Scale by 256 for integer SIMD? Or use float32x4?
            // Image data is u8. Ix/Iy are s16.
            // Result b is float (accumulated).
            // Let's use float32x4 NEON.
            
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
            float32x4_t vBox = vdupq_n_f32(0);
            float32x4_t vBoy = vdupq_n_f32(0);
            
            float32x4_t vW00 = vdupq_n_f32(w00);
            float32x4_t vW01 = vdupq_n_f32(w01);
            float32x4_t vW10 = vdupq_n_f32(w10);
            float32x4_t vW11 = vdupq_n_f32(w11);
            
            for (int dy = -win_r; dy <= win_r; ++dy) {
                // Pointers
                const uint8_t* pPrev = prev_img.getData() + (iy + dy) * w + ix - win_r;
                const int16_t* pIx   = Ix_img.data.data() + (iy + dy) * w + ix - win_r;
                const int16_t* pIy   = Iy_img.data.data() + (iy + dy) * w + ix - win_r;
                
                const uint8_t* pCurrRow0 = curr_img.getData() + (cy_i + dy) * w + cx_i - win_r;
                const uint8_t* pCurrRow1 = curr_img.getData() + (cy_i + dy + 1) * w + cx_i - win_r; // for Y interp
                
                // We handle 7 pixels. 
                // Load 8 (safe since we have border)
                uint8x8_t vPrev = vld1_u8(pPrev);
                
                // Current Rows: Need TR/BR (offset +1). 
                // Load 8 from offset 0, 8 from offset 1 -> or just load 16 unaligned and pick?
                // Load 8 is fine. pCurrRow0[0..7] covers -3..+4. logic needs 0..1 (offset).
                // Wait, if dx goes -3 to +3. p[dx] and p[dx+1].
                // So index 0..6 need neighbors 1..7.
                // Loading 8 pixels (0..7) gives neighbors for 0..6!
                
                uint8x8_t vC00 = vld1_u8(pCurrRow0);    // Top-Lefts
                uint8x8_t vC01 = vld1_u8(pCurrRow0+1);  // Top-Rights
                uint8x8_t vC10 = vld1_u8(pCurrRow1);    // Bot-Lefts
                uint8x8_t vC11 = vld1_u8(pCurrRow1+1);  // Bot-Rights
                
                // Convert to float
                // Use vmovl to u16 then vcvt to f32?
                // Only need low 8 bytes (actually only first 7 lanes used).
                
                uint16x8_t vPrev16 = vmovl_u8(vPrev);
                uint16x8_t vC00_16 = vmovl_u8(vC00);
                uint16x8_t vC01_16 = vmovl_u8(vC01);
                uint16x8_t vC10_16 = vmovl_u8(vC10);
                uint16x8_t vC11_16 = vmovl_u8(vC11);
                
                float32x4_t vPrevF_L = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vPrev16)));
                float32x4_t vPrevF_H = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vPrev16)));
                
                float32x4_t vC00F_L = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vC00_16)));
                float32x4_t vC00F_H = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vC00_16)));
                float32x4_t vC01F_L = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vC01_16)));
                float32x4_t vC01F_H = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vC01_16)));
                float32x4_t vC10F_L = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vC10_16)));
                float32x4_t vC10F_H = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vC10_16)));
                float32x4_t vC11F_L = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vC11_16)));
                float32x4_t vC11F_H = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vC11_16)));
                
                // Interpolate
                // val = C00*w00 + C01*w01 + C10*w10 + C11*w11
                float32x4_t vInterp_L = vmulq_f32(vC00F_L, vW00);
                vInterp_L = vmlaq_f32(vInterp_L, vC01F_L, vW01);
                vInterp_L = vmlaq_f32(vInterp_L, vC10F_L, vW10);
                vInterp_L = vmlaq_f32(vInterp_L, vC11F_L, vW11);

                float32x4_t vInterp_H = vmulq_f32(vC00F_H, vW00);
                vInterp_H = vmlaq_f32(vInterp_H, vC01F_H, vW01);
                vInterp_H = vmlaq_f32(vInterp_H, vC10F_H, vW10);
                vInterp_H = vmlaq_f32(vInterp_H, vC11F_H, vW11);
                
                // Diff = Interp - Prev
                float32x4_t vDiff_L = vsubq_f32(vInterp_L, vPrevF_L);
                float32x4_t vDiff_H = vsubq_f32(vInterp_H, vPrevF_H);
                
                // Load Gradients
                int16x8_t vIx = vld1q_s16(pIx);
                int16x8_t vIy = vld1q_s16(pIy);
                
                float32x4_t vIxF_L = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vIx)));
                float32x4_t vIxF_H = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vIx)));
                float32x4_t vIyF_L = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vIy)));
                float32x4_t vIyF_H = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vIy)));
                
                // Accumulate Sum(Ix*Diff)
                // Only 7 lanes (index 0..6). Lane 7 should be zeroed or handled.
                // We'll mask logic later or simpler:
                // Just use vsetq_lane to 0 for lane 3 of High part (idx 7).
                vDiff_H = vsetq_lane_f32(0.0f, vDiff_H, 3);
                
                vBox = vmlaq_f32(vBox, vIxF_L, vDiff_L);
                vBox = vmlaq_f32(vBox, vIxF_H, vDiff_H);
                vBoy = vmlaq_f32(vBoy, vIyF_L, vDiff_L);
                vBoy = vmlaq_f32(vBoy, vIyF_H, vDiff_H);
            }
            
            // Horizontal Sum
            bx = vgetq_lane_f32(vBox, 0) + vgetq_lane_f32(vBox, 1) + vgetq_lane_f32(vBox, 2) + vgetq_lane_f32(vBox, 3);
            by = vgetq_lane_f32(vBoy, 0) + vgetq_lane_f32(vBoy, 1) + vgetq_lane_f32(vBoy, 2) + vgetq_lane_f32(vBoy, 3);
#else
            // Scalar Mismatch calculation
            for (int dy = -win_r; dy <= win_r; ++dy) {
                int row_offset = (iy + dy) * w + ix;
                int row_offset_curr = (cy_i + dy) * w + cx_i;
                for (int dx = -win_r; dx <= win_r; ++dx) {
                     float valPrev = (float)prev_img.getData()[row_offset + dx]; // Can use direct ptr
                     float valIx = (float)Ix_img.data[row_offset + dx];
                     float valIy = (float)Iy_img.data[row_offset + dx];
                     
                     // Bilinear sample curr
                     int c_idx = row_offset_curr + dx;
                     const uint8_t* ptr = curr_img.getData() + c_idx;
                     float valCurr = w00 * ptr[0] + w01 * ptr[1] + 
                                     w10 * ptr[w] + w11 * ptr[w+1];
                                     
                     float diff = valCurr - valPrev;
                     bx += valIx * diff;
                     by += valIy * diff;
                }
            }
#endif
            
            // Compute Delta
            float delta_x = (Gyy * bx - Gxy * by) * inv_det;
            float delta_y = (Gxx * by - Gxy * bx) * inv_det;
            
            curr_x += delta_x;
            curr_y += delta_y;
            
            if (std::abs(delta_x) < 0.05f && std::abs(delta_y) < 0.05f) break;
        }
    }
    // Pyramid LK Entry Point
    void computeOpticalFlowPyrLK(const ::Image& prev, const ::Image& curr, 
                                 const std::vector<int>& pixels, 
                                 std::vector<float>& dx_out, 
                                 std::vector<float>& dy_out, 
                                 std::vector<float>& dir_out,
                                 int levels=3) 
    {
        int w = curr.width();
        
        dx_out.assign(pixels.size(), 0.0f);
        dy_out.assign(pixels.size(), 0.0f);
        dir_out.assign(pixels.size(), 0.0f);
        
        // Build Pyramids
        ImagePyramid prev_pyr, curr_pyr;
        buildPyramid(prev, prev_pyr, levels);
        buildPyramid(curr, curr_pyr, levels);
        
        for (size_t i = 0; i < pixels.size(); ++i) {
            int idx = pixels[i];
            int py = idx / w;
            int px = idx % w;
            
            // Initial guess (0 flow)
            float flow_x = 0;
            float flow_y = 0;
            
            // Iterate levels
            for (int l = levels - 1; l >= 0; --l) {
                if (l < levels - 1) {
                    flow_x *= 2.0f;
                    flow_y *= 2.0f;
                }
                
                float scaler = 1.0f / (float)(1 << l);
                float p_lvl_x = px * scaler;
                float p_lvl_y = py * scaler;
                
                float c_lvl_x = p_lvl_x + flow_x;
                float c_lvl_y = p_lvl_y + flow_y;
                
                // Refine
                computeLK_Level(prev_pyr, curr_pyr, l, p_lvl_x, p_lvl_y, c_lvl_x, c_lvl_y);
                
                flow_x = c_lvl_x - p_lvl_x;
                flow_y = c_lvl_y - p_lvl_y;
            }
            
            dx_out[i] = flow_x;
            dy_out[i] = flow_y;
            dir_out[i] = std::atan2(dy_out[i], dx_out[i]);
        }
    }
} // namespace LK_Utils



// Alias or Wrapper for Bed Region Mask
::Image createBedRegionMask(int width, int height, const std::vector<std::pair<int, int>>& points) {
    std::vector<std::pair<float, float>> floatPoints;
    for(auto& p : points) floatPoints.push_back({(float)p.first, (float)p.second});
    return createTrapezoidMask(width, height, floatPoints);
}



class FallDetector::Impl {
public:
    std::unique_ptr<OptimizedBlockMotionEstimator> estimator;
    std::vector<MotionObject> current_objects;
    std::vector<std::vector<MotionObject>> object_history;
    InternalConfig config; // Use unified Config
    std::vector<::VisionSDK::ObjectFeatures> full_frame_objects;
    
    // New fields
    // Bed Region
    std::vector<std::pair<int, int>> bed_region = {{0,0}, {100,0}, {100,100}, {0,100}}; // Default

    // Logic
    int fall_confirmation_counter = 0;
    bool in_fall_state = false;
    long long absolute_frame_count = 0; // Persistent frame counter

    // Face Detector
    FaceDetector faceDetector;
    // bool face_model_inited = false; // Removed duplicate
    
    // Profiling
    struct ProfilingData {
        long long total_time = 0;
        long long motion_est_time = 0;
        long long face_detect_time = 0; // Resize + Detect
        long long fall_logic_time = 0;
        int frame_count = 0;
        
        void Reset() {
             total_time = 0; 
             motion_est_time = 0; 
             face_detect_time = 0; 
             fall_logic_time = 0;
             frame_count = 0;
        }
    } prof;

    // Helper for timing
    long long get_now_us() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (long long)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
    }
    ::Image bedMask;
    bool hasBedMask = false;

    VisionSDKCallback callback;
    // bool is_bed_exit = false; // REMOVED GLOBAL
    std::map<int, bool> object_bed_exit_status; // Per-object status
    
    uint64_t last_timestamp = 0; // Timestamp of previous frame
    std::vector<MotionObject> previous_objects; // Added for tracking
    int frame_idx = 0; // Added frame_idx definition
    
    // Fix: Move static variable from Detect() to here
    int fall_consecutive_frames = 0;
    
    // Bed Exit History: stores {inside_ratio, outside_ratio} per object
    // std::deque<std::pair<float, float>> bed_stats_history; // REMOVED GLOBAL
    std::map<int, std::deque<std::pair<float, float>>> object_bed_stats_history;
    
    // Slow Fall Logic: Accumulated descent distance (pixels) per object
    std::map<int, float> object_accumulated_descent;

    // --- Future-Based Post-Fall Check ---
    struct FallCandidate {
        int id;
        float startX;
        float startY;
        int frames_monitored;
        int max_dist_detected; // Just for debug
    };
    std::vector<FallCandidate> candidates;
    bool was_inside_bed = false; // State tracking

    // --- Case 5: Momentum Transition Observation ---
    struct ObservationState {
        int object_id;
        int trigger_frame;           // Frame when high momentum detected
        int frames_observed;         // Frames observed so far (in observation phase)
        int frames_waiting;          // Frames spent waiting for deceleration
        std::vector<float> momentum_samples;  // Momentum values during observation phase
        bool is_active;              // Whether currently in observation mode
        bool waiting_for_deceleration; // Whether waiting for momentum to drop before observation
        
        // Perspective Filter State
        int aligned_frames;
        int valid_angle_frames;
        
        ObservationState() : object_id(-1), trigger_frame(-1), frames_observed(0), 
                            frames_waiting(0), is_active(false), waiting_for_deceleration(false),
                            aligned_frames(0), valid_angle_frames(0), flat_posture_frames(0), accumulated_bed_ratio(0.0f) {}

        // Posture State
        int flat_posture_frames;
        
        // Bed State
        float accumulated_bed_ratio;


    };
    std::map<int, ObservationState> observation_states;

    // Face Model Init State
    bool face_model_inited = false;
    
    // Tracking
    std::map<int, KalmanFilter> kalmanFilters;
    std::map<int, int> track_ttl; // NEW: Persistence counter
    int global_id_counter = 1000;

    // === NEW: Entry Filter and FG Verification ===
    
    // Entry Filter: Track when each object ID first appeared
    std::map<int, int> object_first_seen_frame;  // ID -> frame when first seen
    std::map<int, bool> object_is_new_entry;     // ID -> true if entered from edge
    
    // Pending Fall Verification with FG trend analysis
    struct PendingFall {
        int object_id;
        int trigger_frame;
        int lookback_start;
        std::vector<int> fg_history;  // FG count history from lookback start
        int frames_monitored;
        bool confirmed;
        bool rejected;
        bool is_high_landing; // New: Flag for high landing candidates requiring strict verification
        // NEW: Motion History for advanced verification
        std::vector<float> dx_history;
        std::vector<float> dy_history;
        std::vector<float> strength_history;
        float sf_verified_max_str = 0.0f; // V11/V12 Recall Check: Store Trigger MaxStr to survive verification history resets
        float sf_verified_up_cons = 0.0f; // V11/V12 Recall Check: Store Trigger UpCons to handle post-fall bounces
        float sf_verified_dir_var = 999.0f; // V12 Tuning: Store Trigger Direction Variance (Consistency)
        float pd_trigger_safe_ratio = 0.0f; // V17.1: Store Bed Safe Ratio at trigger
        float historical_bed_max = 0.0f; // V17.5: Max Safe Ratio in lookback window

    };
    std::vector<PendingFall> pending_falls;
    
    // Global FG history buffer (last 120 frames)
    std::deque<int> fg_count_history_buffer;
    
    // Global Object Y-Pos history (last 120 frames per ID)
    std::map<int, std::deque<float>> object_y_history_buffer;
    
    // V17.5: Safe Ratio History (last 120 frames)
    std::map<int, std::deque<float>> object_safe_ratio_history_buffer;
    
    // NEW: Persistent Object Block Cache for Robust FG Tracking (Object-local stats)
    
    // NEW: Persistent Object Block Cache for Robust FG Tracking (Object-local stats)
    std::map<int, std::vector<int>> persistent_object_blocks;
    
    // Parameters
    static constexpr int LOOKBACK_FRAMES = 40;
    static constexpr int OBSERVATION_WINDOW = 30;
    static constexpr float DECLINE_THRESHOLD = 0.75f;
    static constexpr int ENTRY_SUPPRESS_FRAMES = 30;

    // Optical Flow Frame History
    std::deque<::Image> raw_frame_history;

    // Simple 3x3 Box Blur for noise reduction
    void smoothImage(const ::Image& src, ::Image& dst) {
        int w = src.width();
        int h = src.height();
        if (dst.width() != w || dst.height() != h) {
            dst = src.clone(); // Ensure same size/format
        }
        const uint8_t* p_src = src.getData();
        uint8_t* p_dst = dst.getData();

        for (int y = 1; y < h - 1; ++y) {
            for (int x = 1; x < w - 1; ++x) {
                int sum = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        sum += p_src[(y + dy) * w + (x + dx)];
                    }
                }
                p_dst[y * w + x] = (uint8_t)(sum / 9);
            }
        }
    }

    // Integer-only Lucas-Kanade Optical Flow (PC Version)
    // Integer-only Lucas-Kanade Optical Flow (Replaced by Pyramid LK)
    void computeIntegerLK(const ::Image& prev, const ::Image& curr, 
                          const std::vector<int>& pixels, 
                          std::vector<float>& dx_out, 
                          std::vector<float>& dy_out, 
                          std::vector<float>& dir_out) {
         // Forward to Pyramid implementation (default 3 levels)
         LK_Utils::computeOpticalFlowPyrLK(prev, curr, pixels, dx_out, dy_out, dir_out, 3);
    }

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    // NEON-optimized Integer-only Lucas-Kanade (Replaced by Pyramid LK)
    void computeIntegerLK_NEON(const ::Image& prev, const ::Image& curr, 
                               const std::vector<int>& pixels, 
                               std::vector<float>& dx_out, 
                               std::vector<float>& dy_out, 
                               std::vector<float>& dir_out) {
         // Forward to Pyramid implementation (NEON inside)
         LK_Utils::computeOpticalFlowPyrLK(prev, curr, pixels, dx_out, dy_out, dir_out, 3);
    }
#endif

    void LogTrace(int id, int frame, const char* type, float val, const char* msg) {
        FILE* fp = fopen("detection_trace.txt", "a");
        if (fp) {
            fprintf(fp, "F:%d ID:%d Type:%s Val:%.2f Msg:%s\n", frame, id, type, val, msg);
            fclose(fp);
        }
    }

    Impl() {
        // Initialize estimator with default config
        estimator = std::unique_ptr<OptimizedBlockMotionEstimator>(
            new OptimizedBlockMotionEstimator(config.grid_cols, config.grid_rows, config.block_size, config.search_range)
        );
        estimator->setDiffCheckRange(config.history_size);
        estimator->setSearchMode(config.search_mode);
    }
// ... (skip vectors)
    // Vectors reused per frame
    std::vector<MotionVector> motion_vectors;
    std::vector<BlockPosition> positions;
    std::vector<bool> changed_mask;
    std::vector<::Image> active_blocks;
    std::vector<int> active_indices;
    std::deque<int> global_pixel_history; // NEW for Fall Logic
    
    // Background Update Logic
    ::Image backgroundFrame;
    int bg_update_counter = 0;
    std::vector<int32_t> bg_accumulator; // For accumulating frames during init
    int bg_accumulated_count = 0;
    
    void updateBackground(const ::Image& current, const InternalConfig& cfg, int frame_idx) {
        printf("[Debug] updateBackground called for frame %d\n", frame_idx);
        if (backgroundFrame.width() != current.width() || backgroundFrame.height() != current.height()) {
            backgroundFrame = current.clone();
            // Reset accumulator if size changes
            bg_accumulator.assign(current.width() * current.height() * current.getChannels(), 0);
            bg_accumulated_count = 0;
            printf("[Debug] updateBackground Reset Accumulator\n");
            return;
        }
        
        // 1. Initialization Phase (Average Logic)
        if (cfg.bg_init_start_frame > 0 && cfg.bg_init_end_frame > cfg.bg_init_start_frame) {
            if (frame_idx >= cfg.bg_init_start_frame && frame_idx <= cfg.bg_init_end_frame) {
                int size = current.width() * current.height() * current.getChannels();
                if(bg_accumulator.size() != size) bg_accumulator.resize(size, 0);
                
                const unsigned char* curr_ptr = current.getData();
                for(int i=0; i<size; ++i) {
                    bg_accumulator[i] += curr_ptr[i];
                }
                bg_accumulated_count++;
                
                // If this is the END frame, compute average
                if (frame_idx == cfg.bg_init_end_frame && bg_accumulated_count > 0) {
                     unsigned char* bg_ptr = backgroundFrame.getData();
                     for(int i=0; i<size; ++i) {
                         bg_ptr[i] = (unsigned char)(bg_accumulator[i] / bg_accumulated_count);
                     }
                     // Clear accumulator
                     std::vector<int32_t>().swap(bg_accumulator);
                     bg_accumulated_count = 0;
                     printf("[Debug] updateBackground Init Phase Complete.\n");
                }
                return; // During init, don't run normal update
            }
        }
        
        // 2. Periodic Update Phase
        float alpha = cfg.bg_update_alpha;
        if (alpha <= 0.0f) return;
        
        // Weighted Average
        // bg = (1-alpha)*bg + alpha*curr
        int alpha_int = (int)(alpha * 256.0f);
        if (alpha_int < 0) alpha_int = 0;
        if (alpha_int > 256) alpha_int = 256;
        int inv_alpha = 256 - alpha_int;
        
        int size = current.width() * current.height() * current.getChannels();
        unsigned char* bg_ptr = backgroundFrame.getData();
        const unsigned char* curr_ptr = current.getData();  
        
        for(int i=0; i<size; ++i) {
            int val = (bg_ptr[i] * inv_alpha + curr_ptr[i] * alpha_int) >> 8;
            bg_ptr[i] = (unsigned char)val;
        }
        printf("[Debug] updateBackground Periodic Update Complete.\n");
    }
};

FallDetector::FallDetector() : pImpl(std::make_shared<Impl>()) {}
FallDetector::~FallDetector() = default;


void FallDetector::SetConfig(const InternalConfig& config) {
    bool gridChanged = (config.grid_cols != pImpl->config.grid_cols ||
                        config.grid_rows != pImpl->config.grid_rows ||
                        config.block_size != pImpl->config.block_size ||
                        config.search_range != pImpl->config.search_range);
    
    pImpl->config = config; // Update internal config

    if (gridChanged) {
        pImpl->estimator.reset(new OptimizedBlockMotionEstimator(
            config.grid_cols, config.grid_rows, config.block_size, config.search_range));
    }
    
    pImpl->estimator->setDiffCheckRange(config.history_size);
    pImpl->estimator->setSearchMode(config.search_mode);
    pImpl->estimator->setBlockDecay(config.enable_block_decay, config.block_decay_frames);
    pImpl->estimator->setBlockDilation(config.enable_block_dilation);

    // Re-verify bed region
    if (!pImpl->bed_region.empty()) {
        pImpl->bedMask = createBedRegionMask(800, 600, pImpl->bed_region); // Assume size or handle resize dynamically?
    }
    pImpl->hasBedMask = !pImpl->bed_region.empty();

    // Initialize Face Detector
    // Update Verification Flags - Handled in HermesII_sdk.cpp via pImpl->config

    if (!pImpl->face_model_inited) {
         std::cout << "[FallDetector::SetConfig] Initializing FaceDetector..." << std::endl;
         StatusCode ret = pImpl->faceDetector.Init("res/blaze_face_detect_nnp310_128x128.ty");
         if (ret != StatusCode::OK) {
             std::cout << "[FallDetector::SetConfig] FaceDetector Init Failed: " << (int)ret << std::endl;
             // Do NOT set true, allows retry on next Config call or manually? 
             // Actually, Config is usually called once. If it fails, maybe we should try in Detect too?
             // But let's stick to Config first.
         } else {
             std::cout << "[FallDetector::SetConfig] FaceDetector Init OK" << std::endl;
             pImpl->face_model_inited = true;
         }
    }
}

void FallDetector::SetBedRegion(const std::vector<std::pair<int, int>>& points) {
    pImpl->bed_region = points;
    // Invalidate mask, will be recreated in Detect() when frame size is known
    pImpl->hasBedMask = false; 
    if (!points.empty()) pImpl->hasBedMask = true; // Signal that we have a region
    pImpl->bedMask = ::Image(); // Clear
}


void FallDetector::RegisterCallback(VisionSDKCallback cb) {
    pImpl->callback = cb;
}

void FallDetector::SetHistorySize(int n) {
  pImpl->config.history_size = n;
  if(pImpl->estimator) pImpl->estimator->setDiffCheckRange(n);
}

// Removed SetBedRegion and SetTrapPoints as they are replaced by LoadBedRegion
// void FallDetector::SetBedRegion(const ::Image& bedRegion) {
//     pImpl->bedRegion = bedRegion.clone();
// }

// void FallDetector::SetTrapPoints(const std::vector<std::pair<float, float>>& points) {
//     pImpl->trapPoints = points;
// }



// Helper: Calculate Overlap Percentage (Block Intersection / Prev Size)
float calculateOverlap(const MotionObject& curr, const MotionObject& prev) {
    if (prev.blocks.empty()) return 0.0f;
    int intersection = 0;
    // Assuming blocks are sorted or we use a set for O(N log N) or unsorted O(N*M)
    // Since vectors are small (blocks), O(N*M) is fine.
    for (int cb : curr.blocks) {
        for (int pb : prev.blocks) {
            if (cb == pb) {
                intersection++;
                break;
            }
        }
    }
    return (float)intersection / prev.blocks.size();
}

void TrackObjects(std::vector<MotionObject>& current, const std::vector<MotionObject>& previous, 
                  const InternalConfig& config, 
                  std::map<int, KalmanFilter>& kalmanFilters,
                  std::map<int, int>& track_ttl,
                  int& global_id_counter,
                  std::map<int, int>& object_first_seen_frame,
                  std::map<int, bool>& object_is_new_entry,
                  std::map<int, std::vector<int>>& persistent_object_blocks,
                  int frame_idx) 
{
    printf("[Debug] TrackObjects Start. Current: %zu Previous: %zu\n", current.size(), previous.size());
    float threshold = config.tracking_overlap_threshold;
    int mode = config.tracking_mode; // 1=Original, 2=Hungarian, 3=Kalman, 4=SORT
    int grid_cols = config.grid_cols;
    int grid_rows = config.grid_rows;

    // Helper lambda to detect edge entry
    auto detectEdgeEntry = [&](const MotionObject& obj, int new_id) {
        // Calculate bounding box from blocks
        int min_r = grid_rows, max_r = 0, min_c = grid_cols, max_c = 0;
        for (int blk : obj.blocks) {
            int r = blk / grid_cols;
            int c = blk % grid_cols;
            if (r < min_r) min_r = r;
            if (r > max_r) max_r = r;
            if (c < min_c) min_c = c;
            if (c > max_c) max_c = c;
        }
        
        // Check if touches any edge
        bool touches_edge = (min_r == 0 || max_r >= grid_rows - 1 || 
                            min_c == 0 || max_c >= grid_cols - 1);
        
        object_first_seen_frame[new_id] = frame_idx;
        object_is_new_entry[new_id] = touches_edge;
        
        if (touches_edge) {
            printf("[EntryFilter] New object ID %d touches edge at frame %d\n", new_id, frame_idx);
        }
    };

    if (previous.empty()) {
        for (auto& obj : current) {
             bool was_new = (obj.id < 1000);
             if (was_new) obj.id = global_id_counter++; // New ID
             obj.trajectory.push_back({(int)obj.centerX, (int)obj.centerY});
             
             // Track entry filter info for new objects
             if (was_new) {
                 detectEdgeEntry(obj, obj.id);
             }
             
             // Init Kalman for new objects (Mode 3/4)
             if (mode >= 3) {
                 KalmanFilter kf(obj.centerX, obj.centerY);
                 kalmanFilters.insert({obj.id, kf});
             }
        }
        printf("[Debug] TrackObjects End (No Previous Objects).\n");
        return;
    }

    // --- MODE 1: ORIGINAL (Greedy based on Overlap + Distance with Split/Merge) ---
    if (mode == 1) {
        // Association Map: Curr -> [Prev Indices]
        std::map<int, std::vector<int>> currToPrev;
        std::map<int, std::vector<int>> prevToCurr;

        // 1. Identify Overlaps
        for (size_t i = 0; i < current.size(); ++i) {
            for (size_t j = 0; j < previous.size(); ++j) {
                float overlap = calculateOverlap(current[i], previous[j]);
                bool is_match = false;
                if (overlap > threshold) {
                    is_match = true;
                } else {
                    float dx = current[i].centerX - previous[j].centerX;
                    float dy = current[i].centerY - previous[j].centerY;
                    float dist = std::sqrt(dx*dx + dy*dy);
                    if (dist < 3.0f) is_match = true;
                }
                
                if (is_match) {
                    currToPrev[i].push_back(j);
                    prevToCurr[j].push_back(i);
                }
            }
        }

        // 2. Process Associations (Split/Merge/Match)
        // Case A: Split (One Prev -> Many Curr)
        for (auto const& item : prevToCurr) {
            int prevIdx = item.first;
            auto& currIndices = item.second;
            if (currIndices.size() > 1) {
                const auto& pObj = previous[prevIdx];
                float sumX = 0, sumY = 0;
                for (int objIdx : currIndices) {
                     sumX += current[objIdx].centerX;
                     sumY += current[objIdx].centerY;
                }
                float avgX = sumX / currIndices.size();
                float avgY = sumY / currIndices.size();
                
                for (int objIdx : currIndices) {
                    current[objIdx].id = pObj.id;
                    current[objIdx].trajectory = pObj.trajectory;
                    current[objIdx].trajectory.push_back({(int)avgX, (int)avgY});
                    
                    // Fallback Strength Calculation (Centroid Speed)
                    float cx = current[objIdx].centerX;
                    float cy = current[objIdx].centerY;
                    float px = pObj.centerX;
                    float py = pObj.centerY;
                    float dist_blocks = std::sqrt(std::pow(cx-px, 2) + std::pow(cy-py, 2));
                    
                    // If ME strength is suspiciously low but object moved, use centroid speed
                    // Threshold: 0.05 blocks (e.g. ~1-2 pixels)
                    // DEBUG: Print diff
                    // printf("[TrackDebug] ID:%d Str:%.2f DistBlk:%.3f\n", current[objIdx].id, current[objIdx].strength, dist_blocks);
                    
                    if (current[objIdx].strength < 0.5f && dist_blocks > 0.05f) {
                        float pixel_dist = dist_blocks * config.block_size;
                        current[objIdx].strength = pixel_dist;
                        // Also update avgDx/avgDy to reflect this?
                        current[objIdx].avgDx = (cx - px) * config.block_size;
                        current[objIdx].avgDy = (cy - py) * config.block_size;
                        printf("[TrackFallback] ID %d Used Centroid Speed (Blocks: %.3f -> Pixels: %.1f)\n", current[objIdx].id, dist_blocks, pixel_dist);
                    }
                    
                    current[objIdx].acceleration = current[objIdx].strength - pObj.strength;
                }
            }
        }

        // Case B: Merge or 1-to-1
        for (auto const& item : currToPrev) {
            int currIdx = item.first;
            auto& prevIndices = item.second;
             if (prevIndices.size() > 1) {
                 // Merge
                 int minID = 999999;
                 int bestPrevIdx = -1;
                 for (int pIdx : prevIndices) {
                     if (previous[pIdx].id < minID) {
                         minID = previous[pIdx].id;
                         bestPrevIdx = pIdx;
                     }
                 }
                 // Logic: Inherit Smallest ID
                 current[currIdx].id = minID;
                 current[currIdx].trajectory = previous[bestPrevIdx].trajectory;
                 float avgPrevStrength = 0;
                 for (int pIdx : prevIndices) avgPrevStrength += previous[pIdx].strength;
                 avgPrevStrength /= prevIndices.size();
                 
                 // Fallback Strength
                 float dist = 0.0f; // Hard to define dist for merge. Use ME strength.
                 
                 current[currIdx].acceleration = current[currIdx].strength - avgPrevStrength;
                 current[currIdx].trajectory.push_back({(int)current[currIdx].centerX, (int)current[currIdx].centerY});
             } 
             else if (prevIndices.size() == 1) {
                 int pIdx = prevIndices[0];
                 if (prevToCurr[pIdx].size() == 1) {
                     // 1-to-1
                     current[currIdx].id = previous[pIdx].id;
                     current[currIdx].trajectory = previous[pIdx].trajectory;
                     
                     // Fallback Strength (Centroid)
                     float cx = current[currIdx].centerX;
                     float cy = current[currIdx].centerY;
                     float px = previous[pIdx].centerX;
                     float py = previous[pIdx].centerY;
                     float dist_blocks = std::sqrt(std::pow(cx-px, 2) + std::pow(cy-py, 2));
                     
                     if (current[currIdx].strength < 0.5f && dist_blocks > 0.05f) {
                         float pixel_dist = dist_blocks * config.block_size;
                         current[currIdx].strength = pixel_dist;
                         current[currIdx].avgDx = (cx - px) * config.block_size;
                         current[currIdx].avgDy = (cy - py) * config.block_size;
                         printf("[TrackFallback] ID %d Used Centroid Speed (Blocks: %.3f -> Pixels: %.1f)\n", current[currIdx].id, dist_blocks, pixel_dist);
                     }

                     current[currIdx].trajectory.push_back({(int)cx, (int)cy});
                     current[currIdx].acceleration = current[currIdx].strength - previous[pIdx].strength;
                 }
             }
        }
    }
    // --- MODE 2, 3, 4: Standard Matching (1-to-1) ---
    // --- MODE 2, 3, 4: Standard Matching (1-to-1) ---
    else {
        // Prepare Cost Matrix
        // Rows: Active Tracks (from Kalman Filters) for Mode 3/4, or Previous for Mode 2
        // Cols: Current Objects
        
        std::vector<int> trackIDs;
        std::vector<std::pair<float, float>> trackPositions; // Predicted or Last Known

        if (mode >= 3) {
            // Use PERSISTENT TRACKS (Coasting)
            for (auto& kv : kalmanFilters) {
                trackIDs.push_back(kv.first);
                KalmanFilter& kf = kv.second;
                float kx, ky;
                kf.GetState(kx, ky); 
                trackPositions.push_back({kx, ky});
                //printf("[TrackDebug] TrackID %d (TTL %d) Pos (%.1f, %.1f)\n", kv.first, track_ttl[kv.first], kx, ky);
            }
        } else {
            // Mode 2: Use Previous Frame Objects
            for (const auto& p : previous) {
                trackIDs.push_back(p.id);
                trackPositions.push_back({p.centerX, p.centerY});
            }
        }

        int rows = trackIDs.size();
        int cols = current.size();
        std::vector<std::vector<float>> costMatrix(rows, std::vector<float>(cols));

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float cost = 0;
                float tx = trackPositions[i].first;
                float ty = trackPositions[i].second;
                float dx = current[j].centerX - tx;
                float dy = current[j].centerY - ty;
                cost = std::sqrt(dx*dx + dy*dy);
                
                //printf("[TrackDebug] Cost T%d -> Obj%d (%.1f, %.1f) vs (%.1f, %.1f) = Dist %.2f\n", 
                //        trackIDs[i], j, tx, ty, current[j].centerX, current[j].centerY, cost);
                
                // Gating
                if (cost > 100.0f) cost = 99999.0f;
                costMatrix[i][j] = cost;
            }
        }

        std::vector<int> assignment;
        if (mode == 3) {
             // Greedy
             assignment.assign(rows, -1);
             std::vector<bool> colUsed(cols, false);
             for(int i=0; i<rows; ++i) {
                 float minC = 9999.0f;
                 int bestJ = -1;
                 for(int j=0; j<cols; ++j) {
                     if(!colUsed[j] && costMatrix[i][j] < minC) {
                         minC = costMatrix[i][j];
                         bestJ = j;
                     }
                 }
                 if(bestJ != -1 && minC < 50.0f) {
                     assignment[i] = bestJ;
                     colUsed[bestJ] = true;
                 }
             }
        } else {
            // Hungarian (Mode 2 & 4)
            assignment = HungarianAlgorithm::Solve(costMatrix);
        }

        // Apply Assignment
        std::vector<bool> currentMatched(cols, false);
        
        // 1. Process Tracks (Rows)
        for (int i = 0; i < rows; ++i) {
            int trackID = trackIDs[i];
            int j = assignment[i];
            
            bool matched = false;
            
            if (j >= 0 && j < cols) {
                //printf("[TrackDebug] Assignment T%d -> Obj%d. Cost %.2f\n", trackID, j, costMatrix[i][j]);
                if (costMatrix[i][j] <= 50.0f) {
                    matched = true;
                    currentMatched[j] = true;
                    
                    // Inherit ID
                    //printf("[TrackDebug] MERGE: Obj %d inherits T%d\n", j, trackID);
                    current[j].id = trackID;
                    
                    // Inherit Trajectory? 
                    // Previous trajectory is in `previous` vector, but `trackID` comes from Map.
                    // We might not have trajectory history if it was coasting (not in previous).
                    // This is a limitation: Trajectory vector is in MotionObject, not in KalmanFilter.
                    // It's acceptable for now: resumed object has ID, trajectory starts fresh or from last point?
                    // User asked for "Inherit ID". Trajectory is nice to have but secondary.
                    
                    // Update Kalman
                         kalmanFilters.at(trackID).Update(current[j].centerX, current[j].centerY);
                         
                         // Use Kalman Velocity for Strength if ME failed
                         float kvx = kalmanFilters.at(trackID).x[2];
                         float kvy = kalmanFilters.at(trackID).x[3];
                         float k_strength = std::sqrt(kvx*kvx + kvy*kvy);
                         
                         // Threshold 0.5 pixels
                         if (current[j].strength < 0.5f && k_strength > 0.5f) {
                              current[j].strength = k_strength;
                              current[j].avgDx = kvx;
                              current[j].avgDy = kvy;
                          }
                          
                          // Calculate Acceleration (if we can find previous strength)
                          // Note: previous vector might not align with trackIDs if coasting.
                          // Limitation: If coasting, we lost previous strength history unless stored in Kalman/Map.
                          // Iterate previous to find strength?
                          float prevStrength = 0.0f;
                          bool foundPrev = false;
                          for(const auto& p : previous) {
                              if(p.id == trackID) {
                                  prevStrength = p.strength;
                                  foundPrev = true; 
                                  break;
                              }
                          }
                          // If not in previous (was coasting), assume constant velocity (acc=0) or 0 strength?
                          // For now, if found, calc accel.
                          if (foundPrev) {
                               current[j].acceleration = current[j].strength - prevStrength;
                          } else {
                               current[j].acceleration = 0.0f; // Reset if resumed from coasting
                          }

                    // RESET TTL
                    track_ttl[trackID] = 60; // Coast for 60 frames (~2 sec)
                }
            }

            if (!matched && mode >= 3) {
                 // TRACK LOST (Coasting)
                 // Decrement TTL
                 if (track_ttl.find(trackID) == track_ttl.end()) track_ttl[trackID] = 60;
                 
                 track_ttl[trackID]--;
                 if (track_ttl[trackID] <= 0) {
                     kalmanFilters.erase(trackID);
                     track_ttl.erase(trackID);
                     // std::cout << "Track " << trackID << " Expired." << std::endl;
                 }
            }
        }
        
        // --- DUPLICATE REMOVAL LOGIC (Fix ID Flickering) ---
        // Check if any "Zombie" tracks are colliding with "Assigned" objects
        if (mode >= 3) {
            std::vector<int> tracksToRemove;
            for(int j=0; j<cols; ++j) {
                if(currentMatched[j]) {
                    int assignedID = current[j].id;
                    float cx = current[j].centerX;
                    float cy = current[j].centerY;
                    
                    // Check against ALL existing tracks (potential zombies)
                    for(auto& kv : kalmanFilters) {
                        int otherID = kv.first;
                        if (otherID == assignedID) continue; // Skip self
                        
                        // Check distance
                        float kx, ky; 
                        kv.second.GetState(kx, ky);
                        float dx = cx - kx;
                        float dy = cy - ky;
                        float dist = std::sqrt(dx*dx + dy*dy);
                        
                        if (dist < 50.0f) {
                             // COLLISION DETECTED
                             // Scenario: Assigned 1011, but 1010 is also right here.
                             // Prefer Older ID (Smaller)
                             if (otherID < assignedID) {
                                 // Swap! Inherit the older ID
                                 printf("[TrackFix] Swap ID %d -> %d (Prefer Older). Kill %d.\n", assignedID, otherID, assignedID);
                                 tracksToRemove.push_back(assignedID); // Kill the newer one we just assigned
                                 current[j].id = otherID; // Update object to older ID
                                 
                                 // Update Assignment Ref for future loops (though we break/continue)
                                 assignedID = otherID; 
                                 
                                 // Also need to Update the Kalman Filter for the SWAPPED ID (otherID)
                                 kalmanFilters.at(otherID).Update(cx, cy);
                                 track_ttl[otherID] = 60; // Reset TTL for revived old track
                             } else {
                                 // Zombie is newer (or just duplicate). Kill it.
                                 printf("[TrackFix] Kill Duplicate Track %d (Assigned %d is Older/Better)\n", otherID, assignedID);
                                 tracksToRemove.push_back(otherID);
                             }
                        }
                    }
                }
            }
            // Execute Removal
            for(int id : tracksToRemove) {
                kalmanFilters.erase(id);
                track_ttl.erase(id);
                persistent_object_blocks.erase(id);
                
                // Also remove from the 'current' object list so we don't return duplicates
                auto it = std::remove_if(current.begin(), current.end(), 
                                         [id](const MotionObject& obj){ return obj.id == id; });
                if (it != current.end()) {
                    current.erase(it, current.end());
                    // Since 'current' size changed, we should theoretically adjust loops, 
                    // but we are at the end of the loop over 'current' anyway (controlled by 'cols').
                    // actually 'cols' is just current.size(), so we are fine.
                }
            }
        }
        
        // Handle Unmatched Current Objects (New IDs)
        for(int j=0; j<cols; ++j) {
            if(!currentMatched[j]) {
                // New Object
                bool was_new = (current[j].id < 1000);
                if (was_new) current[j].id = global_id_counter++;
                 current[j].trajectory.push_back({(int)current[j].centerX, (int)current[j].centerY});
                 
                 // Track entry filter info for new objects
                 if (was_new) {
                     detectEdgeEntry(current[j], current[j].id);
                 }
                 
                 // Init Kalman
                 if (mode >= 3) {
                     KalmanFilter kf(current[j].centerX, current[j].centerY);
                     kalmanFilters.insert({current[j].id, kf});
                     track_ttl[current[j].id] = 60; // Init TTL
                 }
            }
        }
    }

    // FINAL SAFETY: Guarantee Unique IDs
    // The previous logic (swaps, etc.) might have left duplicates.
    // We sort by ID, then unique them to ensure downstream can trust ID Uniqueness.
    if (!current.empty()) {
        std::sort(current.begin(), current.end(), [](const MotionObject& a, const MotionObject& b){
            return a.id < b.id;
        });
        auto last = std::unique(current.begin(), current.end(), [](const MotionObject& a, const MotionObject& b){
            return a.id == b.id; 
        });
        current.erase(last, current.end());
    }

    // Mode 1 (Legacy) - No change
    if (mode == 1) {
        for (auto& obj : current) {
            if (obj.trajectory.empty()) {
                bool was_new = (obj.id < 1000);
                if (was_new) obj.id = global_id_counter++;
                obj.trajectory.push_back({(int)obj.centerX, (int)obj.centerY});
                if (was_new) {
                    detectEdgeEntry(obj, obj.id);
                }
            }
        }
    }
    
    // NEW: Update Persistent Block Cache for all current (matched) objects
    for (const auto& obj : current) {
        if (obj.id >= 1000) {
            persistent_object_blocks[obj.id] = obj.blocks;
        }
    }

    printf("[Debug] TrackObjects End.\n");
}


// ==================================================================================
//  NEW Fall Detection Logic (Momentum Trend + Pixel Trend)
// ==================================================================================

// Helper: Calculate Trend (Check for "High to Low" pattern)
// Returns true if there is a significant decreasing trend.
// Also validates that the "High" part is high enough.
bool calculateTrend(const std::vector<float>& data, float threshold, bool& outIsHighEnough) {
    if (data.size() < 4) return false;

    // Split into two halves
    size_t half = data.size() / 2;
    float sum1 = 0, sum2 = 0;
    float max1 = 0;

    for (size_t i = 0; i < half; ++i) {
        sum1 += data[i];
        if(data[i] > max1) max1 = data[i];
    }
    for (size_t i = half; i < data.size(); ++i) {
        sum2 += data[i];
    }

    float avg1 = sum1 / half;
    float avg2 = sum2 / (data.size() - half);

    // "High to Low" means first half is significantly higher than second half
    // User Requirement: "High part average must be higher than threshold"
    outIsHighEnough = (avg1 > threshold);
    
    // Trend check: Avg1 should be significantly larger than Avg2
    // e.g. Avg1 > Avg2 * 1.5 or Diff > X?
    return (avg1 > avg2 * 1.2f); // 20% drop at least
}

// Helper: Calculate Trend with Time Window (Check if peak was recent)
// Returns true if a peak occurred within the last 'window' frames and is now lower.
// This allows for lag between Momentum Trend and Pixel Trend.
bool calculateTrendRelaxed(const std::vector<float>& data, int window, float threshold, bool& outIsHighEnough) {
    if (data.size() < (size_t)window) return false;
    
    // Check if there was a peak in the last 'window' frames
    float max_val = -99999.0f;
    int max_idx = -1;
    
    // Look at the "recent past" part of data
    // Data is chronologically ordered? 
    // In detectFallMomentumTrend, we reverse them to be chronological.
    // So data.back() is current.
    
    // Find max in last 'window' frames
    int start_check = std::max(0, (int)data.size() - window);
    for(int i=start_check; i<(int)data.size(); ++i) {
        if(data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    
    outIsHighEnough = (max_val > threshold);
    if (!outIsHighEnough) return false;
    
    // Check if current is significantly lower than peak
    float current = data.back();
    // Adjusted sensitivity for FN 1696-1730
    return (current < max_val * 0.85f); // Relaxed: Current is < 85% of Peak (15% drop)
}

// Helper: Calculate Consistency
// Returns a score (lower is better/more consistent). 
// Using circular variance approximation or dot products.
float calculateConsistency(const std::vector<float>& dxs, const std::vector<float>& dys) {
    if (dxs.empty()) return 1.0f; // High variance if empty

    float sum_dx = 0, sum_dy = 0;
    float sum_mag = 0;

    for (size_t i = 0; i < dxs.size(); ++i) {
        sum_dx += dxs[i];
        sum_dy += dys[i];
        sum_mag += std::sqrt(dxs[i]*dxs[i] + dys[i]*dys[i]);
    }

    float res_mag = std::sqrt(sum_dx*sum_dx + sum_dy*sum_dy);
    
    // Consistency = 1.0 - (Resultant Magnitude / Sum of Magnitudes)
    // If all aligned, Res == Sum, Consistency = 0.
    if (sum_mag < 0.001f) return 1.0f; // No motion, inconsistent?
    
    return 1.0f - (res_mag / sum_mag);
}

// Global Safety: Check if large portion of screen is moving
bool isGlobalMotionExcessive(int changed_block_count, int total_blocks) {
    return (changed_block_count > total_blocks / 4);
}

// Helper: Check if point is exiting screen (Margin check)
bool isExitingScreen(float x, float y, int w, int h) {
    // 5% margin?
    int marginX = w * 0.05;
    int marginY = h * 0.05;
    bool x_edge = (x < marginX) || (x > w - marginX);
    bool y_edge = (y < marginY) || (y > h - marginY);
    return x_edge || y_edge;
}

void detectFallMomentumTrend(
    const std::vector<std::vector<MotionObject>>& history, // Changed to vector
    const std::deque<int>& pixel_history,
    const std::vector<MotionObject>& current_objects,
    int n_history, // For Consistency (e.g. 10)
    int m_trend,   // For Trend (e.g. 20)
    float strength_thresh,
    int w, int h,
    int total_blocks,
    int changed_blocks,
    const uint8_t* bed_mask,
    std::vector<int>& outTriggeredIds,
    std::string& outWarning,
    const std::map<int, int>& object_first_seen_frame,
    const std::map<int, bool>& object_is_new_entry,
    int frame_idx,
    int entry_suppress_frames = 30
) {
    int hist_size = history.size();
    if (hist_size < std::max(n_history, m_trend)) { 
        // printf("[TrendTrace] Early Exit: HistSize %d  Req %d\n", hist_size, std::max(n_history, m_trend));
        return; 
    }
    if (pixel_history.size() < (size_t)m_trend) {
        return;
    }

    for (const auto& curr_obj : current_objects) {
         // 0. Safety: Global Motion (Disabled - Risk to large objects/noise)
        /*
        if (isGlobalMotionExcessive(changed_blocks, total_blocks)) {
             continue;
        }
        */

        // 1. Gather History for this object
        std::vector<float> dxs, dys, mags;
        std::vector<float> recent_dxs, recent_dys; 
        
        bool found_history = true;
        int collected = 0;
        
        for (int i = hist_size - 1; i >= 0 && collected < m_trend; --i) {
            bool found = false;
            for (const auto& h_obj : history[i]) {
                if (h_obj.id == curr_obj.id) {
                    dxs.push_back(h_obj.avgDx);
                    dys.push_back(h_obj.avgDy);
                    mags.push_back(h_obj.strength);
                    
                    if (collected < n_history) {
                        recent_dxs.push_back(h_obj.avgDx);
                        recent_dys.push_back(h_obj.avgDy);
                    }
                    found = true;
                    break;
                }
            }
            if (!found) { found_history = false; break; }
            collected++;
        }
        
        if (!found_history || collected < m_trend) {
             if (curr_obj.blocks.size() > 10) printf("[TrendTrace] Incomplete History for ID %d. Collected: %d\n", curr_obj.id, collected);
             continue;
        }
        
        // Reverse to chronological order
        std::reverse(dxs.begin(), dxs.end());
        std::reverse(dys.begin(), dys.end());
        std::reverse(mags.begin(), mags.end());
        
        // Check Momentum Direction (Most recent / avg)
        float recent_avg_dx = 0, recent_avg_dy = 0;
        for(float v : recent_dxs) recent_avg_dx += v;
        for(float v : recent_dys) recent_avg_dy += v;
        if(!recent_dxs.empty()) {
            recent_avg_dx /= recent_dxs.size();
            recent_avg_dy /= recent_dys.size();
        }
        
        bool y_dominant = std::abs(recent_avg_dy) > std::abs(recent_avg_dx);
        bool potential_fall = false;
        std::string reason = "";
 
        // Prepare pixel history float vector once for reuse
        std::vector<float> pixel_history_float;
        for(int p : pixel_history) pixel_history_float.push_back((float)p); 

        // DEBUG PRINT
        /*
        if (curr_obj.blocks.size() > 10) {
             float avg_mag = 0; for(float m : mags) avg_mag += m; avg_mag /= mags.size();
             printf("[Trend] ID %d Size %zu Mag %.2f Dy %.2f Dx %.2f YDom %d\n", 
                    curr_obj.id, curr_obj.blocks.size(), avg_mag, recent_avg_dy, recent_avg_dx, y_dominant);
        }
        */ 

        bool mom_high_enough = false;
        
        // ADAPTIVE THRESHOLD: Top-Down perspective creates weaker optical flow.
        float effective_strength_thresh = strength_thresh;
        if (curr_obj.blocks.size() > 10) {
            effective_strength_thresh = 1.0f;
        }
        
        bool mom_trend_down = calculateTrendRelaxed(mags, m_trend, effective_strength_thresh, mom_high_enough);
        
        // 2. FN Fix for Low Velocity Fall / Slide / Slow Fall
        if (!mom_high_enough) {
             int dom_count = 0;
             float total_disp = 0;
             float current_dom_avg = y_dominant ? recent_avg_dy : recent_avg_dx;
             float dom_sign = (current_dom_avg > 0) ? 1.0f : -1.0f;
             const std::vector<float>& dom_vec = y_dominant ? recent_dys : recent_dxs;
             
             for(float d : dom_vec) {
                 if (d * dom_sign > 0.5f) { 
                     dom_count++;
                     total_disp += d;
                 }
             }
             
             if (dom_count >= (int)(dom_vec.size() * 0.8) && std::abs(total_disp) > 5.0f) {
                  bool pix_high_enough = false; 
                  bool pix_trend_down = calculateTrendRelaxed(pixel_history_float, m_trend * 2, 0, pix_high_enough); 
                  if (pix_trend_down) {
                      potential_fall = true;
                      reason = y_dominant ? "Slow-Vert+Pix" : "Slow-Horiz+Pix";
                  }
             }
        }
         
        if (mom_trend_down && mom_high_enough) {
            std::vector<float> pix_counts;
            size_t pix_start = (pixel_history.size() > (size_t)m_trend) ? pixel_history.size() - m_trend : 0;
            for(size_t i=pix_start; i < pixel_history.size(); ++i) {
                 pix_counts.push_back((float)pixel_history[i]);
            }
            bool pix_high_enough = false; 
            bool pix_trend_down = calculateTrendRelaxed(pix_counts, m_trend, 0, pix_high_enough); 
            
            if (pix_trend_down) {
                potential_fall = true;
                reason = "Vert+Trend(Relaxed)";
            }
        }
        
        if (potential_fall) {
            // Entry Filter: Suppress new edge-entry objects for ENTRY_SUPPRESS_FRAMES
            auto first_seen_it = object_first_seen_frame.find(curr_obj.id);
            auto is_entry_it = object_is_new_entry.find(curr_obj.id);
            
            bool suppress = false;
            if (first_seen_it != object_first_seen_frame.end() && 
                is_entry_it != object_is_new_entry.end()) {
                int age = frame_idx - first_seen_it->second;
                bool is_entry = is_entry_it->second;
                if (is_entry && age < entry_suppress_frames) {
                    suppress = true;
                    printf("[EntryFilter] Suppressed trigger for ID %d (age=%d, entry=%d)\n", 
                           curr_obj.id, age, is_entry);
                }
            }
            
            if (!suppress) {
                outTriggeredIds.push_back(curr_obj.id);
                outWarning += reason + "; ";
            }
        }
    }
}

StatusCode FallDetector::Detect(const Image& frame, bool& is_fall) {
    is_fall = false;
    pImpl->absolute_frame_count++; // Increment frame counter
    printf("[Debug] FallDetector::Detect Start Frame %lld\n", pImpl->absolute_frame_count);
  
    // 0. Timestamp Validation
    if (pImpl->config.expected_frame_interval_ms > 0 && pImpl->last_timestamp > 0) {
        uint64_t diff = frame.timestamp - pImpl->last_timestamp;
        int error = std::abs((int)diff - pImpl->config.expected_frame_interval_ms);
        
        if (error > pImpl->config.frame_interval_tolerance_ms) {
            printf("[FallDetector] Timestamp Discontinuity Error! Diff: %lu ms, Expected: %d ms\n", 
                   diff, pImpl->config.expected_frame_interval_ms);
            pImpl->last_timestamp = frame.timestamp; 
            return StatusCode::ERROR_TIMESTAMP_DISCONTINUITY;
        }
    }
    pImpl->last_timestamp = frame.timestamp;
    
    // V18 Fix: Declare variables at function scope (V17 Logic that declared them is disabled)
    std::string warningMsg = "";
    std::vector<int> triggered_objects;


    int W = frame.width;
    int H = frame.height;
    int bSize = (pImpl->config.block_size > 0) ? pImpl->config.block_size : 16;
    
    // Ensure safe grid dimensions
    if (pImpl->config.grid_cols <= 0) pImpl->config.grid_cols = (W + bSize - 1) / bSize;
    if (pImpl->config.grid_rows <= 0) pImpl->config.grid_rows = (H + bSize - 1) / bSize;
    
    int grid_cols = pImpl->config.grid_cols;
    int grid_rows = pImpl->config.grid_rows;
    
    // Resize mask check
    if (pImpl->hasBedMask && (pImpl->bedMask.width() != W || pImpl->bedMask.height() != H)) {
         pImpl->bedMask = createBedRegionMask(W, H, pImpl->bed_region);
    }
  
    // 1. Motion Estimation
    #if ENABLE_PERF_PROFILING
    long long t0 = pImpl->get_now_us();
    #endif

    // Needs Gray Image (1 channel) for correct stride/SAD in OptimizedBlockMotionEstimator
    ::Image wrapper(W, H, 1);
    
    if (frame.channels == 3) {
        // Simple RGB -> Gray Conversion
        const uint8_t* src = frame.data;
        uint8_t* dst = wrapper.getData();
        int size = W * H;
        for (int i = 0; i < size; ++i) {
            // Y = 0.299R + 0.587G + 0.114B
            // Fast approximation: (R + 2G + B) / 4 or just G
            int r = src[i*3];
            int g = src[i*3+1];
            int b = src[i*3+2];
            dst[i] = (uint8_t)((r*77 + g*150 + b*29) >> 8); // integer approx for standard weights
        }
    } else if (frame.channels == 1) {
        memcpy(wrapper.getData(), frame.data, W * H);
    }

    // V3: Background Update Logic (Using Gray Image)
    // Background Update
    if (pImpl->config.bg_update_interval_frames > 0) {
        pImpl->bg_update_counter++;
        bool isInitPhase = (pImpl->frame_idx >= pImpl->config.bg_init_start_frame && pImpl->frame_idx <= pImpl->config.bg_init_end_frame);
        
        if (isInitPhase || (pImpl->bg_update_counter >= pImpl->config.bg_update_interval_frames)) {
            // Only periodic update if NOT in init phase (init phase handled inside updateBackground)
            // Wait, updateBackground logic handles check.
            // 4. Background Update
    #if ENABLE_PERF_PROFILING
    long long t3 = pImpl->get_now_us();
    #endif
    printf("[Debug] Calling updateBackground...\n");
    pImpl->updateBackground(wrapper, pImpl->config, pImpl->frame_idx); // Using frame_idx or absolute?
                                                                      // Internal frame_idx is usually 0 if not set?
                                                                      // Wait, pImpl->frame_idx is 0. 
                                                                      // Maybe we should pass absolute_frame_count?
    printf("[Debug] updateBackground Done.\n");
            if (!isInitPhase) pImpl->bg_update_counter = 0;
        }
    }

    // --- raw_frame_history Management ---
    pImpl->raw_frame_history.push_back(wrapper.clone());
    int max_history = pImpl->config.opt_flow_frame_distance + 1;
    if ((int)pImpl->raw_frame_history.size() > max_history) {
        pImpl->raw_frame_history.pop_front();
    }
    
    // Generate and Save BG Mask
    int global_fg_count = 0;
    if (pImpl->backgroundFrame.width() > 0) 
    {
        // Create Binary Mask
        int w = wrapper.width();
        int h = wrapper.height();
        
        std::vector<unsigned char> maskData(w * h);
        const uint8_t* curr = wrapper.getData();
        const uint8_t* bg = pImpl->backgroundFrame.getData();
        int diff_thr = pImpl->config.bg_diff_threshold;
        
        const unsigned char* bed_data = (pImpl->hasBedMask) ? pImpl->bedMask.getData() : nullptr;
        int bed_w = (pImpl->hasBedMask) ? pImpl->bedMask.width() : 0;
        int bed_h = (pImpl->hasBedMask) ? pImpl->bedMask.height() : 0;

        for(int i = 0; i < w * h; ++i) {
            int diff = std::abs((int)curr[i] - (int)bg[i]);
            
            // Apply Bed Mask filter immediately
            if (bed_data) {
                if (bed_data[i] == 0) { // 0 is Inside Bed
                     diff = 0; // Force to background
                }
            }
            
            if (diff > diff_thr) {
                maskData[i] = 255;
                global_fg_count++;
            } else {
                maskData[i] = 0;
            }
        }

        // Identify and store full-frame foreground objects
        pImpl->full_frame_objects = find_objects_optimized(maskData.data(), w, h, 20); // Min area 20 pixels

        // --- NEW: Optical Flow (Integer-LK) for top 2 objects ---
        if ((int)pImpl->raw_frame_history.size() >= max_history) {
            // Sort objects by area descending
            std::vector<::VisionSDK::ObjectFeatures*> sorted_objs;
            for (auto& obj : pImpl->full_frame_objects) {
                sorted_objs.push_back(&obj);
            }
            std::sort(sorted_objs.begin(), sorted_objs.end(), [](::VisionSDK::ObjectFeatures* a, ::VisionSDK::ObjectFeatures* b) {
                return a->area > b->area;
            });

            // NEW: Robust LK uses smoothed frames
            ::Image smoothed_prev, smoothed_curr;
            pImpl->smoothImage(pImpl->raw_frame_history.front(), smoothed_prev);
            pImpl->smoothImage(pImpl->raw_frame_history.back(), smoothed_curr);

            const ::Image& prev_frame = smoothed_prev; 
            const ::Image& curr_frame = smoothed_curr;

            int num_to_process = std::min(2, (int)sorted_objs.size());
            for (int i = 0; i < num_to_process; ++i) {
                auto* obj = sorted_objs[i];
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                pImpl->computeIntegerLK_NEON(prev_frame, curr_frame, obj->pixels, obj->pixel_dx, obj->pixel_dy, obj->pixel_dir);
#else
                pImpl->computeIntegerLK(prev_frame, curr_frame, obj->pixels, obj->pixel_dx, obj->pixel_dy, obj->pixel_dir);
#endif
            }
        }
        
        // Optional: Save BG Mask if configured
        if (pImpl->config.enable_save_bg_mask) {
             // (Logic for saving here if needed)
        }
    }
            
            // Save
            // char filename[256];
            // // Ensure directory check (handled by caller possibly, but we should be safe)
            // std::string savePath = pImpl->config.save_image_path.empty() ? "." : pImpl->config.save_image_path;
            // snprintf(filename, sizeof(filename), "%s/bg_mask_%05d.bmp", savePath.c_str(), pImpl->frame_idx);
            
            // // SaveBMP (Minimal logic, header+data)
            // FILE* f = fopen(filename, "wb");
            // if(f) {
            //     // Grayscale BMP Header? Or standard 24bit?
            //     // 8-bit BMP requires Palette. 24-bit is easier.
            //     // Let's save as 24-bit BGR for compatibility.
            //     int filesize = 54 + 3 * w * h;
            //     unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0,0,0, 54,0,0,0};
            //     unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
                
            //     bmpfileheader[ 2] = (unsigned char)(filesize);
            //     bmpfileheader[ 3] = (unsigned char)(filesize>>8);
            //     bmpfileheader[ 4] = (unsigned char)(filesize>>16);
            //     bmpfileheader[ 5] = (unsigned char)(filesize>>24);

            //     bmpinfoheader[ 4] = (unsigned char)(w);
            //     bmpinfoheader[ 5] = (unsigned char)(w>>8);
            //     bmpinfoheader[ 6] = (unsigned char)(w>>16);
            //     bmpinfoheader[ 7] = (unsigned char)(w>>24);
            //     bmpinfoheader[ 8] = (unsigned char)(h); 
            //     bmpinfoheader[ 9] = (unsigned char)(h>>8); // Negative for Top-Down? No, standard is Bottom-Up.
            //     // If we want raw save, let's use +h (Top-Down inverse usually)
            //     // Standard BMP: Height > 0 means Bottom-Up.
            //     // Our data is Top-Down. To save correctly, we should flip or use negative height (some viewers support).
            //     // Let's use negative height for Top-Down order if header allows (V5/V4 header).
            //     // But simplified here: Just write Top-Down data with +Height -> Image will be flipped.
            //     // We'll flip row writing.
            //     bmpinfoheader[11] = (unsigned char)(h>>24);

            //     fwrite(bmpfileheader,1,14,f);
            //     fwrite(bmpinfoheader,1,40,f);
                
            //     int pad = (4 - (w * 3) % 4) % 4;
            //     unsigned char bmppad[3] = {0,0,0};
                
            //     // Write Bottom-Up (h-1 to 0)
            //     for(int y=h-1; y>=0; y--) {
            //         for(int x=0; x<w; x++) {
            //             unsigned char v = maskData[y*w+x];
            //             unsigned char pixel[3] = {v,v,v};
            //             fwrite(pixel, 1, 3, f);
            //         }
            //         fwrite(bmppad, 1, pad, f);
            //     }
            //     fclose(f);
            // }

    #if ENABLE_PERF_PROFILING
    long long t1 = pImpl->get_now_us(); // Wrapper Init (Part of Motion Est prep)
    #endif

    #if ENABLE_PERF_PROFILING
    long long t2 = pImpl->get_now_us(); // Motion Est (placeholder, actual ME is below blockBasedMotionEstimation)
    // Wait, blockBasedMotionEstimation is called later!
    // Moving t1, t2 logic...
    // The previous block was JUST image conversion.
    #endif

    // --- 0. Face Detection Integration ---
    #if ENABLE_PERF_PROFILING
    long long t_face_start = pImpl->get_now_us();
    #endif

    bool has_face = false;
    FaceROI face_roi = {0,0,0,0,0.0f};

    // Retry Init if failed in Config (Optional, or just rely on SetConfig)
    if (pImpl->config.enable_face_detection && !pImpl->face_model_inited) {
         // Try one more time? Or just skip?
         // Let's print warning once?
         // Or try to init again?
         StatusCode ret = pImpl->faceDetector.Init("res/blaze_face_detect_nnp310_128x128.ty");
         if (ret == StatusCode::OK) {
              pImpl->face_model_inited = true;
         } else {
             // Print error every 100 frames to avoid spam?
             // Just print error
             std::cout << "[FallDetector::Detect] FaceDetector not initialized (Retry Failed: " << (int)ret << ")" << std::endl;
         }
    }

    // Resize for Face Detection (Uses RGB frame)
    Image faceInput;
    // std::cout << "[FallDetector] Calling FaceDetector Resize..." << std::endl; // Commented out
    if (pImpl->config.enable_face_detection) {
        if (pImpl->faceDetector.Resize(frame, faceInput)) {
             // std::cout << "[FallDetector] Resize success. Calling Detect..." << std::endl;
             std::vector<FaceROI> faces;
             int ret = pImpl->faceDetector.Detect(faceInput, faces);
             // std::cout << "[FallDetector] Detect returned " << ret << " num_faces=" << faces.size() << std::endl;
             if (ret == 0 && !faces.empty()) {
                 has_face = true;
                 face_roi = faces[0]; 
             }
        } else {
            // std::cout << "[FallDetector] Resize failed!" << std::endl;
        }
    }
    
    #if ENABLE_PERF_PROFILING
    long long t_face_end = pImpl->get_now_us();
    pImpl->prof.face_detect_time += (t_face_end - t_face_start);
    #endif
    
    #if ENABLE_PERF_PROFILING
    long long t_me_start = pImpl->get_now_us();
    #endif

    // Run    // 2. Motion Estimation Execute
    printf("[Debug] Estimator->blockBasedMotionEstimation Start\n");
    pImpl->estimator->blockBasedMotionEstimation(wrapper, 
                                               pImpl->motion_vectors, 
                                               pImpl->positions, 
                                               pImpl->changed_mask, 
                                               pImpl->active_blocks, 
                                               pImpl->active_indices,
                                               pImpl->config.block_dilation_threshold);
    
    // NEW: Bed Region Foreground Masking (User Request)
    // "找完前景點時, 在床的區域內的點, 直接當背景點"
    if (pImpl->hasBedMask) {
        int cols = pImpl->config.grid_cols;
        int rows = pImpl->config.grid_rows;
        int bw = frame.width / cols;
        int bh = frame.height / rows;
        const unsigned char* bed_data = pImpl->bedMask.getData();
        int bed_w = pImpl->bedMask.width();
        int bed_h = pImpl->bedMask.height();

        for (size_t i = 0; i < pImpl->changed_mask.size(); ++i) {
            if (pImpl->changed_mask[i]) {
                int r = i / cols;
                int c = i % cols;
                int cx = c * bw + bw / 2;
                int cy = r * bh + bh / 2;
                
                // Check if center is in bed mask (0 = Inside)
                if (cx >= 0 && cx < bed_w && cy >= 0 && cy < bed_h) {
                    if (bed_data[cy * bed_w + cx] == 0) {
                        // Force to background
                        pImpl->changed_mask[i] = false;
                        
                        // Safety Check: active_blocks might not be same size
                        if (i < pImpl->active_blocks.size()) {
                            pImpl->active_blocks[i] = ::Image(); // Clear block image data
                        }
                    }
                }
            }
        }
    }
    

    #if ENABLE_PERF_PROFILING
    long long t_me_end = pImpl->get_now_us();
    pImpl->prof.motion_est_time += (t_me_end - t_me_start) + (t1 - t0); // Include conversion time
    #endif
    
    #if ENABLE_PERF_PROFILING
    long long t_logic_start = pImpl->get_now_us();
    #endif

    // 2. Extract Objects
    // 2. Extract Objects
    // Save previous
    pImpl->previous_objects = pImpl->current_objects;
    
    pImpl->current_objects = extractMotionObjects(pImpl->motion_vectors, 
                                                  pImpl->changed_mask,
                                                  pImpl->config.grid_rows, 
                                                  pImpl->config.grid_cols, 
                                                  pImpl->config.object_extraction_threshold, 
                                                  pImpl->config.object_merge_radius);
                                                  
    // TRACKING
    TrackObjects(pImpl->current_objects, 
                 pImpl->previous_objects, 
                 pImpl->config, 
                 pImpl->kalmanFilters, 
                 pImpl->track_ttl, 
                 pImpl->global_id_counter,
                 pImpl->object_first_seen_frame, 
                 pImpl->object_is_new_entry, 
                 pImpl->persistent_object_blocks, 
                 (int)pImpl->frame_idx);
    
    // INJECT LOST TRACKS (Coasting)
    // Ensures trend analysis sees valid data even if motion stops (High -> Low)
    std::set<int> current_ids;
    for(const auto& o : pImpl->current_objects) current_ids.insert(o.id);
    
    for(const auto& kv : pImpl->kalmanFilters) {
        int id = kv.first;
        if(current_ids.find(id) == current_ids.end()) {
             // Lost Track - Inject Predicted
             MotionObject predObj;
             predObj.id = id;
             float kx = 0, ky = 0;
             // Use const cast if GetState is non-const (it shouldn't be but safety first)
             // kv.second is a reference to value in map.
             auto& kf = kv.second; // copy or ref
             // kf.GetState(kx, ky); // GetState is void(float&, float&)
             kx = kf.x[0]; // Direct access as seen in TrackObjects
             ky = kf.x[1];
             float kvx = kf.x[2];
             float kvy = kf.x[3];
             
             predObj.centerX = kx;
             predObj.centerY = ky;
             predObj.avgDx = kvx;
             predObj.avgDy = kvy;
             predObj.strength = std::sqrt(kvx*kvx + kvy*kvy);
             
             // Restore Blocks
             if(pImpl->persistent_object_blocks.count(id)) {
                 predObj.blocks = pImpl->persistent_object_blocks[id];
             }
             
             // Only inject if it has blocks (history)
             if (!predObj.blocks.empty()) {
                  pImpl->current_objects.push_back(predObj);
                  printf("[FallDetector] Injected Coasting Object %d (Str: %.2f, Blocks: %zu)\n", id, predObj.strength, predObj.blocks.size());
             }
        }
    }
    
    // 2.1 Update Safe Area Ratio
    if (pImpl->hasBedMask) {
        int cols = pImpl->config.grid_cols; 
        int rows = pImpl->config.grid_rows;
        float blockW = (float)W / cols;
        float blockH = (float)H / rows;

        for (auto& obj : pImpl->current_objects) {
            int safe_blocks = 0;
            for (int blkIdx : obj.blocks) {
                int r = blkIdx / cols;
                int c = blkIdx % cols;
                int px = (int)((c + 0.5f) * blockW);
                int py = (int)((r + 0.5f) * blockH);
                if (px >= 0 && px < W && py >= 0 && py < H) {
                     if (pImpl->bedMask.at(px, py) == 0) safe_blocks++;
                }
            }
            if(!obj.blocks.empty())
                obj.safe_area_ratio = (float)safe_blocks / obj.blocks.size();
        }
    }

    // 2.5 Check Bed Exit (Temporal Analysis) - PER OBJECT
    if (pImpl->hasBedMask) {
        int cols = pImpl->config.grid_cols;
        int rows = pImpl->config.grid_rows;
        float blockW = (float)W / cols;
        float blockH = (float)H / rows;

        // Reset per-frame status? Or keep persistent? 
        // Logic says "Is Bed Exit detected NOW based on history"
        // So we re-evaluate per frame. 
        // Ideally we should CLEANUP history for objects that are gone. 
        // For simplicity, we iterate CURRENT objects.
        
        for (auto& obj : pImpl->current_objects) {
            int exit_blocks = 0;   // Blocks OUTSIDE bed
            int inside_blocks = 0; // Blocks INSIDE bed
            int total_motion_blocks = 0;

            // Only consider blocks belonging to THIS object
            for (int blkIdx : obj.blocks) {
                // Check if this block has significant motion (using motion vector)
                // We need to look up motion vector for this block index
                if (blkIdx < (int)pImpl->motion_vectors.size()) {
                    int dx = pImpl->motion_vectors[blkIdx].dx;
                    int dy = pImpl->motion_vectors[blkIdx].dy;
                    
                    if(dx*dx + dy*dy > 2*2) { 
                        total_motion_blocks++;
                        int r = blkIdx / cols;
                        int c = blkIdx % cols;
                        int px = (int)((c + 0.5f) * blockW);
                        int py = (int)((r + 0.5f) * blockH);
                        
                        if (px >= 0 && px < W && py >= 0 && py < H) {
                            if (pImpl->bedMask.at(px, py) == 255) {
                                exit_blocks++;
                            } else {
                                inside_blocks++;
                            }
                        }
                    }
                }
            }

            // Calculate Ratios
            float inside_ratio = 0.0f;
            float outside_ratio = 0.0f;
            
            if (total_motion_blocks > 2) { // Minimum 3 blocks of motion to be valid stats?
                inside_ratio = (float)inside_blocks / total_motion_blocks;
                outside_ratio = (float)exit_blocks / total_motion_blocks;
            }
            
            // Update History for this Object
            std::deque<std::pair<float, float>>& history = pImpl->object_bed_stats_history[obj.id];
            history.push_back({inside_ratio, outside_ratio});
            
            if ((int)history.size() > pImpl->config.bed_exit_history_len) {
                history.pop_front();
            }
            
            // Analyze Trend
            pImpl->object_bed_exit_status[obj.id] = false; // Default false for this frame
            
            int required_len = pImpl->config.bed_exit_history_len;
            if ((int)history.size() >= required_len * 0.8) {
                 int history_size = history.size();
                 int half_idx = history_size / 2;
                 
                 float avg_inside_old = 0.0f, avg_outside_old = 0.0f;
                 float avg_inside_new = 0.0f, avg_outside_new = 0.0f;
                 
                 int count_old = 0, count_new = 0;
                 for(int i=0; i<history_size; ++i) {
                     if (i < half_idx) {
                         avg_inside_old += history[i].first;
                         avg_outside_old += history[i].second;
                         count_old++;
                     } else {
                         avg_inside_new += history[i].first;
                         avg_outside_new += history[i].second;
                         count_new++;
                     }
                 }
                 if (count_old > 0) { avg_inside_old /= count_old; avg_outside_old /= count_old; }
                 if (count_new > 0) { avg_inside_new /= count_new; avg_outside_new /= count_new; }
                 
                 bool start_inside = (avg_inside_old > avg_outside_old);
                 bool end_outside = (avg_outside_new > avg_inside_new);
                 
                 if (start_inside && end_outside) {
                     pImpl->object_bed_exit_status[obj.id] = true;
                     // printf("[Debug] Obj %d Bed Exit Confirmed (Frame %lld)\n", obj.id, pImpl->absolute_frame_count);
                 }
            }
        }
        
        // Cleanup old objects? 
        // We can check if object_bed_stats_history contains IDs not in current_objects.
        // For efficiency, maybe do it every N frames or just let it grow (IDs are unique). 
        // Since global_id_counter increases, this map WILL grow indefinitely. We MUST clean up.
        // Simple Cleanup:
        /*
        for (auto it = pImpl->object_bed_stats_history.begin(); it != pImpl->object_bed_stats_history.end(); ) {
             bool found = false;
             for(const auto& obj : pImpl->current_objects) if(obj.id == it->first) { found = true; break; }
             if(!found) {
                 pImpl->object_bed_exit_status.erase(it->first);
                 it = pImpl->object_bed_stats_history.erase(it);
             } else {
                 ++it;
             }
        }
        */
        // Actually, we should keep history for a bit? No, if object is gone, history is irrelevant.
        // But `Detect` might track objects across frames. If tracking is lost, ID changes.
    }

    // 3. Fall Logic
    std::string warning = "";
    float max_strength = 0.0f;
    
    // =========================================================
    // NEW: Compute Pixel Stats (Count & Brightness)
    // =========================================================
    const unsigned char* currData = frame.data;
    const unsigned char* bgData = pImpl->backgroundFrame.empty() ? nullptr : pImpl->backgroundFrame.getData();
    // Use existing W, H declarations
    // int W = frame.width;
    // int H = frame.height;
    int bg_thresh = pImpl->config.bg_diff_threshold;
    
    // DEBUG PRINT
    // printf("[Detect] Frame %lld. W=%d H=%d. BG=%p\n", pImpl->absolute_frame_count, W, H, bgData);

    if (bgData) 
    {
        int grid_cols = pImpl->config.grid_cols;
        int grid_rows = pImpl->config.grid_rows;
        int bw = W / grid_cols;
        int bh = H / grid_rows;

        // Efficient Block-Based FG Scoring (Stationary-Aware)
        std::vector<int> block_fg_scores(grid_rows * grid_cols, 0);
        std::vector<bool> blocks_to_scan(grid_rows * grid_cols, false);

        // A. Mark motion blocks
        for (size_t i = 0; i < pImpl->changed_mask.size(); ++i) {
            if (pImpl->changed_mask[i]) blocks_to_scan[i] = true;
        }

        // B. Mark persistent object blocks (even if stationary)
        for (auto const& kv : pImpl->persistent_object_blocks) {
            for (int blk : kv.second) {
                if (blk >= 0 && blk < (int)blocks_to_scan.size()) blocks_to_scan[blk] = true;
            }
        }

        // C. Perform Pixel Scanning
        for (int i = 0; i < grid_rows * grid_cols; ++i) {
            if (!blocks_to_scan[i]) continue;

            int r = i / grid_cols;
            int c = i % grid_cols;
            int startX = c * bw;
            int startY = r * bh;
            int endX = std::min(W, startX + bw);
            int endY = std::min(H, startY + bh);
            
            int count = 0;
            for (int y = startY; y < endY; ++y) {
                for (int x = startX; x < endX; ++x) {
                    int idx = (y * W + x) * 3;
                    int diff = 0;
                    if (pImpl->backgroundFrame.getChannels() == 1) {
                        int bgVal = bgData[y * W + x];
                        int grayVal = (currData[idx] + currData[idx+1]*2 + currData[idx+2]) / 4;
                        diff = std::abs(grayVal - bgVal);
                    } else {
                        // BGR diff
                        diff = (std::abs((int)currData[idx] - (int)bgData[idx]) +
                                std::abs((int)currData[idx+1] - (int)bgData[idx+1]) +
                                std::abs((int)currData[idx+2] - (int)bgData[idx+2])) / 3;
                    }

                    if (diff > bg_thresh) {
                        count++;
                    }
                }
            }
            block_fg_scores[i] = count;
            
            // Still contribute to global count if it was a motion block
            if (pImpl->changed_mask[i]) global_fg_count += count;
        }

        // D. Ghost Object Injection (Maintain stationary objects in current_objects)
        std::set<int> current_ids;
        for (const auto& obj : pImpl->current_objects) current_ids.insert(obj.id);

        for (auto const& kv : pImpl->persistent_object_blocks) {
            int id = kv.first;
            if (current_ids.find(id) == current_ids.end()) {
                // This ID is tracked but has NO motion this frame. Inject as ghost.
                MotionObject ghost;
                ghost.id = id;
                ghost.blocks = kv.second;
                ghost.strength = 0.0f;
                ghost.acceleration = 0.0f;
                
                // Get position from Kalman or last seen?
                if (pImpl->kalmanFilters.count(id)) {
                    float kx, ky;
                    pImpl->kalmanFilters.at(id).GetState(kx, ky);
                    ghost.centerX = kx; ghost.centerY = ky;
                } else {
                    // Fallback to center of blocks
                    float sumX = 0, sumY = 0;
                    for (int blk : ghost.blocks) {
                        sumX += (blk % grid_cols) + 0.5f;
                        sumY += (blk / grid_cols) + 0.5f;
                    }
                    ghost.centerX = sumX / ghost.blocks.size();
                    ghost.centerY = sumY / ghost.blocks.size();
                }
                pImpl->current_objects.push_back(ghost);
            }
        }

        for (auto& obj : pImpl->current_objects) {
            obj.total_frame_pixel_count = global_fg_count; // Assign to object
            
            // Calculate local FG count for this object
            obj.pixel_count = 0;
            for (int blk : obj.blocks) {
                if (blk >= 0 && blk < (int)block_fg_scores.size()) {
                    obj.pixel_count += block_fg_scores[blk];
                }
            }
            
            // V17.1: Update Global Object trajectory history
            auto& y_hist = pImpl->object_y_history_buffer[obj.id];
            y_hist.push_back(obj.centerY);
            if (y_hist.size() > 120) y_hist.pop_front();
            
            // V17.5: Update Global Safe Ratio history
            auto& sr_hist = pImpl->object_safe_ratio_history_buffer[obj.id];
            sr_hist.push_back(obj.safe_area_ratio);
            if (sr_hist.size() > 120) sr_hist.pop_front();

            long long sum_brightness = 0;
            int count_pixels = 0;
            
            // CONVEX HULL PIXEL COUNTING LOGIC
            // 1. Collect Block Coordinates
            // CONVEX HULL PIXEL COUNTING LOGIC
            // 1. Collect Block Coordinates
            struct Point { double x, y; }; // Changed to double
            std::vector<Point> blockPts;
            int grid_cols = pImpl->config.grid_cols;
            int grid_rows = pImpl->config.grid_rows;
            int bw = W / grid_cols;
            int bh = H / grid_rows;
            
            int min_r = grid_rows, max_r = 0;
            int min_c = grid_cols, max_c = 0;
            
            for (int blkIdx : obj.blocks) {
                int r = blkIdx / grid_cols;
                int c = blkIdx % grid_cols;
                if(r < min_r) min_r = r;
                if(r > max_r) max_r = r;
                if(c < min_c) min_c = c;
                if(c > max_c) max_c = c;
                
                // Push 4 Corners
                blockPts.push_back({(double)c, (double)r});
                blockPts.push_back({(double)c+1, (double)r});
                blockPts.push_back({(double)c, (double)r+1});
                blockPts.push_back({(double)c+1, (double)r+1});
            }
            
            // 2. Compute Convex Hull (Monotone Chain)
            // Sort by x then y
            if (blockPts.size() > 2) {
                 std::sort(blockPts.begin(), blockPts.end(), [](const Point& a, const Point& b){
                     return a.x < b.x || (a.x == b.x && a.y < b.y);
                 });
                 
                 std::vector<Point> hull;
                 // Lower chain
                 for(const auto& p : blockPts) {
                     while(hull.size() >= 2) {
                         const Point& o = hull[hull.size()-2];
                         const Point& a = hull.back();
                         // Cross product
                         double cp = (a.x - o.x) * (p.y - o.y) - (a.y - o.y) * (p.x - o.x);
                         if (cp <= 0) hull.pop_back(); else break;
                     }
                     hull.push_back(p);
                 }
                 // Upper chain
                 int lower_len = hull.size();
                 for(int i=(int)blockPts.size()-2; i>=0; --i) {
                     const Point& p = blockPts[i];
                     while(hull.size() > lower_len) {
                         const Point& o = hull[hull.size()-2];
                         const Point& a = hull.back();
                         double cp = (a.x - o.x) * (p.y - o.y) - (a.y - o.y) * (p.x - o.x);
                         if (cp <= 0) hull.pop_back(); else break;
                     }
                     hull.push_back(p);
                 }
                 hull.pop_back(); // Remove duplicate last
                 blockPts = hull; // Use hull as points
            }
            
            // 3. Scan Bounding Box in Image Space
            // Expand BB slightly
            int startX = std::max(0, (int)(min_c * bw));
            int startY = std::max(0, (int)(min_r * bh));
            int endX = std::min(W, (int)((max_c + 1) * bw));
            int endY = std::min(H, (int)((max_r + 1) * bh));
            
            for(int y=startY; y<endY; ++y) {
                // Optimization: Compute Grid Row once
                int gr = y / bh;
                
                for(int x=startX; x<endX; ++x) {
                    int gc = x / bw;
                    
                    // Point In Hull (Check Center of Grid Cell)
                    double testX = gc + 0.5;
                    double testY = gr + 0.5;
                    
                    bool inside = false;
                    size_t n = blockPts.size();
                    if (n < 3) {
                        // Fallback
                        inside = true; 
                    } else {
                        // Ray casting algorithm
                        for (size_t i = 0, j = n - 1; i < n; j = i++) {
                            if (((blockPts[i].y > testY) != (blockPts[j].y > testY)) &&
                                (testX < (blockPts[j].x - blockPts[i].x) * (testY - blockPts[i].y) / (blockPts[j].y - blockPts[i].y) + blockPts[i].x)) {
                                inside = !inside;
                            }
                        }
                    }
                    
                    if (inside) {
                        int idx = (y * W + x) * 3; 
                        // Check Foreground
                        int diff = 0;
                        
                        if (pImpl->backgroundFrame.getChannels() == 1) {
                            // BG is Gray, Input is RGB
                            int bgVal = bgData[y * W + x];
                            // Simple average or G channel?
                            int grayVal = (currData[idx] + currData[idx+1]*2 + currData[idx+2]) / 4;
                            diff = std::abs(grayVal - bgVal);
                        } else {
                            // BG is RGB
                            diff += std::abs((int)currData[idx] - (int)bgData[idx]);
                            diff += std::abs((int)currData[idx+1] - (int)bgData[idx+1]);
                            diff += std::abs((int)currData[idx+2] - (int)bgData[idx+2]);
                            diff /= 3;
                        }
                        
                        // Check Foreground
                        if (diff > bg_thresh) {
                            count_pixels++;
                            int bri = (currData[idx] + currData[idx+1]*2 + currData[idx+2])/4;
                            sum_brightness += bri;
                        }
                    }
                }
            }
            
            obj.pixel_count = count_pixels;
            obj.avg_brightness = (count_pixels > 0) ? (float)sum_brightness / count_pixels : 0.0f;
        }
    }
    // printf("[Detect] Pixel Stats Done.\n");

    // Update object history
    pImpl->object_history.push_back(pImpl->current_objects);
    // Ensure we keep enough history for the new 10+10 logic (at least 30-40 frames)
    int min_history = 40; 
    int limit = std::max(pImpl->config.fall_window_size, min_history);
    
    // 5. Tracking for Slow Falls (Accumulated Descent)
    int triggered = 0; // 0=None, 1=Fast
    int slow_triggered = 0; // 0=None, 1=Slow
    int detected_id = -1;
    std::vector<int> slow_fall_ids; // NEW: Track slow falls
    
    std::string fall_type = "";
    int case_num = 0;


    // --- NEW LOGIC: Direction-Based Detection (User Request) ---
    // 1. Global Motion Safety Guard: Reject if > 1/4 screen is moving
    int total_grid_blocks = pImpl->config.grid_cols * pImpl->config.grid_rows;
    if (pImpl->active_blocks.size() > (size_t)(total_grid_blocks / 4)) {
        // Safe: Too much motion (Camera move / Light switch)
        // printf("[FallDetector] Global Safety: Too much motion (%zu blocks)\n", pImpl->active_blocks.size());
        triggered_objects.clear(); 
    } 
    else if (!pImpl->current_objects.empty()) 
    {
        // 2. Global FG History Update
        pImpl->fg_count_history_buffer.push_back((int)pImpl->active_blocks.size());
        if(pImpl->fg_count_history_buffer.size() > 100) pImpl->fg_count_history_buffer.pop_front();

        // 3. Object Analysis
        for (auto& curr : pImpl->current_objects) {
            // Reconstruct History for this object
            // User requirement: "Recently n frames... m frames momentum trend"
            // Let's use n=20, m=10 for trends.
            int n_history = 20; 
            std::vector<float> hist_dy, hist_dx, hist_str;
            std::vector<int> hist_fg; // Global FG corresponding to object history frames
            
            // Collect history (Oldest -> Newest)
            // pImpl->object_history is [Old ... New] ? No, usually buffer is push_back.
            // Index 0 is oldest? Or newest? 
            // Standard implementation: push_back new frame. So back() is newest.
            // We iterate from back-n to back.
            
            if (pImpl->object_history.size() < 5) continue; 
            
            int collected = 0;
            int h_size = pImpl->object_history.size();
            int start_idx = std::max(0, h_size - n_history);

            for(int i=start_idx; i<h_size; ++i) {
                bool found = false;
                for(const auto& h_obj : pImpl->object_history[i]) {
                    if(h_obj.id == curr.id) {
                        hist_dy.push_back(h_obj.avgDy);
                        hist_dx.push_back(h_obj.avgDx);
                        hist_str.push_back(h_obj.strength);
                        found = true;
                        break;
                    }
                }
                if(found) {
                     // Get Global FG for this frame index (Synced?)
                     // pImpl->fg_count_history_buffer is updated every frame.
                     // Assuming sync: global buffer size increases same as obj history size if start together.
                     // But buffer is deque. 
                     // Let's use relative index from end.
                     int offset_from_end = (h_size - 1) - i;
                     int fg_idx = (int)pImpl->fg_count_history_buffer.size() - 1 - offset_from_end;
                     if (fg_idx >= 0 && fg_idx < (int)pImpl->fg_count_history_buffer.size()) {
                         hist_fg.push_back(pImpl->fg_count_history_buffer[fg_idx]);
                     } else {
                         hist_fg.push_back(0);
                     }
                     collected++;
                }
            }
            
            if (collected < 10) continue; // Need enough history

            // --- Case Determination ---
            // "Calculate momentum direction of last n frames"
            // Let's take average or sum
            float sum_dy = 0, sum_dx = 0;
            for(float v : hist_dy) sum_dy += v;
            for(float v : hist_dx) sum_dx += v;
            
            bool y_dominant = (std::abs(sum_dy) > std::abs(sum_dx));
            bool is_upward = (sum_dy < 0);
            
            // --- Baseline Stats for Debugging Missed Falls ---
            float avg_h_str = 0, avg_l_str = 0;
            {
                int len = hist_str.size();
                int half = len / 2;
                float s_h = 0, s_l = 0;
                for(int k=0; k<half; ++k) s_h += hist_str[k];
                for(int k=half; k<len; ++k) s_l += hist_str[k];
                avg_h_str = s_h / std::max(1, half);
                avg_l_str = s_l / std::max(1, len-half);
            }
            float s_h_fg = 0, s_l_fg = 0;
            {
                int len = hist_fg.size();
                int half = len / 2;
                for(int k=0; k<half; ++k) s_h_fg += hist_fg[k];
                for(int k=half; k<len; ++k) s_l_fg += hist_fg[k];
            }
            int box_w = 0, box_h = 0, m1, m2, m3, m4;
            getObjectBoundingBoxPixels(curr, pImpl->config.grid_cols, pImpl->config.grid_rows, W, H, box_w, box_h, m1, m2, m3, m4);

            printf("[DET_LOG] F:%lld ID:%d YDom:%d Up:%d Dy:%.1f StrH:%.2f StrL:%.2f (R:%.2f) FGH:%.0f FGL:%.0f (R:%.2f) Box:%dx%d\n",
                   pImpl->frame_idx, curr.id, (int)y_dominant, (int)is_upward, sum_dy, 
                   avg_h_str, avg_l_str, (avg_h_str > 0 ? avg_l_str/avg_h_str : 0.0f),
                   s_h_fg, s_l_fg, (s_h_fg > 0 ? s_l_fg/s_h_fg : 0.0f), box_w, box_h);

            bool potential_fall = false;
            
            
            // Helper: Trend Check (High -> Low)
            auto checkTrend = [](const std::vector<float>& data, float high_thresh, float drop_ratio) -> bool {
                int len = data.size();
                int half = len / 2;
                float sum_high = 0, sum_low = 0;
                for(int k=0; k<half; ++k) sum_high += data[k];
                for(int k=half; k<len; ++k) sum_low += data[k];
                
                float avg_high = sum_high / half;
                float avg_low = sum_low / (len - half);
                
                return (avg_high > high_thresh && avg_low < avg_high * drop_ratio);
            };

            // Helper: Global FG Trend Check (High -> Low)
            auto checkFGTrend = [](const std::vector<int>& data) -> bool {
                int len = data.size();
                int half = len / 2;
                float sum_high = 0, sum_low = 0;
                for(int k=0; k<half; ++k) sum_high += data[k];
                for(int k=half; k<len; ++k) sum_low += data[k];
                return (sum_low < sum_high * 0.8f); // Default 80% drop
            };

            // Lower mom_high_thresh to 4.0
            float mom_high_thresh = 4.0f; // Relaxed from 6.0
            float mom_drop_ratio = 0.8f; // Relaxed from 0.5
            float fg_drop_ratio = 0.8f; // Relaxed from 0.75
            
            
            /* ============================================================
             * CASE 1-4: COMMENTED OUT - Replaced by Case 5
             * ============================================================
            if (y_dominant && !is_upward) {
                // --- CASE 1: Upward Momentum ---
                // 1-a. Consistency Check (Reject if Consistent)
                // Using Direction Variance from Object if available, or calc from history
                float dir_var = curr.direction_variance; // Existing logic computes this
                if (dir_var < 0.2f) { // Consistent -> Walking Away
                    // printf("[NewLogic] ID %d Rejected: Upward Consistent (Var %.2f)\n", curr.id, dir_var);
                    continue; 
                }
                
                // 1-b. Momentum Trend (High -> Low)
                // 1-c. FG Trend (High -> Low)
                if (checkTrend(hist_str, mom_high_thresh, mom_drop_ratio) && checkFGTrend(hist_fg)) {
                     
                     
                     // Get bounding box size for this object
                     int bbox_w, bbox_h, min_x, min_y, max_x, max_y;
                     getObjectBoundingBoxPixels(curr, pImpl->config.grid_cols, pImpl->config.grid_rows, 
                                                W, H, bbox_w, bbox_h, min_x, min_y, max_x, max_y);
                     printf("[Case1 BBox] ID %d: %dx%d pixels (min=%d,%d max=%d,%d)\n", 
                            curr.id, bbox_w, bbox_h, min_x, min_y, max_x, max_y);

                    if (bbox_w < 240 && bbox_h < 240) {
                        potential_fall = true;
                        fall_type = "Upward_Collapse";
                        case_num = 1;
                    }
                }
            } 
            else if (y_dominant && is_upward) {
                // --- CASE 2: Downward Momentum ---
                // 2-a. Momentum Trend (High -> Low)
                // 2-b. FG Trend (High -> Low)
                if (checkTrend(hist_str, mom_high_thresh, mom_drop_ratio) && checkFGTrend(hist_fg)) {
                     potential_fall = true;
                     fall_type = "Downward_Fall";
                     case_num = 2;
                }
                
                // --- CASE 3: Pixel-Dominant Trigger (Low Momentum / Sit / Roll) ---
                // If momentum is weak (e.g. 1.0 < str < 4.0) but FG Drop is HUGE (e.g. > 45%), trigger it.
                // Guard: Must NOT be moving Up (Walking Away). avgDy should be >= 0 or very slightly negative.
                if (!potential_fall) {
                    float strict_fg_ratio = 0.55f; // Require 45% drop
                    bool is_huge_fg_drop = false;
                    {
                        int len = hist_fg.size();
                        int half = len / 2;
                        float sum_high = 0, sum_low = 0;
                        for(int k=0; k<half; ++k) sum_high += hist_fg[k];
                        for(int k=half; k<len; ++k) sum_low += hist_fg[k];
                        if (sum_high > 100 && sum_low < sum_high * strict_fg_ratio) is_huge_fg_drop = true;
                    }

                    // Check if "Not Walking Away" (Y-momentum not significantly negative)
                    bool not_walking_away = (sum_dy > -5.0f); // Allow slight noise, but generally neutral or down.

                    if (is_huge_fg_drop && not_walking_away) {
                        // Check if ANY motion exists (don't trigger on static light switch)
                        bool has_some_motion = false;
                        float max_str = 0.0f;
                        for(float s : hist_str) if(s > 1.5f) { has_some_motion = true; } // Low threshold
                        
                        if (has_some_motion) {
                            potential_fall = true;
                            fall_type = "Pixel_Collapse";
                            case_num = 3;
                             printf("[NewLogic] Pixel-Dominant Trigger! ID %d. FG Drop Confirmed. (SumDy: %.1f)\n", curr.id, sum_dy);
                        }
                    }
                }
                
                // --- CASE 4: Sustained FG Decline (Peak-to-Current, Any Direction) ---
                // Data3 pattern: FG gradually declines over extended period
                // Strategy: Within 20-frame window, check if current FG dropped significantly from peak
                if (!potential_fall) {
                    printf("[C4_ENTRY] ID:%d entering Case 4 check\n", curr.id);
                    int len = hist_fg.size();
                    printf("[C4_DEBUG] ID:%d hist_fg.size=%d (need>=10)\n", curr.id, len);
                    if (len >= 10) {  // Need sufficient history
                        // Find peak in first 60% of window and current minimum in last 40%
                        int peak_search_end = (int)(len * 0.6f);
                        int recent_search_start = (int)(len * 0.6f);
                        
                        float peak_fg = 0;
                        float recent_sum = 0;
                        int recent_count = 0;
                        
                        for(int k=0; k < peak_search_end; ++k) {
                            if (hist_fg[k] > peak_fg) peak_fg = hist_fg[k];
                        }
                        
                        // Use AVERAGE (not minimum) to avoid 0-value problem
                        // Only count non-zero values
                        for(int k=recent_search_start; k < len; ++k) {
                            if (hist_fg[k] > 0) {
                                recent_sum += hist_fg[k];
                                recent_count++;
                            }
                        }
                        
                        float recent_avg_fg = (recent_count > 0) ? (recent_sum / recent_count) : 0.0f;
                        float drop_ratio = (peak_fg > 100 && recent_avg_fg > 0) ? ((peak_fg - recent_avg_fg) / peak_fg) : 0.0f;
                        
                        // DEBUG: Check thresholds
                        bool drop_ok = (drop_ratio >= 0.45f);
                        bool peak_ok = (peak_fg > 3000);
                        bool avg_ok = (recent_avg_fg > 2000);
                        
                        if (peak_ok && !drop_ok) {
                            printf("[C4_FAIL] ID:%d peak:%.0f avg:%.0f drop:%.1f%% (need>=45%%) FAILED DROP\\n", 
                                   curr.id, peak_fg, recent_avg_fg, drop_ratio * 100);
                        } else if (peak_ok && drop_ok && !avg_ok) {
                            printf("[C4_FAIL] ID:%d peak:%.0f avg:%.0f drop:%.1f%% FAILED AVG (need>2000)\\n", 
                                   curr.id, peak_fg, recent_avg_fg, drop_ratio * 100);
                        }
                        
                        // Trigger if:
                        // 1. Significant drop (>= 45%) - relaxed to catch data3
                        // 2. Peak was meaningful (> 3000) - lowered from 5000
                        // 3. Recent avg is NOT near zero (> 2000) - prevents "leaving scene" false positives
                        if (drop_ratio >= 0.45f && peak_fg >3000 && recent_avg_fg > 2000) {
                            // Additional guard: some movement
                            float avg_str = 0;
                            for(float s : hist_str) avg_str += s;
                            avg_str /= hist_str.size();
                            
                            if (avg_str > 1.0f) {
                                potential_fall = true;
                                fall_type = "Sustained_Decline";
                                case_num = 4;
                                printf("[NewLogic] Peak-to-Current FG Decline! ID %d peak:%.0f avg:%.0f drop:%.1f%%\n", 
                                       curr.id, peak_fg, recent_avg_fg, drop_ratio * 100);
                            }
                        }
                    }
                }
            }
            * ============================================================ */
            
            // --- CASE 5: Momentum Transition Detection (Two-Phase) ---
            // Phase 1: Wait for deceleration (momentum drops below threshold)
            // Phase 2: Observe 60 frames of sustained low momentum
            
            // Step 1: Calculate recent 3-frame momentum average
            int recent_window = std::min(3, (int)hist_str.size());
            if (recent_window > 0) {
                float recent_mom_sum = 0.0f;
                for (int k = hist_str.size() - recent_window; k < (int)hist_str.size(); ++k) {
                    recent_mom_sum += hist_str[k];
                }
                float recent_mom_avg = recent_mom_sum / recent_window;

                const float threshold1 = 5.5f;  // High momentum trigger (peak detection)
                const float decel_threshold = 4.0f;  // Deceleration complete threshold
                const float threshold2 = 5.5f;  // Low momentum threshold - Balanced for ~90% TP retention
                const int observation_frames = 120;  // Extended from 60 to 120 frames (4 seconds @ 30fps)
                const int max_waiting_frames = 30;  // Max frames to wait for deceleration

                auto& state = pImpl->observation_states[curr.id];

                // Step 2: Trigger on high momentum peak
                if (!state.is_active && !state.waiting_for_deceleration && recent_mom_avg >= threshold1) {
                    // Enter waiting phase - wait for momentum to drop
                    state.waiting_for_deceleration = true;
                    state.object_id = curr.id;
                    state.trigger_frame = pImpl->frame_idx;
                    state.frames_waiting = 0;
                    state.frames_observed = 0;
                    state.momentum_samples.clear();
                    printf("[Case5] ID:%d PEAK detected at frame %d (peak_mom=%.2f >= %.2f) - waiting for deceleration\n", 
                           curr.id, pImpl->frame_idx, recent_mom_avg, threshold1);
                    pImpl->LogTrace(curr.id, pImpl->frame_idx, "TRIGGER", recent_mom_avg, "Peak Momentum > 5.5");
                }

                // Step 3: Waiting phase - check if deceleration is complete
                if (state.waiting_for_deceleration && state.object_id == curr.id) {
                    state.frames_waiting++;
                    
                    // Check if momentum has dropped below deceleration threshold
                    if (curr.strength < decel_threshold) {
                        // Deceleration complete - start observation phase
                        state.waiting_for_deceleration = false;
                        state.is_active = true;
                        state.frames_observed = 0;
                        state.momentum_samples.clear();
                        printf("[Case5] ID:%d DECELERATION complete at frame %d (curr_mom=%.2f < %.2f) - starting observation\n", 
                               curr.id, pImpl->frame_idx, curr.strength, decel_threshold);
                    } else if (state.frames_waiting >= max_waiting_frames) {
                        // Timeout - momentum didn't drop in time, reset
                        state.waiting_for_deceleration = false;
                        printf("[Case5] ID:%d waiting TIMEOUT at frame %d (waited %d frames, mom still %.2f)\n", 
                               curr.id, pImpl->frame_idx, state.frames_waiting, curr.strength);
                    }
                }

                // Step 4: Observation phase - collect momentum samples
                if (state.is_active && !state.waiting_for_deceleration && state.object_id == curr.id) {
                    state.momentum_samples.push_back(curr.strength);
                    state.frames_observed++;
                    
                    // Step 4.0: Aspect Ratio Check (Flat Posture)
                    int box_w = 0, box_h = 0, m1, m2, m3, m4;
                    getObjectBoundingBoxPixels(curr, pImpl->config.grid_cols, pImpl->config.grid_rows, frame.width, frame.height, box_w, box_h, m1, m2, m3, m4);
                    if (box_h > 0) {
                        float aspect_ratio = (float)box_w / box_h;
                        if (aspect_ratio > 1.0f) {
                            state.flat_posture_frames++;
                        }
                    }

                    // Step 4.0b: Bed Ratio Accumulation
                    // Calculate ratio of blocks in bed
                    if (pImpl->hasBedMask) {
                        int bed_blocks = 0;
                        int blk_w = pImpl->config.block_size; // Assuming square blocks approx for center
                        int cols = pImpl->config.grid_cols;
                        const unsigned char* bed_data = pImpl->bedMask.getData();
                        int W = pImpl->bedMask.width();
                        int H = pImpl->bedMask.height();
                        
                        for (int blk_idx : curr.blocks) {
                            int r = blk_idx / cols;
                            int c = blk_idx % cols;
                            int px = c * (frame.width / cols) + (frame.width / cols) / 2;
                            int py = r * (frame.height / pImpl->config.grid_rows) + (frame.height / pImpl->config.grid_rows) / 2;
                            
                            if (px >= 0 && px < W && py >= 0 && py < H) {
                                if (bed_data[py * W + px] == 0) { // 0 is Inside Bed
                                    bed_blocks++;
                                }
                            }
                        }
                        if (!curr.blocks.empty()) {
                            float ratio = (float)bed_blocks / curr.blocks.size();
                            state.accumulated_bed_ratio += ratio;
                        }
                    }


                    
                    // Step 4.1: Continuous Perspective Check (Accumulate stats)
                    if (state.frames_observed % 5 == 0) { // Check every 5 frames to save CPU? Or every frame? Let's do every frame for now or every 2nd.
                         // Every frame is fine, find_objects_optimized is already called.
                         
                         if (!pImpl->full_frame_objects.empty()) {
                            // Calculate scaling factors (Pixels per Block)
                            float blk_w = (float)frame.width / pImpl->config.grid_cols;
                            float blk_h = (float)frame.height / pImpl->config.grid_rows;
                            
                            // Find matching object in full_frame_objects
                            // Compare in BLOCK coordinates
                            float min_dist = 999999.0f;
                            int best_idx = -1;
                            
                            for (size_t i = 0; i < pImpl->full_frame_objects.size(); ++i) {
                                // Convert FG object (Pixels) to Blocks
                                float fg_blk_cx = pImpl->full_frame_objects[i].cx / blk_w;
                                float fg_blk_cy = pImpl->full_frame_objects[i].cy / blk_h;
                                
                                float dx = fg_blk_cx - curr.centerX;
                                float dy = fg_blk_cy - curr.centerY;
                                float dist = dx*dx + dy*dy;
                                
                                if (dist < min_dist) {
                                    min_dist = dist;
                                    best_idx = (int)i;
                                }
                            }
                            
                            // printf("[Case5-DEBUG] ID:%d F:%lld best_idx=%d min_dist=%.2f\n", curr.id, pImpl->frame_idx, best_idx, min_dist);

                            if (best_idx >= 0 && min_dist < 25.0f) {  // Within 5 grid blocks
                                auto& fg_obj = pImpl->full_frame_objects[best_idx];
                                
                                // Calculate perspective line angle in PIXELS
                                float persp_px_x = (416.0f / 800.0f) * frame.width;
                                float persp_px_y = (474.0f / 450.0f) * frame.height;
                                
                                float curr_px_x = curr.centerX * blk_w;
                                float curr_px_y = curr.centerY * blk_h;
                                
                                float dx_to_persp = persp_px_x - curr_px_x;
                                float dy_to_persp = persp_px_y - curr_px_y;
                                float persp_angle = std::atan2(dy_to_persp, dx_to_persp) * 180.0f / 3.14159f;
                                
                                float obj_angle = fg_obj.angle * 180.0f / 3.14159f;
                                
                                float angle_diff = std::abs(obj_angle - persp_angle);
                                if (angle_diff > 180.0f) angle_diff = 360.0f - angle_diff;
                                if (angle_diff > 90.0f) angle_diff = 180.0f - angle_diff;
                                
                                state.valid_angle_frames++;
                                if (angle_diff < 30.0f) {
                                    state.aligned_frames++;
                                }
                            }
                         }
                    }

                    // Step 4.2: After observation_frames, evaluate
                    if (state.frames_observed >= observation_frames) {
                        float obs_mom_sum = 0.0f;
                        float obs_min_mom = 999.0f;
                        float obs_max_mom = -1.0f;
                        
                        for (float m : state.momentum_samples) {
                            obs_mom_sum += m;
                            if(m < obs_min_mom) obs_min_mom = m;
                            if(m > obs_max_mom) obs_max_mom = m;
                        }
                        float obs_mom_avg = obs_mom_sum / state.momentum_samples.size();
                        
                        // Adaptive Threshold based on Posture
                        float current_threshold2 = threshold2;
                        bool is_flat_posture = (state.flat_posture_frames > (state.frames_observed * 0.5f));
                        
                        if (is_flat_posture) {
                            current_threshold2 = 11.0f; // Relaxed threshold for struggling/seizure (high momentum but lying down)
                            printf("[Case5-DEBUG] ID:%d ADAPTIVE THRESHOLD: Flat Ratio=%.2f (>0.5) (Flat: %d/%d) -> Threshold=11.0\n",
                                   curr.id, (float)state.flat_posture_frames/state.frames_observed, state.flat_posture_frames, state.frames_observed);
                        }
                        
                        if (obs_mom_avg < current_threshold2) {
                            
                            // Bed Filter Check
                            bool is_bed_event = false;
                            float avg_bed_ratio = 0.0f;
                            int center_in_bed_flag = 0;

                            if (pImpl->hasBedMask) {
                                avg_bed_ratio = state.accumulated_bed_ratio / state.frames_observed;
                                
                                // Check Center
                                int cx = (int)(curr.centerX * (frame.width / pImpl->config.grid_cols)); 
                                int cy = (int)(curr.centerY * (frame.height / pImpl->config.grid_rows));
                                
                                if (cx >= 0 && cx < pImpl->bedMask.width() && cy >= 0 && cy < pImpl->bedMask.height()) {
                                    if (pImpl->bedMask.getData()[cy * pImpl->bedMask.width() + cx] == 0) {
                                        center_in_bed_flag = 1;
                                    }
                                }
                                
                                if (center_in_bed_flag || avg_bed_ratio > 0.30f) {
                                    is_bed_event = true;
                                     printf("[Case5] ID:%d Rejected: Bed Event (Center:%d, AvgRatio:%.2f > 0.30)\n", 
                                            curr.id, center_in_bed_flag, avg_bed_ratio);
                                     pImpl->LogTrace(curr.id, pImpl->frame_idx, "FILTER_BED", avg_bed_ratio, "Bed Region Reject");
                                }
                            }

                            // NEW: Perspective Line Angle Check (Majority Vote)
                            bool perspective_aligned = false;
                            
                            if (!is_bed_event) {
                                float alignment_ratio = 0.0f;
                                if (state.valid_angle_frames > 0) {
                                    alignment_ratio = (float)state.aligned_frames / state.valid_angle_frames;
                                }
                                
                                printf("[Case5-DEBUG] ID:%d Observation Persp Check: Aligned=%d/%d (Ratio=%.2f)\n", 
                                       curr.id, state.aligned_frames, state.valid_angle_frames, alignment_ratio);
                                
                                // If > 75% of valid frames are aligned, consider it a non-fall (lying down aligned)
                                if (state.valid_angle_frames > 10 && alignment_ratio > 0.75f) {
                                    perspective_aligned = true;
                                    printf("[Case5] ID:%d PERSPECTIVE ALIGNED (Majority): Ratio=%.2f > 0.75\n", curr.id, alignment_ratio);
                                    pImpl->LogTrace(curr.id, pImpl->frame_idx, "FILTER_PERSP", alignment_ratio, "Perspective Aligned > 0.75");
                                } else {
                                    printf("[Case5] ID:%d NOT ALIGNED (Majority): Ratio=%.2f <= 0.75\n", curr.id, alignment_ratio);
                                }
                            }
                            
                            // Fall detected only if NOT aligned with perspective line AND NOT Bed Event
                            if (is_bed_event) {
                                // Rejected
                            } else if (!perspective_aligned) {
                                potential_fall = true;
                                fall_type = "Momentum_Transition";
                                case_num = 5;
                                printf("[Case5] FALL DETECTED ID:%d obs_avg=%.2f (min=%.2f, max=%.2f) < %.2f (Adaptive: %s, Flat: %d/%d) (triggered at F:%d, observed %d frames)\n", 
                                       curr.id, obs_mom_avg, obs_min_mom, obs_max_mom, current_threshold2, (is_flat_posture?"YES":"NO"), state.flat_posture_frames, state.frames_observed, state.trigger_frame, state.frames_observed);
                                
                                pImpl->LogTrace(curr.id, pImpl->frame_idx, "DETECTED", obs_mom_avg, (is_flat_posture ? "Adaptive Threshold Passed" : "Normal Threshold Passed"));

                            } else {
                                printf("[Case5] ID:%d observation ENDED - PERSPECTIVE ALIGNED (obs_avg=%.2f, min=%.2f, max=%.2f) < %.2f but NOT a fall (Flat: %d/%d)\n", 
                                       curr.id, obs_mom_avg, obs_min_mom, obs_max_mom, current_threshold2, state.flat_posture_frames, state.frames_observed);
                            }
                        } else {
                            printf("[Case5] ID:%d observation ENDED - NO FALL (obs_avg=%.2f, min=%.2f, max=%.2f) >= %.2f (Adaptive: %s, Flat: %d/%d)\n", 
                                   curr.id, obs_mom_avg, obs_min_mom, obs_max_mom, current_threshold2, (is_flat_posture?"YES":"NO"), state.flat_posture_frames, state.frames_observed);
                            pImpl->LogTrace(curr.id, pImpl->frame_idx, "FILTER_MOMENTUM", obs_mom_avg, "Observation Avg > Threshold");
                        }
                        
                        // Reset state
                        state.is_active = false;
                        state.momentum_samples.clear();
                        state.frames_observed = 0;
                        state.aligned_frames = 0;
                        state.valid_angle_frames = 0;
                        state.flat_posture_frames = 0;
                        state.accumulated_bed_ratio = 0.0f;
                    }
                }
            }
            
            
            if (potential_fall) {
                // Common Checks: Bed Region & Leaving Scene & Static
                
                // 1-d / 2-c: Check Last Position in Bed?
                // User: "Check last possible position not in bed".
                // We use Current Center/Bottom.
                bool in_bed = false;
                if (pImpl->hasBedMask) {
                    // Check Bottom Point
                    int cx = (int)(curr.centerX * pImpl->config.block_size);
                    // Use max row for bottom
                    int max_r = 0;
                    for(int b : curr.blocks) { int r = b / pImpl->config.grid_cols; if(r>max_r) max_r=r; }
                    int cy = (max_r + 1) * pImpl->config.block_size;
                    
                    if (cx>=0 && cx<W && cy>=0 && cy<H) {
                         // Mask: 0 = Inside Bed
                         if (pImpl->bedMask.getData()[cy*W + cx] == 0) in_bed = true;
                    }
                }
                
                if (in_bed) {
                    // printf("[NewLogic] ID %d Rejected: In Bed Region\n", curr.id);
                    // continue; // DISABLED FOR DEBUG
                }
                
                // 1-f / 2-e: Leaving Scene Guard
                // "Evaluate if person is leaving frame"
                // Check if near border AND velocity points out
                bool near_border = (curr.centerX < 2.0f || curr.centerX > (pImpl->config.grid_cols - 2.0f) ||
                                    curr.centerY < 2.0f || curr.centerY > (pImpl->config.grid_rows - 2.0f));
                if (near_border) {
                     // Check Velocity Direction relative to border
                     // E.g. x < 2 and dx < 0 -> Leaving Left
                     bool leaving = false;
                     float vx = curr.avgDx;
                     float vy = curr.avgDy;
                     float cx = curr.centerX;
                     float cy = curr.centerY;
                     float w_g = pImpl->config.grid_cols;
                     float h_g = pImpl->config.grid_rows;
                     
                     if (cx < 2.0f && vx < -0.1f) leaving = true;
                     if (cx > w_g - 2.0f && vx > 0.1f) leaving = true;
                     if (cy < 2.0f && vy < -0.1f) leaving = true;
                     if (cy > h_g - 2.0f && vy > 0.1f) leaving = true; // Bottom border usually floor
                     
                     if (leaving) {
                         printf("[NewLogic] ID %d Rejected: Leaving Scene\n", curr.id);
                         continue;
                     }
                }
                
                // Trigger
                printf("[NewLogic] FALL DETECTED ID %d Type: %s. StrTrend: High->Low. FGTrend: High->Low.\n", curr.id, fall_type.c_str());
                detected_id = curr.id;
                triggered_objects.push_back(curr.id);
                warningMsg = fall_type;
            }
        }
    }

    

    
    if (slow_triggered > 0 && detected_id != -1) {
         bool exists = false;
         for(int tid : triggered_objects) if(tid == detected_id) exists = true;
         if(!exists) {
             // printf("[DEBUG] PUSH BACK SLOW for ID %d\n", detected_id);
             triggered_objects.push_back(detected_id);
         }
    }
    
    if (!triggered_objects.empty()) triggered = 1;

    if ((int)pImpl->object_history.size() > limit) {
        pImpl->object_history.erase(pImpl->object_history.begin());
    }
    // printf("[Detect] History Updated. Size=%zu\n", pImpl->object_history.size());

    
    // Debug print for bed exit
    // Iterate all objects to see who is exiting
    /*
    for(auto const& [oid, status] : pImpl->object_bed_exit_status) {
        if(status) printf("[Debug] Frame %lld Obj %d has Bed Exit Status.\n", pImpl->absolute_frame_count, oid);
    }
    */

    // Update Global Pixel History
    pImpl->global_pixel_history.push_back(global_fg_count);
    if(pImpl->global_pixel_history.size() > 100) pImpl->global_pixel_history.pop_front();

    // === NEW: FG Count History Buffer for Trend Verification ===
    pImpl->fg_count_history_buffer.push_back(global_fg_count);
    if (pImpl->fg_count_history_buffer.size() > 120) pImpl->fg_count_history_buffer.pop_front();

    // V17 Post-Trigger Logic Removed for V18 Experiment

    // Note: is_fall may have been set to true by pending_falls confirmation above
    // Only reset if we're also going to process new triggers
    
    // MERGE FIX: Add Slow Fall IDs to triggered list
    for(int sid : slow_fall_ids) {
        bool exists = false;
        for(int pid : triggered_objects) if(pid == sid) { exists = true; break; }
        if(!exists) triggered_objects.push_back(sid);
    }

    // V18 DIRECT CONFIRMATION (Bypass V17 Gates)
    if (!triggered_objects.empty()) {
        is_fall = true;
        // warning = "V18_Fall"; // Optional: pass reason
        for(int pid : triggered_objects) {
            printf("[V18] *** FALL CONFIRMED *** ID %d (Direct V18 Trigger)\n", pid);
        }
    }

#if 0 // V17 CANDIDATE VERIFICATION DISABLED
    // V17 Logic Removed for Compilation
#endif
    
    // Maintain Pending Falls Cleanup (empty loop if commented out above)
    if (!pImpl->candidates.empty()) 
    {
        std::vector<Impl::FallCandidate> kept_candidates;
        for(auto& c : pImpl->candidates) {
            bool found = false;
            float cx = 0, cy = 0;
            float vx = 0, vy = 0;
            for(const auto& o : pImpl->current_objects) {
                if (o.id == c.id) { 
                    cx = o.centerX; 
                    cy = o.centerY; 
                    vx = o.avgDx;
                    vy = o.avgDy;
                    found = true; 
                    break; 
                }
            }
            
            if (found) {
                 float dist = std::sqrt(pow(cx - c.startX, 2) + pow(cy - c.startY, 2));
                 
                 // Debug: Track Candidate Movement
                 float motion_speed = std::sqrt(pow(vx, 2) + pow(vy, 2));
                 
                 printf("DEBUG: Cand %d (Frame %d/%d) Start(%.1f,%.1f) Curr(%.1f,%.1f) Dist: %.2f (Thresh: %.1f) Speed: %.2f\n", 
                        c.id, c.frames_monitored + 1, pImpl->config.post_fall_check_frames, 
                        c.startX, c.startY, cx, cy, dist, pImpl->config.post_fall_distance_threshold, motion_speed);

                 // REJECTION LOGIC
                 bool rejected = false;
                 // 1. Distance from Start (Legacy)
                 if (dist > pImpl->config.post_fall_distance_threshold) {
                     printf("[FallDetector] Fall Candidate %d REJECTED (Dist: %.2f > %.2f)\n", c.id, dist, pImpl->config.post_fall_distance_threshold);
                     rejected = true;
                 }
                 // 2. Instant Motion Speed (New fix for "observing center movement")
                 // If object is still moving significantly (e.g. walking), reject.
                 // Threshold 2.5?
                 else if (motion_speed > 2.5f) {
                      printf("[FallDetector] Fall Candidate %d REJECTED (Speed: %.2f > 2.5)\n", c.id, motion_speed);
                      rejected = true;
                 }

                 if (rejected) continue; 

                 
                 c.frames_monitored++;
                 if (c.frames_monitored >= pImpl->config.post_fall_check_frames) 
                 {
                     // Confirmed
                     // is_fall = true; // DISABLED LEGACY CONFIRMATION
                     warning = "FALL DETECTED (Confirmed)! Obj " + std::to_string(c.id);
                     printf("[FallDetector] Fall Candidate %d CONFIRMED.\n", c.id);
                     kept_candidates.push_back(c); 
                 } else {
                     kept_candidates.push_back(c);
                 }
            } else {
                 printf("[FallDetector] Fall Candidate %d LOST.\n", c.id);
            }
        }
        pImpl->candidates = kept_candidates;
    }
    
    // Fallback: If immediate warning exists but no candidates yet (processing delay?), pass it?
    // No, logic is strictly: Trigger -> Monitor -> Confirm.
    // warning = warningMsg; // Don't use raw warningMsg anymore, only confirmed warning. 
    // Is this correct? Yes. But I should probably log warningMsg.
    if (!warningMsg.empty()) {
        // printf("[FallDetector] Instant Trigger: %s\n", warningMsg.c_str());
    }

    // Find max strength for callback, even if not a fall
    // Find max strength for callback, even if not a fall
    for(const auto& obj : pImpl->current_objects) {
        if(obj.strength > max_strength) max_strength = obj.strength;
    }
    
    // Temporal Consistency (Debounce)
    // static int fall_consecutive_frames = 0; // Removed static
    if (is_fall) {
        pImpl->fall_consecutive_frames++;
    } else {
        pImpl->fall_consecutive_frames = 0;
    }
    if (pImpl->fall_consecutive_frames >= pImpl->config.fall_duration) {
         // Keep is_fall = true
    } else {
         //is_fall = false;
         //printf("!!!!!!!!!!!!!!??????\n");
    }

    // 3. Visualization and Saving
    /*
    if (pImpl->config.enable_save_images) {
        ...
        saveBMP_RGB(out_name, rgbData.data(), W, H);
    }
    */

    if(is_fall) 
    {
        printf("[FallDetector] frame %d detect True Fall!!, Case %d, type=%s\n", pImpl->frame_idx, case_num, fall_type.c_str());
        
        // Print bounding box info for all detected objects
        for(const auto& obj : pImpl->current_objects) {
            int bbox_w, bbox_h, min_x, min_y, max_x, max_y;
            getObjectBoundingBoxPixels(obj, pImpl->config.grid_cols, pImpl->config.grid_rows,
                                       W, H, bbox_w, bbox_h, min_x, min_y, max_x, max_y);
            printf("[FallConfirm BBox] ID %d: %dx%d pixels (min=%d,%d max=%d,%d)\n",
                   obj.id, bbox_w, bbox_h, min_x, min_y, max_x, max_y);
        }
        
        std::cout << "[FallDetector] " << warning << std::endl;
    }

    // SAFETY OVERRIDE: Removed - FG Verification may confirm falls without new triggers
    // The is_fall flag is now managed by:
    // 1. Pending falls confirmation (FG trend analysis)
    // 2. New trigger processing
    // DO NOT reset is_fall to false here!

    // 7. Invoke Callback
    if (pImpl->callback) {
        VisionSDKEvent event;
        // event.timestamp = frame.timestamp; // NOT IN STRUCT
        event.frame_index = pImpl->frame_idx++; // Use and increment frame_idx
        event.is_fall_detected = is_fall;
        
        // Aggregate Bed Exit Status
        bool any_bed_exit = false;
        for(auto const& [oid, status] : pImpl->object_bed_exit_status) {
            if(status) { any_bed_exit = true; break; }
        }
        event.is_bed_exit = any_bed_exit;
        // event.is_weak_movement = false; // NOT IN STRUCT
        event.is_strong = (pImpl->fall_consecutive_frames >= pImpl->config.fall_strong_threshold); // Heuristic
        event.confidence = (float)pImpl->fall_consecutive_frames / 10.0f; // Simplified
        if(event.confidence > 1.0f) event.confidence = 1.0f;
        
        // Face Info
        event.is_face = has_face;
        event.face_x = face_roi.x1;
        event.face_y = face_roi.y1;
        event.face_w = face_roi.x2 - face_roi.x1;
        event.face_h = face_roi.y2 - face_roi.y1;
        // event.face_score = face_roi.score; // NOT IN STRUCT

        pImpl->callback(event);
    }
    
    #if ENABLE_PERF_PROFILING
    long long t_logic_end = pImpl->get_now_us();
    pImpl->prof.fall_logic_time += (t_logic_end - t_logic_start);
    long long t_total_end = pImpl->get_now_us();
    pImpl->prof.total_time += (t_total_end - t0);
    
    pImpl->prof.frame_count++;
    if (pImpl->prof.frame_count >= 100) {
        double avg_total = (double)pImpl->prof.total_time / pImpl->prof.frame_count / 1000.0;
        double avg_me = (double)pImpl->prof.motion_est_time / pImpl->prof.frame_count / 1000.0;
        double avg_face = (double)pImpl->prof.face_detect_time / pImpl->prof.frame_count / 1000.0;
        double avg_logic = (double)pImpl->prof.fall_logic_time / pImpl->prof.frame_count / 1000.0;
        
        printf("[FallDetector Profiling] Avg Time (ms) - Total: %.2f, ME: %.2f, Face: %.2f, Logic: %.2f\n", 
               avg_total, avg_me, avg_face, avg_logic);
        
        pImpl->prof.Reset();
    }
    #endif

    printf("[Debug] FallDetector::Detect End Frame %lld\n", pImpl->absolute_frame_count);
    return StatusCode::OK;
}

const std::vector<MotionObject>& FallDetector::GetMotionObjects() const {
    return pImpl->current_objects;
}

std::vector<::VisionSDK::ObjectFeatures> FallDetector::GetFullFrameObjects() const {
    return pImpl->full_frame_objects;
}

std::vector<std::pair<int, int>> FallDetector::GetBedRegion() const {
    return pImpl->bed_region;
}
void FallDetector::SetBackground(const Image& frame) {
    if (!pImpl) return;
    if (!frame.data) return;
    ::Image wrapper(frame.width, frame.height, frame.channels);
    wrapper.setData(frame.data, frame.width * frame.height * frame.channels);
    pImpl->backgroundFrame = wrapper;
    printf("[FallDetector] Background Explicitly Set to %dx%d image\n", frame.width, frame.height);
}


std::vector<uint8_t> FallDetector::GetChangedBlocks() const {
    std::vector<uint8_t> ret;
    if(!pImpl) return ret;
    ret.reserve(pImpl->changed_mask.size());
    for(bool b : pImpl->changed_mask) ret.push_back(b ? 1 : 0);
    return ret;
}

std::vector<MotionVector> FallDetector::GetMotionVectors() const {
    if(!pImpl) return {};
    return pImpl->motion_vectors;
}

} // VisionSDK
