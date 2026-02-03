
//(ARMv7, NEON):
// arm-linux-gnueabihf-g++ -O3 -std=gnu++11 -mfpu=neon -mfloat-abi=hard Bruce_C_v3_profiler.cpp -o motion_est
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <limits>
#include <iostream>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <cstdint>

//#define SAVE_BMP

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#include "Image.h" 
#include "Option.h"

using namespace std;
using std::vector;
using std::string;
using std::numeric_limits;

int diamond_count = 0;
int threeStep_count = 0;
int changed_blocks = 0;


// 
struct FunctionTimer {
    std::map<string, double> total_ms;
    std::map<string, uint64_t> calls;

    inline void add(const string& name, double ms) {
        total_ms[name] += ms;
        calls[name] += 1ULL;
    }

    void printAverage() const {
        std::cout << "\n=== Function Average Timings (ms) ===\n";
        for (const auto& kv : total_ms) {
            const string& name = kv.first;
            double sum = kv.second;
            uint64_t cnt = 0;
            auto it = calls.find(name);
            if (it != calls.end()) cnt = it->second;
            double avg = (cnt ? sum / (double)cnt : 0.0);
            printf("%-28s : %9.3f  (calls=%llu)\n",
                   name.c_str(), avg, (unsigned long long)cnt);
        }
        std::cout << "=====================================\n";
    }
};

// RAII 
struct TimerGuard {
    FunctionTimer* prof;
    string name;
    std::chrono::high_resolution_clock::time_point t0;
    TimerGuard(FunctionTimer& p, const string& n) : prof(&p), name(n), t0(std::chrono::high_resolution_clock::now()) {}
    ~TimerGuard() {
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
        prof->add(name, ms);
    }
};

// ------------------------ 
struct BlockPosition {
    int i = 0, j = 0;
    int x_start = 0, x_end = 0;
    int y_start = 0, y_end = 0;
};
struct MotionVector {
    int dx = 0, dy = 0;
    MotionVector() {}
    MotionVector(int _dx, int _dy) : dx(_dx), dy(_dy) {}
};

// ------------------------ Estimator ------------------------
class OptimizedBlockMotionEstimator {
public:
    OptimizedBlockMotionEstimator(int h_blocks, int v_blocks, int block_sz, int sr)
    : horizontal_blocks(h_blocks), vertical_blocks(v_blocks),
      block_size(block_sz), search_range(sr) {}

    
void blockBasedMotionEstimation(const Image& curr_frame_in,
                                std::vector<MotionVector>& motion_vectors,
                                std::vector<BlockPosition>& positions,            
                                std::vector<bool>& changed_blocks_mask,
                                std::vector<Image>& active_blocks,
                                std::vector<int>& active_indices)
{
    TimerGuard total_t(profiler, "blockBasedMotionEstimation_total");

    motion_vectors.clear();
    positions.clear();               //
    changed_blocks_mask.clear();
    active_blocks.clear();
    active_indices.clear();

    vector<MotionVector> motion_vectors_;

    const int total_blocks = horizontal_blocks * vertical_blocks;
    //printf("positions_cache.size() %d\n", positions_cache.size());
    if (prev_frame.empty()) 
    {
        buildPositionsOnce(curr_frame_in, positions_cache);
        if ((int)positions_cache.size() != total_blocks) 
        {
            positions_cache.clear();
            positions_cache.reserve(total_blocks);
            const int bw = curr_frame_in.width()  / horizontal_blocks;
            const int bh = curr_frame_in.height() / vertical_blocks;
            for (int by = 0; by < vertical_blocks; ++by) {
                for (int bx = 0; bx < horizontal_blocks; ++bx) {
                    BlockPosition p;
                    p.i = by;
                    p.j = bx;
                    p.x_start = bx * bw;  p.x_end = p.x_start + bw;
                    p.y_start = by * bh;  p.y_end = p.y_start + bh;
                    positions_cache.push_back(p);
                    motion_vectors_.push_back(MotionVector(0,0));
                }
            }
        }

        positions.assign(positions_cache.begin(), positions_cache.end());

        motion_vectors.assign(total_blocks, MotionVector(0, 0));
        changed_blocks_mask.assign(total_blocks, false);

        // 設定上一幀
        prev_frame = curr_frame_in.clone();   // 安全：不 swap 呼叫端的緩衝

        return;
    }

    const Image& curr_frame = curr_frame_in;


    

    // 
    if ((int)positions_cache.size() != total_blocks) 
    {
        buildPositionsOnce(curr_frame, positions_cache);
        if ((int)positions_cache.size() != total_blocks) 
        {
            positions_cache.clear();
            const int bw = curr_frame.width()  / horizontal_blocks;
            const int bh = curr_frame.height() / vertical_blocks;
            for (int by = 0; by < vertical_blocks; ++by) 
            {
                for (int bx = 0; bx < horizontal_blocks; ++bx) 
                {

                    BlockPosition p;
                    p.i = by;
                    p.j = bx;
                    p.x_start = bx * bw;  p.x_end = p.x_start + bw;
                    p.y_start = by * bh;  p.y_end = p.y_start + bh;
                    positions_cache.push_back(p);
                    motion_vectors_.push_back(MotionVector(0,0));
                }
            }
        }
    }

    // 
    positions.assign(positions_cache.begin(), positions_cache.end());
    //for(size_t iii = 0;iii<positions.size();iii++)
    {
    //    printf("test %d %d\n", positions[iii].i, positions[iii].j);
    }
    // --- 偵測變動區塊---
    auto t_diff0 = std::chrono::high_resolution_clock::now();
    changed_blocks_mask.assign(total_blocks, false); // 先備妥大小
    int active_blocks_count = detectChangedBlocks(prev_frame, curr_frame, changed_blocks_mask, 0.05);
    changed_blocks += active_blocks_count;
    auto t_diff1 = std::chrono::high_resolution_clock::now();
    profiler.add("detectChangedBlocks_fullframe",
        std::chrono::duration_cast<std::chrono::microseconds>(t_diff1 - t_diff0).count() / 1000.0);

    if ((int)changed_blocks_mask.size() != total_blocks) {
        changed_blocks_mask.resize(total_blocks, false);
    }

    // --- 只對變動區塊建立 ROI ---
    auto t_roi0 = std::chrono::high_resolution_clock::now();
    divideActiveBlocks(curr_frame, positions_cache, changed_blocks_mask,
                       active_blocks, active_indices);
    auto t_roi1 = std::chrono::high_resolution_clock::now();
    profiler.add("divideActiveBlocks",
        std::chrono::duration_cast<std::chrono::microseconds>(t_roi1 - t_roi0).count() / 1000.0);

    //
    auto t_motion0 = std::chrono::high_resolution_clock::now();
    motion_vectors.assign(total_blocks, MotionVector(0, 0));  

    for (size_t k = 0; k < active_indices.size(); ++k) {
        int idx = active_indices[k];
        // 
        if (idx < 0 || idx >= total_blocks) continue;
        if ((int)positions_cache.size() <= idx) continue;
        if ((int)active_blocks.size()  <= (int)k) continue;

        const BlockPosition& pos = positions_cache[idx];
        const Image& block       = active_blocks[k];

        const int bh = block.height();
        const int bw = block.width();
        int dx = 0, dy = 0, sad = 0;

        optimizedMotionEstimation(block, prev_frame,
                                  pos.x_start, pos.y_start,
                                  bh, bw, dx, dy, sad, idx);

        motion_vectors[idx] = MotionVector(dx, dy);
    }

    updateMotionHistoryFast(motion_vectors);


    auto t_motion1 = std::chrono::high_resolution_clock::now();
    profiler.add("motionEstimation_activeBlocks",
        std::chrono::duration_cast<std::chrono::microseconds>(t_motion1 - t_motion0).count() / 1000.0);

    // --- [D] 更新上一幀 ---
    prev_frame = curr_frame.clone();

}



    // 結束時呼叫：印出各函式平均耗時
    void printAverageTimings() const {
        profiler.printAverage();
    }

private:
    // 參數
    int horizontal_blocks = 8;
    int vertical_blocks   = 6;
    int block_size        = 16;
    int search_range      = 6;

    // 內部狀態
    Image prev_frame;
    vector<BlockPosition> positions_cache;

    vector<vector<MotionVector>> motion_history;


    // Profiler
    mutable FunctionTimer profiler;
    // 
    void buildPositionsOnce(const Image& img, vector<BlockPosition>& out) {
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
                BlockPosition p; p.x_start = xs; p.x_end = xe; p.y_start = ys; p.y_end = ye;p.i=by;p.j=bx;
                out.push_back(p);
            }
        }
    }
    void predictMotionVectorFast(int block_idx, int& pdx, int& pdy) 
    {
        pdx = 0;
        pdy = 0;

        // 若沒有歷史記錄 → 回傳 0
        if (motion_history.empty()) 
            return;

        // 取上一幀的 motion vector
        const auto& last_vectors = motion_history.back();

        // block index 保護
        if (block_idx < (int)last_vectors.size()) {
            pdx = last_vectors[block_idx].dx;
            pdy = last_vectors[block_idx].dy;
        }
    }
    void updateMotionHistoryFast(const vector<MotionVector>& motion_vectors) {
        motion_history.push_back(motion_vectors);

        // �u�O�d�̪�5�V
        if (motion_history.size() > 5) {
            motion_history.erase(motion_history.begin());
        }
    }
    // 
    void divideIntoBlocks(const Image& image,
                          vector<Image>& blocks,
                          const vector<BlockPosition>& positions)
    {
        blocks.clear();
        blocks.reserve(positions.size());
        for (const auto& pos : positions) {
            Image roi = image.getROI(pos.x_start, pos.x_end, pos.y_start, pos.y_end);
            blocks.push_back(roi);
        }
    }
    // 
    void divideActiveBlocks(const Image& image,
                            const vector<BlockPosition>& positions,
                            const vector<bool>& changed_mask,
                            vector<Image>& active_blocks,
                            vector<int>& active_indices)
    {
        TimerGuard _tg(profiler, "divideActiveBlocks");

        active_blocks.clear();
        active_indices.clear();

        int total = positions.size();
        active_blocks.reserve(total);

        for (int i = 0; i < total; ++i) {
            if (!changed_mask[i]) continue;

            const BlockPosition& pos = positions[i];
            Image roi = image.getROI(pos.x_start, pos.x_end, pos.y_start, pos.y_end);
            active_blocks.push_back(roi);
            active_indices.push_back(i);
        }
    }

    // NEON
    static inline int sad_u8_row(const uint8_t* a, const uint8_t* b, int len, FunctionTimer* prof=nullptr) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        // 單行 NEON 版本
        int sad = 0;
        int n = len;
        while (n >= 16) {
            uint8x16_t va = vld1q_u8(a);
            uint8x16_t vb = vld1q_u8(b);
            uint8x16_t vdiff = vabdq_u8(va, vb); // |a-b|
            // 8->4->2->1 逐步 pairwise 加總
            uint16x8_t vpad1 = vpaddlq_u8(vdiff);    // 16x8
            uint32x4_t vpad2 = vpaddlq_u16(vpad1);   // 32x4
            uint64x2_t vpad3 = vpaddlq_u32(vpad2);   // 64x2
            sad += (int)vgetq_lane_u64(vpad3, 0);
            sad += (int)vgetq_lane_u64(vpad3, 1);
            a += 16; b += 16; n -= 16;
        }
        while (n--) sad += std::abs((int)(*a++) - (int)(*b++));
        if (prof) prof->add("computeSAD", 0.0); // 計次（時間由 computeSAD 外層統一計）
        return sad;
#else
        //
        int sad = 0;
        for (int i=0;i<len;++i) sad += std::abs((int)a[i] - (int)b[i]);
        if (prof) prof->add("computeSAD", 0.0);
        return sad;
#endif
    }

    // 區塊 SAD（block_curr vs prev_frame 上 (ref_x,ref_y)）
    int computeSAD(const Image& block_curr, const Image& prev_frame,
                   int ref_x, int ref_y)
    {
        TimerGuard _tg(profiler, "computeSAD");
        const int bw = block_curr.width();
        const int bh = block_curr.height();

        const int stride_prev = prev_frame.width();
        const uint8_t* pblock = block_curr.getData();
        const uint8_t* pprev  = prev_frame.getData() + ref_y * stride_prev + ref_x;

        int sad = 0;
        for (int y=0; y<bh; ++y) {
            const uint8_t* a = pblock + y * bw;
            const uint8_t* b = pprev  + y * stride_prev;
            sad += sad_u8_row(a, b, bw, &profiler);
        }
        return sad;
    }

    // 二值化
    Image binarizeDifference(const Image& prev, const Image& curr, int threshold = 30) {
        TimerGuard _tg(profiler, "binarizeDifference");

        const int w = prev.width();
        const int h = prev.height();
        Image out(w, h, 1);

        const uint8_t* pprev = prev.getData();
        const uint8_t* pcurr = curr.getData();
        uint8_t* pout       = out.getData();

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        const uint8x16_t vthr = vdupq_n_u8((uint8_t)threshold);
        const uint8x16_t v255 = vdupq_n_u8(255);
#endif
        for (int y=0; y<h; ++y) {
            const uint8_t* a = pprev + y*w;
            const uint8_t* b = pcurr + y*w;
            uint8_t*       o = pout  + y*w;

            int x=0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
            for (; x<=w-16; x+=16) {
                uint8x16_t va = vld1q_u8(a + x);
                uint8x16_t vb = vld1q_u8(b + x);
                uint8x16_t vdiff = vabdq_u8(va, vb);
                uint8x16_t vmask = vcgtq_u8(vdiff, vthr);
                uint8x16_t vres  = vandq_u8(vmask, v255);
                vst1q_u8(o + x, vres);
            }
#endif
            for (; x<w; ++x) {
                int d = (int)b[x] - (int)a[x];
                if (d<0) d = -d;
                o[x] = (d>threshold) ? 255 : 0;
            }
        }
        return out;
    }

    // 變動區塊檢測（回傳變動區塊數），同時輸出 mask
    int detectChangedBlocks(const Image& prev, const Image& curr,
                            vector<bool>& changed_blocks_mask,
                            double threshold_ratio)
    {
        TimerGuard _tg(profiler, "detectChangedBlocks");

        Image bin = binarizeDifference(prev, curr, 30);
        const uint8_t* diff = bin.getData();
        const int w = bin.width();
        const int h = bin.height();

        const int total_blocks = horizontal_blocks * vertical_blocks;
        changed_blocks_mask.clear();
        changed_blocks_mask.reserve(total_blocks);

        int bh = h / vertical_blocks;
        int bw = w / horizontal_blocks;

        int changed = 0;
        for (int by=0; by<vertical_blocks; ++by) {
            for (int bx=0; bx<horizontal_blocks; ++bx) {
                int xs = bx*bw;
                int ys = by*bh;
                int xe = (bx==horizontal_blocks-1)? w : xs + bw;
                int ye = (by==vertical_blocks-1)? h : ys + bh;

                int count_white = 0;
                for (int y=ys; y<ye; ++y) {
                    const uint8_t* row = diff + y*w + xs;
                    for (int x=0; x< (xe - xs); ++x) {
                        count_white += (row[x] ? 1 : 0);
                    }
                }
                const int pixels = (xe - xs) * (ye - ys);
                const double ratio = (pixels>0) ? (double)count_white / (double)pixels : 0.0;
                bool flag = (ratio > threshold_ratio);
                changed_blocks_mask.push_back(flag);
                if (flag) ++changed;
            }
        }
        return changed;
    }

    // 
    int calculateTextureComplexityFast(const Image& block) {
        // 以鄰近差分估計（快速）：sum |I(x)-I(x-1)| + |I(y)-I(y-1)|
        const int w = block.width(), h = block.height();
        const uint8_t* p = block.getData();
        int score = 0;
        for (int y=0; y<h; ++y) {
            const uint8_t* row = p + y*w;
            for (int x=1; x<w; ++x) score += std::abs((int)row[x] - (int)row[x-1]);
        }
        for (int y=1; y<h; ++y) {
            const uint8_t* row = p + y*w;
            const uint8_t* prv = p + (y-1)*w;
            for (int x=0; x<w; ++x) score += std::abs((int)row[x] - (int)prv[x]);
        }
        return score / 2;
    }

    // 
    //void predictMotionVectorFast(int /*block_id*/, int& pdx, int& pdy) {
    //    pdx = 0; pdy = 0;
    //}

    // 
    void fastDiamondSearch(const Image& block_curr, const Image& prev,
                           int x_start, int y_start, int bh, int bw,
                           int init_dx, int init_dy,
                           int& best_dx, int& best_dy, int& best_sad)
    {
        diamond_count++;
        TimerGuard _tg(profiler, "fastDiamondSearch");

        best_dx = init_dx;
        best_dy = init_dy;

        // 初值 SAD
        auto safe_in = [&](int rx, int ry) {
            return (rx>=0 && ry>=0 && rx+bw<=prev.width() && ry+bh<=prev.height());
        };
        best_sad = numeric_limits<int>::max();
        if (safe_in(x_start+best_dx, y_start+best_dy)) {
            best_sad = computeSAD(block_curr, prev, x_start+best_dx, y_start+best_dy);
        }

        static const int small_diamond[5][2] = { {0,0},{-1,0},{1,0},{0,-1},{0,1} };
        bool improved = true;
        int iter = 0, max_iter = 3;

        while (improved && iter < max_iter) {
            improved = false;
            int cur_dx = best_dx, cur_dy = best_dy;

            for (int k=0; k<5; ++k) {
                int cdx = cur_dx + small_diamond[k][0];
                int cdy = cur_dy + small_diamond[k][1];
                //printf("search_range %d\n", search_range);
                if (std::abs(cdx) > search_range || std::abs(cdy) > search_range) continue;

                int rx = x_start + cdx;
                int ry = y_start + cdy;
                if (!safe_in(rx, ry)) continue;

                int sad = computeSAD(block_curr, prev, rx, ry);
                if (sad < best_sad) {
                    best_sad = sad; best_dx = cdx; best_dy = cdy; improved = true;
                }
            }
            ++iter;
        }
    }

    // 三步搜尋
    void threeStepSearch(const Image& block_curr, const Image& prev,
                         int x_start, int y_start, int bh, int bw,
                         int& best_dx, int& best_dy, int& best_sad)
    {
        threeStep_count++;
        TimerGuard _tg(profiler, "threeStepSearch");

        best_dx = 0; best_dy = 0;
        best_sad = numeric_limits<int>::max();

        auto safe_in = [&](int rx, int ry) {
            return (rx>=0 && ry>=0 && rx+bw<=prev.width() && ry+bh<=prev.height());
        };
        const int steps[3] = {4, 2, 1};

        for (int si=0; si<3; ++si) {
            int step = steps[si];
            int cur_dx = best_dx, cur_dy = best_dy;

            for (int dy=-step; dy<=step; dy+=step) {
                for (int dx=-step; dx<=step; dx+=step) {
                    int cdx = cur_dx + dx;
                    int cdy = cur_dy + dy;

                    if (std::abs(cdx) > search_range || std::abs(cdy) > search_range) continue;

                    int rx = x_start + cdx;
                    int ry = y_start + cdy;
                    if (!safe_in(rx, ry)) continue;

                    int sad = computeSAD(block_curr, prev, rx, ry);
                    if (sad < best_sad) { best_sad = sad; best_dx = cdx; best_dy = cdy; }
                }
            }
        }
    }

    void optimizedMotionEstimation(const Image& block_curr, const Image& prev,
                                   int x_start, int y_start, int bh, int bw,
                                   int& best_dx, int& best_dy, int& best_sad, int idx)
    {
        TimerGuard _tg(profiler, "optimizedMotionEstimation");

        int pdx=0, pdy=0;
        predictMotionVectorFast(idx, pdx, pdy);

        //int complexity = calculateTextureComplexityFast(block_curr);
        //if (complexity > 200) 
        {
            fastDiamondSearch(block_curr, prev, x_start, y_start, bh, bw, pdx, pdy,
                              best_dx, best_dy, best_sad);
        } 
       // else {
       //     threeStepSearch(block_curr, prev, x_start, y_start, bh, bw,
        //                    best_dx, best_dy, best_sad);
        //}
    }
};



bool saveBMP(const std::string& filename, const uint8_t* gray, int width, int height)
{
    FILE* f = fopen(filename.c_str(), "wb");
    if (!f) return false;

    int filesize = 54 + 3 * width * height;
    uint8_t header[54] = {
        0x42, 0x4D,              // 'BM'
        0,0,0,0,                 // size
        0,0, 0,0,
        54,0,0,0,                // offset
        40,0,0,0,                // header size
        0,0,0,0,                 // width
        0,0,0,0,                 // height
        1,0, 24,0,               // planes, bpp
        0,0,0,0,                 // compression
        0,0,0,0,                 // image size
        0,0,0,0, 0,0,0,0,        // x/y ppm
        0,0,0,0, 0,0,0,0
    };

    int w = width, h = height;
    header[ 2] = (uint8_t)(filesize    );
    header[ 3] = (uint8_t)(filesize>> 8);
    header[ 4] = (uint8_t)(filesize>>16);
    header[ 5] = (uint8_t)(filesize>>24);
    header[18] = (uint8_t)(w    );
    header[19] = (uint8_t)(w>> 8);
    header[20] = (uint8_t)(w>>16);
    header[21] = (uint8_t)(w>>24);
    header[22] = (uint8_t)(h    );
    header[23] = (uint8_t)(h>> 8);
    header[24] = (uint8_t)(h>>16);
    header[25] = (uint8_t)(h>>24);

    fwrite(header, 1, 54, f);

    // BMP 的行資料必須是 4-byte 對齊
    int padSize = (4 - (w*3) % 4) % 4;
    uint8_t pad[3] = {0,0,0};

    // BMP 由下到上寫入
    for (int y = h - 1; y >= 0; --y) {
        for (int x = 0; x < w; ++x) {
            uint8_t g = gray[y * w + x];
            uint8_t rgb[3] = {g, g, g};
            fwrite(rgb, 1, 3, f);
        }
        fwrite(pad, 1, padSize, f);
    }

    fclose(f);
    return true;
}
void drawRect(uint8_t* img, int w, int h,
              int x1, int y1, int x2, int y2, uint8_t val)
{
    for (int x=x1; x<x2; ++x) {
        if (y1>=0 && y1<h) img[y1*w + x] = val;
        if (y2-1>=0 && y2-1<h) img[(y2-1)*w + x] = val;
    }
    for (int y=y1; y<y2; ++y) {
        if (x1>=0 && x1<w) img[y*w + x1] = val;
        if (x2-1>=0 && x2-1<w) img[y*w + x2-1] = val;
    }
}

void drawArrow(uint8_t* img, int w, int h,
               int x, int y, int dx, int dy, uint8_t val)
{
    int x2 = x + dx;
    int y2 = y + dy;
    int steps = std::max(abs(dx), abs(dy));
    if (steps == 0) return;  // 

    for (int i = 0; i <= steps; ++i) {
        int xx = x + i * dx / steps;
        int yy = y + i * dy / steps;
        if (xx >= 0 && xx < w && yy >= 0 && yy < h)
            img[yy * w + xx] = val;
    }
}
void drawArrow2(uint8_t* img, int w, int h,
               int x, int y, int dx, int dy, uint8_t val)
{


    int x2 = x + dx; // int x2 = x + dx;
    int y2 = y + dy;//int y2 = y + dy;
    int steps = std::max(abs(dx), abs(dy));
    if (steps == 0) return;  // 

    // 
    for (int i = 0; i <= steps; ++i) {
        int xx = x + i * dx / steps;
        int yy = y + i * dy / steps;
        if (xx >= 0 && xx < w && yy >= 0 && yy < h)
            img[yy * w + xx] = val;
    }

    // 箭頭長度太短就不畫頭
    if (steps < 3) return;

    // 箭頭頭部長度 (約線長的 20%)
    float head_len = std::max(3.0f, steps * 0.2f);
    // 箭頭角度 (30°)
    float angle = 30.0f * M_PI / 180.0f;

    float len = std::sqrt((float)(dx * dx + dy * dy));
    if (len < 1.0f) return;
    float ux = dx / len;
    float uy = dy / len;

    // 算箭頭兩側方向
    float sin_a = std::sin(angle);
    float cos_a = std::cos(angle);

    // 左右兩邊的旋轉向量
    float lx = cos_a * ux - sin_a * uy;
    float ly = sin_a * ux + cos_a * uy;
    float rx = cos_a * ux + sin_a * uy;
    float ry = -sin_a * ux + cos_a * uy;

    // 算出左右兩端點
    int left_x = (int)(x2 - head_len * lx);
    int left_y = (int)(y2 - head_len * ly);
    int right_x = (int)(x2 - head_len * rx);
    int right_y = (int)(y2 - head_len * ry);

    // 畫箭頭左、右支線
    auto drawLine = [&](int x1, int y1, int x2, int y2) {
        int ddx = x2 - x1, ddy = y2 - y1;
        int dsteps = std::max(abs(ddx), abs(ddy));
        if (dsteps == 0) return;
        for (int i = 0; i <= dsteps; ++i) {
            int xx = x1 + i * ddx / dsteps;
            int yy = y1 + i * ddy / dsteps;
            if (xx >= 0 && xx < w && yy >= 0 && yy < h)
                img[yy * w + xx] = val;
        }
    };

    drawLine(x2, y2, left_x, left_y);
    drawLine(x2, y2, right_x, right_y);
}
// int main() {
//     // �ڭ̫�Ӥw�g�� real frame source
//     // �u���@�Ӥ������� Image �Ӫ��A�Q�׬O�۳y��
//     int width = 640;
//     int height = 480;
//     Image frame(width, height, 1);

//     OptimizedBlockMotionEstimator estimator(32, 8, 8, 6);

//     auto last_time = chrono::high_resolution_clock::now();
//     float total_time = 0.f;

//     for (int i = 0; i < 100; ++i) {
//         // �s�դ@�� frame ���ơ]�ھڹϧγ̫��ε{�ҦC���F
//         for (int y = 0; y < height; ++y) {
//             for (int x = 0; x < width; ++x) {
//                 frame.at(x, y) = (uint8_t)((x + y + i) & 0xFF);
//             }
//         }

//         Image processed_frame = estimator.preprocessFrame(frame);

//         vector<MotionVector> motion_vectors;
//         vector<BlockPosition> positions;
//         vector<bool> changed_blocks_mask;

//         estimator.blockBasedMotionEstimation(processed_frame,
//             motion_vectors,
//             positions,
//             changed_blocks_mask);

//         auto current_time = chrono::high_resolution_clock::now();
//         auto duration = chrono::duration_cast<chrono::milliseconds>(
//             current_time - last_time).count();
//         float fps = duration > 0 ? 1000.0f / duration : 0.0f;
//         last_time = current_time;

//         total_time += (float)(duration);
//         float fps_m = (float)(i) / (total_time / 1000.f);
//         printf("fps %f\n", fps_m);
//     }

//     return 0;
// }


int main(int argc, char** argv) {
    const int width = 800;
    const int height = 600;
    int num_images = 370; // folder150455=370 images
    double folder_loop_times = 1;
    int folder_loop_times_ = 1;

    const size_t frame_size = width * height;

    Image frame(width, height, 1);

    std::vector<unsigned char> frame1(frame_size);
    std::vector<unsigned char> frame2(frame_size);


#if defined(__ARM_NEON) || defined(__ARM_NEON__)

printf("USE NEON\n");
#endif



    // std::ifstream file1("frame1_800x600.raw", std::ios::binary);
    // if (!file1) {
    //     std::cerr << "無法打開 frame1_800x600.raw" << std::endl;
    //     return 1;
    // }
    // file1.read(reinterpret_cast<char*>(frame1.data()), frame_size);
    // file1.close();

    // std::ifstream file2("frame2_800x600.raw", std::ios::binary);
    // if (!file2) {
    //     std::cerr << "無法打開 frame2_800x600.raw" << std::endl;
    //     return 1;
    // }
    // file2.read(reinterpret_cast<char*>(frame2.data()), frame_size);
    // file2.close();

    OptimizedBlockMotionEstimator estimator(12, 6, 16, 6);

    auto last_time = chrono::high_resolution_clock::now();
    int i = 0;
    int total_frames = 0;
    struct timespec start_TYCV, end_TYCV;
    double timecost_TYCV  = 0;

    vector<double> total_ms(num_images, 0.0);
    vector<int> total_block(num_images, 0.0);
    

    ofstream ofs_all("150455_800x600_AlgoC_FocalTrans_all_mov.txt");
    if (!ofs_all.is_open()) {
        cerr << "cannot open all_mov.txt for write\n";
        return 1;
    }

    while (folder_loop_times_ > 0) 
    {
        i = 0;
        while (i < num_images) 
        {

            const int frame_idx = i;  // <--- 固定這一圈的幀索引

            char raw_name[128];
            sprintf(raw_name, "images_150455_800x600/frame_%05d.raw", frame_idx);
            std::ifstream file1(raw_name, std::ios::binary);
            if (!file1) {
                std::cerr << "無法打開 " << raw_name << std::endl;
                return 1;
            }
            file1.read(reinterpret_cast<char*>(frame1.data()), frame_size);
            file1.close();

            clock_gettime(CLOCK_MONOTONIC, &start_TYCV);
            frame.setData(frame1.data(), frame_size);

            // 
            vector<MotionVector> motion_vectors;
            vector<BlockPosition> positions;
            vector<bool> changed_blocks_mask;
            vector<Image> active_blocks;
            vector<int> active_indices;

            estimator.blockBasedMotionEstimation(frame,
                                                motion_vectors,
                                                positions,
                                                changed_blocks_mask,
                                                active_blocks,
                                                active_indices);

            for (size_t jj = 0;jj<positions.size();++jj)
            {
                if(!changed_blocks_mask[jj]) continue;
                //int idx = active_indices[jj];

                BlockPosition pos_ = positions[jj];
                MotionVector mov_ = motion_vectors[jj];

                ofs_all<<i<<" "<<pos_.i <<" "<<pos_.j<<" "<<mov_.dx<<" "<<mov_.dy<<endl;



            }





            auto current_time = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(
                                current_time - last_time).count();
            last_time = current_time;
            float fps = duration > 0 ? 1000.0f / duration : 0.0f;

            int block_changed = std::count(changed_blocks_mask.begin(),
                                        changed_blocks_mask.end(), true);

            cout << "frame " << frame_idx
                << " changed = " << block_changed
                << " fps = " << fps << endl;

            clock_gettime(CLOCK_MONOTONIC, &end_TYCV);
            double my_sec = (end_TYCV.tv_sec - start_TYCV.tv_sec)
                        + (end_TYCV.tv_nsec - start_TYCV.tv_nsec) / 1e9;

            timecost_TYCV += my_sec;
            total_frames++;

            //
            if (frame_idx >= 0 && frame_idx < num_images) {
                total_ms[frame_idx]    += my_sec;
                total_block[frame_idx] += block_changed;
            }

            printf("Bruce C opt [%d] FPS %.9f s \n\n",
                total_frames, float(total_frames) / timecost_TYCV);

#ifdef SAVE_BMP

            uint8_t* img = frame.getData();

            // ✅ 2) 三個容器對齊長度
            const size_t N = std::min(positions.size(),
                            std::min(motion_vectors.size(), changed_blocks_mask.size()));

            for (int idx : active_indices) 
            {
                if (idx < 0 || idx >= changed_blocks_mask.size()) continue;
                if (!changed_blocks_mask[idx]) continue;

                const auto& pos = positions[idx];
                const auto& mv = motion_vectors[idx];

                drawRect(img, width, height,
                        pos.x_start, pos.y_start, pos.x_end, pos.y_end, 255);

                int cx = (pos.x_start + pos.x_end) / 2;
                int cy = (pos.y_start + pos.y_end) / 2;
                drawArrow2(img, width, height, cx, cy, -mv.dx * 5, -mv.dy * 5, 255);

            }

            char fname[128];
            sprintf(fname, "output_C_150455_320x180/out_%05d.bmp", frame_idx);
            saveBMP(fname, img, width, height);
#endif

            //
            ++i;
        }
        folder_loop_times_--;
    }



    estimator.printAverageTimings();
     //寫出平均值
    ofstream ofs("150455_timings_AlgoC_new.txt");
    if (!ofs.is_open()) {
        cerr << "cannot open timings.txt for write\n";
        return 1;
    }
    // 格式：index avg_sec
    for (int i = 0; i < num_images; ++i) {
        double avg_ms = total_ms[i] / folder_loop_times;
        ofs << i << " " << avg_ms << "\n";
    }

    ofstream ofs2("150455_changed_AlgoC_new.txt");
    if (!ofs2.is_open()) {
        cerr << "cannot open changed.txt for write\n";
        return 1;
    }
    // 格式：index avg_block num
    for (int i = 0; i < num_images; ++i) {
        int avg_c = total_block[i] / folder_loop_times;
        ofs2 << i << " " << avg_c << "\n";
    }
    
    printf("diamond_count %d\n", diamond_count);
    printf("threeStep_count %d\n", threeStep_count);
    printf("changed_blocks %d\n", changed_blocks);
    return 0;
}