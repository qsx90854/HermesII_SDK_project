#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <algorithm>
#include <numeric>
#include <limits>
#include "DefineSetting.h"
#include "Image_multi_type.h"

#include <iostream>
#include <fstream>
#include <utility> // 包含 std::pair
#include "ConfigLoader.h"

using namespace std;
using namespace cv;
using PeriodPair = std::pair<int, int>;

//紀錄床的範圍
Image bedRegion;
std::vector<Point2f> trapPoints;
//紀錄跌倒的時間段
std::vector<PeriodPair> periods;

int wait_time = 0;


struct MotionObject {
    int id;                    // unique id within frame
    std::vector<int> blocks;   // 所有屬於此物件的 block index
    float avgDx;               // 平均 dx
    float avgDy;               // 平均 dy
    float strength;            // 物件移動強度 = avg(|dx|+|dy|)
    float centerX, centerY;    // 物件中心位置（重要！）
};

struct BlockPosition {
    int i, j;           // 區塊索引
    int y_start, y_end;
    int x_start, x_end;
};

struct MotionVector {
    int dx, dy;
    MotionVector() : dx(0), dy(0) {}
    MotionVector(int x, int y) : dx(x), dy(y) {}
};
//std::vector<MotionObject> extractMotionObjects(
//    const std::vector<MotionVector>& blocks,
//    int rows, int cols, float threshold)
//{
//    std::vector<MotionObject> objs;
//    std::vector<char> visited(rows * cols, 0);
//
//    auto idx = [&](int r, int c) { return r * cols + c; };
//
//    const int dr[8] = { -1,-1,-1,0,0, 1,1,1 };
//    const int dc[8] = { -1, 0, 1,-1,1,-1,0,1 };
//
//
//
//
//    int objId = 0;
//
//    for (int r = 0; r < rows; r++) {
//        for (int c = 0; c < cols; c++) {
//            int start = idx(r, c);
//            float mag = std::sqrt(blocks[start].dx * blocks[start].dx +
//                blocks[start].dy * blocks[start].dy);
//            if (mag < threshold) continue;
//            if (visited[start]) continue;
//
//            MotionObject obj;
//            obj.id = objId++;
//
//            std::vector<int> stack = { start };
//            visited[start] = 1;
//
//            float sumDx = 0, sumDy = 0;
//            int count = 0;
//            int sumR = 0, sumC = 0;
//
//            while (!stack.empty()) {
//                int cur = stack.back(); stack.pop_back();
//
//                int cr = cur / cols;
//                int cc = cur % cols;
//
//                obj.blocks.push_back(cur);
//
//                sumDx += blocks[cur].dx;
//                sumDy += blocks[cur].dy;
//                sumR += cr;
//                sumC += cc;
//                count++;
//
//                for (int i = 0; i < 8; i++) {
//                    int nr = cr + dr[i], nc = cc + dc[i];
//                    if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
//                    int ni = idx(nr, nc);
//                    if (visited[ni]) continue;
//
//                    float mag2 = std::sqrt(blocks[ni].dx * blocks[ni].dx +
//                        blocks[ni].dy * blocks[ni].dy);
//                    if (mag2 < threshold) continue;
//
//                    visited[ni] = 1;
//                    stack.push_back(ni);
//                }
//            }
//
//            obj.avgDx = sumDx / count;
//            obj.avgDy = sumDy / count;
//            obj.centerX = sumC / (float)count;
//            obj.centerY = sumR / (float)count;
//            obj.strength = std::sqrt(obj.avgDx * obj.avgDx + obj.avgDy * obj.avgDy);
//            //printf("obj.strength : %f\n", obj.strength);
//            objs.push_back(obj);
//        }
//    }
//    return objs;
//}
std::vector<MotionObject> extractMotionObjects(
    const std::vector<MotionVector>& blocks,
    int rows, int cols,
    float threshold,
    int searchRadius = 3      // ⭐ 新增參數：搜尋半徑
)
{
    std::vector<MotionObject> objs;
    std::vector<char> visited(rows * cols, 0);

    auto idx = [&](int r, int c) { return r * cols + c; };

    int objId = 0;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {

            int start = idx(r, c);

            float mag = std::sqrt(blocks[start].dx * blocks[start].dx +
                blocks[start].dy * blocks[start].dy);

            if (mag < threshold) continue;
            if (visited[start]) continue;

            MotionObject obj;
            obj.id = objId++;

            std::vector<int> stack = { start };
            visited[start] = 1;

            float sumDx = 0, sumDy = 0;
            int count = 0;
            int sumR = 0, sumC = 0;

            // ★★★★★ Flood-fill（帶搜尋半徑）
            while (!stack.empty()) {
                int cur = stack.back();
                stack.pop_back();

                int cr = cur / cols;
                int cc = cur % cols;

                obj.blocks.push_back(cur);

                sumDx += blocks[cur].dx;
                sumDy += blocks[cur].dy;
                sumR += cr;
                sumC += cc;
                count++;

                // ===== ★ 使用 searchRadius 擴展搜尋範圍 =====
                for (int dr = -searchRadius; dr <= searchRadius; dr++) {
                    for (int dc = -searchRadius; dc <= searchRadius; dc++) {

                        if (dr == 0 && dc == 0) continue;

                        int nr = cr + dr;
                        int nc = cc + dc;

                        if (nr < 0 || nr >= rows || nc < 0 || nc >= cols)
                            continue;

                        int ni = idx(nr, nc);
                        if (visited[ni]) continue;

                        float mag2 = std::sqrt(blocks[ni].dx * blocks[ni].dx +
                            blocks[ni].dy * blocks[ni].dy);

                        if (mag2 < threshold) continue;

                        visited[ni] = 1;
                        stack.push_back(ni);
                    }
                }
                // ===== ★ 結束搜尋半徑程式段 =====
            }

            obj.avgDx = sumDx / count;
            obj.avgDy = sumDy / count;
            obj.centerX = sumC / (float)count;
            obj.centerY = sumR / (float)count;
            obj.strength = std::sqrt(obj.avgDx * obj.avgDx + obj.avgDy * obj.avgDy);

            objs.push_back(obj);
        }
    }

    return objs;
}

std::vector<PeriodPair> readFallingFrames(const std::string& file_path) {
    std::vector<PeriodPair> fallingPeriods;
    std::ifstream file(file_path);

    // 檢查檔案是否成功開啟
    if (!file.is_open()) {
        std::cerr << "錯誤：無法開啟檔案 " << file_path << std::endl;
        return fallingPeriods;
    }

    std::string line;
    while (std::getline(file, line)) {
        // 1. 移除前後的空白字元
        // 這裡我們只做一個簡易的移除空白，複雜的可以用 C++20 或手動遍歷
        std::string cleanLine = line;

        // 尋找逗號的位置
        size_t commaPos = cleanLine.find(',');
        if (commaPos == std::string::npos) {
            if (!cleanLine.empty()) {
                std::cerr << "警告：資料格式不正確 (缺少逗號): " << line << "，已跳過。" << std::endl;
            }
            continue; // 跳過空行或格式錯誤的行
        }

        try {
            // 提取開始和結束的字串部分
            std::string startStr = cleanLine.substr(0, commaPos);
            std::string endStr = cleanLine.substr(commaPos + 1);

            // 使用 stringstream 來進行更穩健的字串到整數轉換（並忽略內部的空白）
            int startFrame, endFrame;

            std::stringstream ssStart(startStr);
            ssStart >> startFrame;

            std::stringstream ssEnd(endStr);
            ssEnd >> endFrame;

            // 檢查轉換是否成功且數值合理
            if (ssStart.fail() || ssEnd.fail() || startFrame > endFrame) {
                std::cerr << "警告：資料格式或數值錯誤: " << line << "，已跳過。" << std::endl;
                continue;
            }

            // 成功轉換，存入向量
            fallingPeriods.push_back(std::make_pair(startFrame, endFrame));

        }
        catch (const std::exception& e) {
            std::cerr << "處理行時發生未知錯誤: " << line << ". 錯誤: " << e.what() << std::endl;
        }
    }

    return fallingPeriods;
}

/**
 * @brief 判斷給定的影格索引是否處於任何一個跌倒時間段內。
 * 判斷條件是：start_frame <= frame_index <= end_frame
 *
 * @param frame_index 當前要檢查的影格索引。
 * @param falling_periods 跌倒時間段的向量。
 * @return bool 如果影格處於任何一個跌倒時間段內，則返回 true，否則返回 false。
 */
bool isFrameFalling(int frame_index, const std::vector<PeriodPair>& falling_periods) {
    for (const auto& period : falling_periods) {
        int start_frame = period.first;
        int end_frame = period.second;

        // 核心判斷邏輯
        if (frame_index >= start_frame && frame_index <= end_frame) {
            return true;
        }
    }
    return false;
}
// 將每一幀找到的物件 block 畫在影像上
void drawMotionObjects(
    cv::Mat& frame,
    const std::vector<MotionObject>& objects,
    int rows, int cols)
{
    int imgW = frame.cols;
    int imgH = frame.rows;

    float blockW = (float)imgW / cols;
    float blockH = (float)imgH / rows;

    for (const auto& obj : objects) {

        // 每個物件使用不同顏色（用 id 做 hash）
        cv::Scalar color(
            (37 * obj.id) % 255,
            (17 * obj.id + 80) % 255,
            (97 * obj.id + 150) % 255
        );

        // 取得該物件的所有 block
        for (int idx : obj.blocks) {

            int r = idx / cols;
            int c = idx % cols;

            // 取得 block 在影像中的像素位置
            int x1 = (int)(c * blockW);
            int y1 = (int)(r * blockH);
            int x2 = (int)((c + 1) * blockW);
            int y2 = (int)((r + 1) * blockH);

            // 畫 block 的邊框（可換成填色）
            cv::rectangle(frame,
                cv::Point(x1, y1),
                cv::Point(x2, y2),
                color,
                2);
        }

        // -----------------------------
        // 畫物件中心點（以 block 為座標）
        // -----------------------------
        int cx = (int)((obj.centerX + 0.5f) * blockW);
        int cy = (int)((obj.centerY + 0.5f) * blockH);
        cv::circle(frame, cv::Point(cx, cy), 6, color, cv::FILLED);

        // -----------------------------
        // 標上 ID（可關掉）
        // -----------------------------
        //cv::putText(frame,
        //    "ID:" + std::to_string(obj.id),
        //    cv::Point(cx, cy - 10),
        //    cv::FONT_HERSHEY_SIMPLEX,
        //    0.6,
        //    color,
        //    2);
        cv::putText(frame,
            "mv:" + std::to_string(obj.strength),
            cv::Point(cx, cy - 10),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(0,0,255),//color,
            2);
    }
}

bool detectObjectTemporalMotion(
    const std::vector<std::vector<MotionObject>>& history,
    float movementThreshold,
    int M, int N,
    std::string& outWarning)
{
    int T = history.size();
    if (T < M) return false;

    int start = T - M;
    int countFrames = 0;

    for (int f = start; f < T; f++) 
    {
        for (auto& obj : history[f]) 
        {
            //if (obj.strength > movementThreshold) // &&  obj.avgDy < 0
            if (obj.strength > movementThreshold &&  obj.avgDy < 0 && obj.blocks.size() > 4)
            {
                countFrames++;
                //printf("OVER\n");
                printf("obj.strength %f,  DXDY :(%f, %f)\n", obj.strength, obj.avgDx, obj.avgDy);
                break;
            }
        }
        if (countFrames >= N) {
            outWarning = "⚠️ 警告：最近 " + std::to_string(M) +
                " 幀內，有至少 " + std::to_string(N) +
                " 幀出現移動物件超過門檻！";
            return true;
        }
    }
    //printf("---------------------------\n");
    return false;
}


inline float cross(const Point2f& a, const Point2f& b, const Point2f& c)
{
    // cross(AB, AC)
    return (b.x - a.x) * (c.y - a.y) -
        (b.y - a.y) * (c.x - a.x);
}
bool isPointInConvexQuad(const Point2f poly[4], const Point2f& p)
{
    float c1 = cross(poly[0], poly[1], p);
    float c2 = cross(poly[1], poly[2], p);
    float c3 = cross(poly[2], poly[3], p);
    float c4 = cross(poly[3], poly[0], p);

    // 全部 >=0 或 全部 <=0 即為凸多邊形內
    bool pos = (c1 >= 0 && c2 >= 0 && c3 >= 0 && c4 >= 0);
    bool neg = (c1 <= 0 && c2 <= 0 && c3 <= 0 && c4 <= 0);

    return pos || neg;
}
bool isBlockInsideTrapezoid(
    const Point2f quad[4],
    int bx, int by,
    int blockW, int blockH)
{
    Point2f C[4] = {
        {float(bx),             float(by)},
        {float(bx + blockW),    float(by)},
        {float(bx + blockW),    float(by + blockH)},
        {float(bx),             float(by + blockH)}
    };

    // 四個角都必須 inside
    return  isPointInConvexQuad(quad, C[0]) &&
        isPointInConvexQuad(quad, C[1]) &&
        isPointInConvexQuad(quad, C[2]) &&
        isPointInConvexQuad(quad, C[3]);
}
inline int crossProduct(const Point& a, const Point& b, int x, int y) {
    return (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x);
}
// 建立遮罩：背景為 255 (保留)，梯形內為 0 (抹除)
Image createTrapezoidMask(int width, int height, const std::vector<Point2f>& trapPoints) 
{
    Image mask(width, height, 1);

    // 1. 初始化背景為全白 (255)
    // 假設 Image 內部是連續記憶體，你可以用 memset，或者用雙層迴圈填滿 255
    // 這裡用最保險的迴圈寫法：
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            mask.at(x, y) = 255;
        }
    }

    // 2. 計算 Bounding Box (只在這個範圍內計算幾何，加快生成速度)
    int minX = width, maxX = 0;
    int minY = height, maxY = 0;

    for (const auto& p : trapPoints) {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
    }

    // 邊界防呆
    minX = std::max(0, minX);
    maxX = std::min(width - 1, maxX);
    minY = std::max(0, minY);
    maxY = std::min(height - 1, maxY);

    const Point& p0 = trapPoints[0];
    const Point& p1 = trapPoints[1];
    const Point& p2 = trapPoints[2];
    const Point& p3 = trapPoints[3];

    // 3. 掃描 Bounding Box，將梯形內部設為 0
    for (int y = minY; y <= maxY; y++) {
        for (int x = minX; x <= maxX; x++) {
            // 判斷點是否在多邊形內 (假設點順序正確)
            int cp1 = crossProduct(p0, p1, x, y);
            int cp2 = crossProduct(p1, p2, x, y);
            int cp3 = crossProduct(p2, p3, x, y);
            int cp4 = crossProduct(p3, p0, x, y);

            bool allPositive = (cp1 >= 0) && (cp2 >= 0) && (cp3 >= 0) && (cp4 >= 0);
            bool allNegative = (cp1 <= 0) && (cp2 <= 0) && (cp3 <= 0) && (cp4 <= 0);

            if (allPositive || allNegative) {
                mask.at(x, y) = 0; // 梯形區域設為 0 (遮蔽)
            }
        }
    }

    return mask;
}


class OptimizedBlockMotionEstimator {
private:
    int block_size;
    int search_range;
    int horizontal_blocks;
    int vertical_blocks;
    Image prev_frame;  // 改用 Image
    vector<vector<MotionVector>> motion_history;

    // 性能監控
    int frame_count;
    double total_time;

public:
    OptimizedBlockMotionEstimator(int h_blocks = 8, int v_blocks = 6,
        int block_sz = 16, int search_rng = 6)
        : horizontal_blocks(h_blocks), vertical_blocks(v_blocks),
        block_size(block_sz), search_range(search_rng),
        frame_count(0), total_time(0.0) {}

    // 優化的預處理 - 使用整數運算
    Image preprocessFrame(const Image& frame) {
        // 轉換為灰階
        Image gray = frame.toGray();

        // 調整大小以確保可以被區塊整除
        int h = gray.height();
        int w = gray.width();
        int new_h = (h / vertical_blocks) * vertical_blocks;
        int new_w = (w / horizontal_blocks) * horizontal_blocks;

        Image resized = gray.resize(new_w, new_h);
        return resized;
    }

    // 優化的區塊分割 - 預計算索引
    void divideIntoBlocks(const Image& image, vector<Image>& blocks,
        vector<BlockPosition>& positions) {
        blocks.clear();
        positions.clear();

        int h = image.height();
        int w = image.width();
        int block_h = h / vertical_blocks;
        int block_w = w / horizontal_blocks;

        for (int i = 0; i < vertical_blocks; i++) {
            for (int j = 0; j < horizontal_blocks; j++) {
                int y_start = i * block_h;
                int y_end = y_start + block_h;
                int x_start = j * block_w;
                int x_end = x_start + block_w;

                Image block = image.getROI(x_start, x_end, y_start, y_end);
                blocks.push_back(block);

                BlockPosition pos;
                pos.i = i;
                pos.j = j;
                pos.y_start = y_start;
                pos.y_end = y_end;
                pos.x_start = x_start;
                pos.x_end = x_end;
                positions.push_back(pos);
            }
        }
    }

    // 計算前後幀差異並二值化
    Image binarizeDifference(const Image& prev, const Image& curr, int threshold = 30) {
        Image binary_diff(prev.width(), prev.height(), 1);

        // 計算絕對差異並二值化
        for (int y = 0; y < prev.height(); y++) 
        {
            for (int x = 0; x < prev.width(); x++) 
            {
                
                int diff = abs(static_cast<int>(curr.at(x, y)) - static_cast<int>(prev.at(x, y)));
                

                //原本的做法
                //binary_diff.at(x, y) = (diff > threshold) ? 255 : 0;
                //新的做法: 會跟床的mask做and, 在床的區域內會被刪掉
                binary_diff.at(x, y) = ((diff > threshold) ? 255 : 0)& bedRegion.at(x, y);
            }
        }
            
        return binary_diff;
    }

    // 檢測有變化的區塊
    int detectChangedBlocks(const Image& prev, const Image& curr,
        vector<bool>& changed_blocks_mask,
        vector<BlockPosition>& positions,
        float threshold_ratio = 0.01f) 
    {
        if (prev.empty()) {
            changed_blocks_mask.assign(horizontal_blocks * vertical_blocks, false);
            return 0;
        }

        // 前後幀畫素值，差超過門檻30則二值化255，反之為0
        Image binary_diff = binarizeDifference(prev, curr, 25);




        vector<Image> blocks;
        divideIntoBlocks(binary_diff, blocks, positions);

        changed_blocks_mask.clear();
        int counter = 0;

        for (const auto& block : blocks) 
        {
            
            // 計算區塊內變化像素的比例
            int change_pixels = 0;
            for (int y = 0; y < block.height(); y++) {
                for (int x = 0; x < block.width(); x++) {
                    if (block.at(x, y) > 0) {
                        change_pixels++;
                    }
                }
            }

            float change_ratio = static_cast<float>(change_pixels) /
                (float)(block.height() * block.width());
            bool changed = change_ratio > threshold_ratio;
            changed_blocks_mask.push_back(changed);

            if (changed) counter++;
        }

        return counter;
    }

    // 計算兩個區塊的SAD（絕對誤差和）
    int calculateSADFast(const Image& block1, const Image& block2) {
        int h = min(block1.height(), block2.height());
        int w = min(block1.width(), block2.width());

        int sad = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                sad += abs(static_cast<int>(block1.at(x, y)) -
                    static_cast<int>(block2.at(x, y)));
            }
        }

        return sad;
    }

    // 快速鑽石搜尋 - 減少搜尋點
    void fastDiamondSearch(const Image& block_curr, const Image& prev_frame,
        int x_start, int y_start, int block_h, int block_w,
        int predicted_dx, int predicted_dy,
        int& best_dx, int& best_dy, int& best_sad) {
        best_dx = predicted_dx;
        best_dy = predicted_dy;
        best_sad = numeric_limits<int>::max();

        // 簡化的鑽石模式，減少搜尋點
        //const int small_diamond[][2] = { {0,0}, {-1,0}, {1,0}, {0,-1}, {0,1} };
        //int diamond_size = 5;
        const int small_diamond[][2] = { {0, 0},{0, -2}, {-2, 0}, {2, 0}, {0, 2}, {-1, -1}, {1, -1}, {-1, 1}, {1, 1} };
        int diamond_size = 9;


        bool improved = true;
        int search_iterations = 0;
        const int max_iterations = 6;

        while (improved && search_iterations < max_iterations) {
            improved = false;
            int current_best_dx = best_dx;
            int current_best_dy = best_dy;

            for (int k = 0; k < diamond_size; k++) {
                int dx_offset = small_diamond[k][0];
                int dy_offset = small_diamond[k][1];

                int candidate_dx = current_best_dx + dx_offset;
                int candidate_dy = current_best_dy + dy_offset;

                // 邊界檢查
                if (abs(candidate_dx) > search_range || abs(candidate_dy) > search_range)
                    continue;

                int ref_y_start = y_start + candidate_dy;
                int ref_y_end = ref_y_start + block_h;
                int ref_x_start = x_start + candidate_dx;
                int ref_x_end = ref_x_start + block_w;

                if (ref_y_start < 0 || ref_y_end > prev_frame.height() ||
                    ref_x_start < 0 || ref_x_end > prev_frame.width())
                    continue;

                Image block_prev = prev_frame.getROI(ref_x_start, ref_x_end,
                    ref_y_start, ref_y_end);
                int sad = calculateSADFast(block_curr, block_prev);

                if (sad < best_sad) {
                    best_sad = sad;
                    best_dx = candidate_dx;
                    best_dy = candidate_dy;
                    improved = true;
                }
            }

            search_iterations++;
        }
    }
    void LDSP_SDSP_Search(const Image& block_curr, const Image& prev_frame,
        int x_start, int y_start, int block_h, int block_w,
        int predicted_dx, int predicted_dy,
        int& best_dx, int& best_dy, int& best_sad) {

        // --- 1. 初始化 ---
        // 先計算起始預測點 (Center) 的 SAD，作為比較的基準
        best_dx = predicted_dx;
        best_dy = predicted_dy;
        best_sad = numeric_limits<int>::max();

        // 初始位置邊界檢查與 SAD 計算 (Center Point)
        {
            int ref_x = x_start + best_dx;
            int ref_y = y_start + best_dy;

            // 確保預測點本身沒有出界
            if (abs(best_dx) <= search_range && abs(best_dy) <= search_range &&
                ref_x >= 0 && ref_x + block_w <= prev_frame.width() &&
                ref_y >= 0 && ref_y + block_h <= prev_frame.height()) {

                Image block_prev = prev_frame.getROI(ref_x, ref_x + block_w, ref_y, ref_y + block_h);
                best_sad = calculateSADFast(block_curr, block_prev);
            }
        }

        // 定義 LDSP (大鑽石) 的 8 個周圍點 (不含中心，半徑=2)
        // 形狀: 十字延伸2格 + 四個角落(1,1)
        const int ldsp_offsets[][2] = {
            {0, -2}, {1, -1}, {2, 0}, {1, 1},
            {0, 2}, {-1, 1}, {-2, 0}, {-1, -1}
        };
        const int ldsp_count = 8;

        // 定義 SDSP (小鑽石) 的 4 個周圍點 (不含中心，半徑=1)
        // 形狀: 上下左右
        const int sdsp_offsets[][2] = {
            {0, -1}, {1, 0}, {0, 1}, {-1, 0}
        };
        const int sdsp_count = 4;

        // --- 2. 階段一：LDSP (Large Diamond Search Pattern) ---
        // 持續移動大鑽石，直到最佳點落在中心
        int search_iterations = 0;
        const int max_iterations = 10; // 防止無窮迴圈的安全機制
        bool center_is_best = false;

        while (!center_is_best && search_iterations < max_iterations) {
            center_is_best = true; // 假設中心是最好的，除非找到更好的鄰居

            // 暫存這一輪找到的最佳位置，避免在同一輪中連續跳動
            int next_center_dx = best_dx;
            int next_center_dy = best_dy;
            int local_min_sad = best_sad;

            for (int k = 0; k < ldsp_count; k++) {
                int candidate_dx = best_dx + ldsp_offsets[k][0];
                int candidate_dy = best_dy + ldsp_offsets[k][1];

                // 全域搜尋範圍檢查
                if (abs(candidate_dx) > search_range || abs(candidate_dy) > search_range)
                    continue;

                int ref_x = x_start + candidate_dx;
                int ref_y = y_start + candidate_dy;

                // 影像邊界檢查
                if (ref_x < 0 || ref_x + block_w > prev_frame.width() ||
                    ref_y < 0 || ref_y + block_h > prev_frame.height())
                    continue;

                // 取得 ROI 並計算 SAD
                Image block_prev = prev_frame.getROI(ref_x, ref_x + block_w, ref_y, ref_y + block_h);
                int sad = calculateSADFast(block_curr, block_prev);

                // 更新局部最佳 (注意：這裡跟 local_min_sad 比)
                if (sad < local_min_sad) {
                    local_min_sad = sad;
                    next_center_dx = candidate_dx;
                    next_center_dy = candidate_dy;
                    center_is_best = false; // 發現鄰居比中心好，需要繼續移動
                }
            }

            // 更新全域最佳狀態
            if (!center_is_best) {
                best_dx = next_center_dx;
                best_dy = next_center_dy;
                best_sad = local_min_sad;
            }

            search_iterations++;
        }

        // --- 3. 階段二：SDSP (Small Diamond Search Pattern) ---
        // 當 LDSP 停止 (最佳點在中心) 或超過迭代次數後，做最後一次精細搜尋
        // 以目前的 best_dx, best_dy 為中心

        // 注意：SDSP 只要跑一次即可，不需要 while 迴圈
        for (int k = 0; k < sdsp_count; k++) {
            int candidate_dx = best_dx + sdsp_offsets[k][0];
            int candidate_dy = best_dy + sdsp_offsets[k][1];

            if (abs(candidate_dx) > search_range || abs(candidate_dy) > search_range)
                continue;

            int ref_x = x_start + candidate_dx;
            int ref_y = y_start + candidate_dy;

            if (ref_x < 0 || ref_x + block_w > prev_frame.width() ||
                ref_y < 0 || ref_y + block_h > prev_frame.height())
                continue;

            Image block_prev = prev_frame.getROI(ref_x, ref_x + block_w, ref_y, ref_y + block_h);
            int sad = calculateSADFast(block_curr, block_prev);

            if (sad < best_sad) {
                best_sad = sad;
                best_dx = candidate_dx;
                best_dy = candidate_dy;
            }
        }
    }
    // 三步搜尋算法 - 經典快速算法
    void threeStepSearch(const Image& block_curr, const Image& prev_frame,
        int x_start, int y_start, int block_h, int block_w,
        int& best_dx, int& best_dy, int& best_sad) {
        best_dx = 0;
        best_dy = 0;
        best_sad = numeric_limits<int>::max();

        const int step_sizes[] = { 4, 2, 1 };

        for (int s = 0; s < 3; s++) {
            int step = step_sizes[s];
            int current_best_dx = best_dx;
            int current_best_dy = best_dy;

            // 在當前步長下搜尋9個點
            for (int dy = -step; dy <= step; dy += step) {
                for (int dx = -step; dx <= step; dx += step) {
                    int candidate_dx = current_best_dx + dx;
                    int candidate_dy = current_best_dy + dy;

                    // 邊界檢查
                    if (abs(candidate_dx) > search_range || abs(candidate_dy) > search_range)
                        continue;

                    int ref_y_start = y_start + candidate_dy;
                    int ref_y_end = ref_y_start + block_h;
                    int ref_x_start = x_start + candidate_dx;
                    int ref_x_end = ref_x_start + block_w;

                    if (ref_y_start < 0 || ref_y_end > prev_frame.height() ||
                        ref_x_start < 0 || ref_x_end > prev_frame.width())
                        continue;

                    Image block_prev = prev_frame.getROI(ref_x_start, ref_x_end,
                        ref_y_start, ref_y_end);
                    int sad = calculateSADFast(block_curr, block_prev);

                    if (sad < best_sad) {
                        best_sad = sad;
                        best_dx = candidate_dx;
                        best_dy = candidate_dy;
                    }
                }
            }
        }
    }

    // 快速運動預測
    void predictMotionVectorFast(int block_idx, int& predicted_dx, int& predicted_dy) {
        predicted_dx = 0;
        predicted_dy = 0;

        if (motion_history.empty())
            return;

        // 只使用最近一幀的歷史
        const auto& last_vectors = motion_history.back();
        if (block_idx < static_cast<int>(last_vectors.size())) {
            predicted_dx = last_vectors[block_idx].dx;
            predicted_dy = last_vectors[block_idx].dy;
        }
    }

    // 快速紋理複雜度計算
    int calculateTextureComplexityFast(const Image& block) {
        int texture = 0;

        if (block.height() > 8 && block.width() > 8) {
            // 下採樣計算
            Image small_block = block.resize(8, 8);

            // 計算梯度
            for (int y = 0; y < small_block.height() - 1; y++) {
                for (int x = 0; x < small_block.width() - 1; x++) {
                    texture += abs(static_cast<int>(small_block.at(x, y + 1)) -
                        static_cast<int>(small_block.at(x, y)));  // y方向
                    texture += abs(static_cast<int>(small_block.at(x + 1, y)) -
                        static_cast<int>(small_block.at(x, y)));  // x方向
                }
            }
        }
        else {
            // 小區塊直接計算
            for (int y = 0; y < block.height() - 1; y++) {
                for (int x = 0; x < block.width() - 1; x++) {
                    texture += abs(static_cast<int>(block.at(x, y + 1)) -
                        static_cast<int>(block.at(x, y)));
                    texture += abs(static_cast<int>(block.at(x + 1, y)) -
                        static_cast<int>(block.at(x, y)));
                }
            }
        }

        return texture;
    }

    // 優化的運動估計策略
    void optimizedMotionEstimation(const Image& block_curr, const Image& prev_frame,
        int x_start, int y_start, int block_h, int block_w,
        int block_idx, int& best_dx, int& best_dy, int& best_sad) {
        // 快速運動預測
        int predicted_dx, predicted_dy;
        predictMotionVectorFast(block_idx, predicted_dx, predicted_dy);

        // 快速紋理分析
        int texture = calculateTextureComplexityFast(block_curr);


#if 1
        /*fastDiamondSearch(block_curr, prev_frame, x_start, y_start,
            block_h, block_w, predicted_dx, predicted_dy,
            best_dx, best_dy, best_sad);*/
        LDSP_SDSP_Search(block_curr, prev_frame, x_start, y_start,
            block_h, block_w, predicted_dx, predicted_dy,
            best_dx, best_dy, best_sad);
#else
        // 固定使用三步搜尋（根據原註解）
        threeStepSearch(block_curr, prev_frame, x_start, y_start,
            block_h, block_w, best_dx, best_dy, best_sad);
#endif
        // 如果要啟用自適應策略，取消下面註解
        /*
        if (texture < 300) {
            // 平滑區域使用快速鑽石搜尋
            fastDiamondSearch(block_curr, prev_frame, x_start, y_start,
                            block_h, block_w, predicted_dx, predicted_dy,
                            best_dx, best_dy, best_sad);
        } else {
            // 複雜區域使用三步搜尋
            threeStepSearch(block_curr, prev_frame, x_start, y_start,
                            block_h, block_w, best_dx, best_dy, best_sad);
        }
        */
    }

    // 極速運動估計主函數
    void blockBasedMotionEstimation(const Image& curr_frame,
        vector<MotionVector>& motion_vectors,
        vector<BlockPosition>& positions,
        vector<bool>& changed_blocks_mask) {
        auto start_time = chrono::high_resolution_clock::now();

        motion_vectors.clear();
        positions.clear();
        changed_blocks_mask.clear();

        if (prev_frame.empty()) {
            prev_frame = curr_frame.clone();
            int total = horizontal_blocks * vertical_blocks;
            motion_vectors.assign(total, MotionVector(0, 0));
            changed_blocks_mask.assign(total, false);
            return;
        }

        // 變化檢測
        int counter = detectChangedBlocks(prev_frame, curr_frame,
            changed_blocks_mask, positions);
        //cout << "changed counter = " << counter << "/" << changed_blocks_mask.size() << endl;

        vector<Image> blocks_prev, blocks_curr;
        vector<BlockPosition> temp_pos;
        divideIntoBlocks(prev_frame, blocks_prev, temp_pos);
        divideIntoBlocks(curr_frame, blocks_curr, positions);

        int active_blocks_count = 0;

        // 只處理有變化的區塊
        for (size_t idx = 0; idx < changed_blocks_mask.size(); idx++) {
            if (!changed_blocks_mask[idx]) {
                motion_vectors.push_back(MotionVector(0, 0));
                continue;
            }

            const BlockPosition& pos = positions[idx];
            const Image& block_curr = blocks_curr[idx];
            int block_h = block_curr.height();
            int block_w = block_curr.width();

            int best_dx, best_dy, best_sad;

            try {
                // 使用優化的運動估計
                optimizedMotionEstimation(block_curr, prev_frame,
                    pos.x_start, pos.y_start,
                    block_h, block_w, idx,
                    best_dx, best_dy, best_sad);

                // 簡化的可靠性檢查
                if (best_sad > block_h * block_w * 10) {
                    best_dx = best_dy = 0;
                }

                motion_vectors.push_back(MotionVector(best_dx, best_dy));
                active_blocks_count++;

            }
            catch (...) {
                motion_vectors.push_back(MotionVector(0, 0));
            }
        }

        // 更新歷史（限制大小）
        updateMotionHistoryFast(motion_vectors);
        prev_frame = curr_frame.clone();

        // 性能統計
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(
            end_time - start_time).count();
        total_time += duration;
        frame_count++;

        if (frame_count % 30 == 0) {
            double avg_time = total_time / frame_count;
            //cout << "平均處理時間: " << avg_time << "ms, 活動區塊: "
            //    << active_blocks_count << endl;
        }
    }

    // 快速歷史更新
    void updateMotionHistoryFast(const vector<MotionVector>& motion_vectors) {
        motion_history.push_back(motion_vectors);

        // 只保留最近5幀
        if (motion_history.size() > 5) {
            motion_history.erase(motion_history.begin());
        }
    }

#ifdef _CV_DEBUG_
    // 優化的可視化
    Mat visualizeOnFrameFast(const Image& frame,
        const vector<MotionVector>& motion_vectors,
        const vector<BlockPosition>& positions,
        const vector<bool>& changed_blocks_mask) {
        // 轉換為 BGR 格式用於繪圖 (僅顯示用)
        Mat display_frame;
        Mat temp = frame.toMat();

        if (temp.channels() == 1) {
            cvtColor(temp, display_frame, COLOR_GRAY2BGR);
        }
        else {
            display_frame = temp.clone();
        }

        int active_vectors = 0;

        for (size_t idx = 0; idx < changed_blocks_mask.size(); idx++) {
            if (!changed_blocks_mask[idx])
                continue;

            const MotionVector& vector = motion_vectors[idx];
            if (vector.dx == 0 && vector.dy == 0)
                continue;

            const BlockPosition& pos = positions[idx];
            int dx = vector.dx;
            int dy = vector.dy;

            int center_x = (pos.x_start + pos.x_end) / 2;
            int center_y = (pos.y_start + pos.y_end) / 2;

            // 計算向量終點
            int scale = 3;
            int end_x = center_x - dx * scale;
            int end_y = center_y - dy * scale;

            // 根據運動強度選擇顏色
            int magnitude = abs(dx) + abs(dy);

            Scalar color;
            if (magnitude > 10) {
                color = Scalar(0, 0, 255);  // 紅色
            }
            else if (magnitude > 5) {
                color = Scalar(0, 165, 255);  // 橙色
            }
            else {
                color = Scalar(0, 255, 255);  // 黃色
            }

            // 繪製箭頭
            arrowedLine(display_frame,
                Point(center_x, center_y),
                Point(end_x, end_y),
                color, 2, LINE_AA, 0, 0.2);

            // 標記變化區塊
            rectangle(display_frame,
                Point(pos.x_start, pos.y_start),
                Point(pos.x_end, pos.y_end),
                color, 2);

            active_vectors++;
        }

        // 顯示性能信息
        string info_text = "Active: " + to_string(active_vectors);
        putText(display_frame, info_text, Point(10, 25),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);

        return display_frame;
    }
#endif

    void resetPrevFrame() {
        prev_frame = Image();  // 重置為空影像
        motion_history.clear();
    }
};

std::vector<Point2f> loadPointsFromTxt(const std::string& filename, float width_scale, float height_scale) {
    std::vector<Point2f> points;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "錯誤: 無法開啟檔案 " << filename << std::endl;
        return points; // 回傳空 vector
    }

    int x, y;
    char comma; // 用來讀取並忽略中間的 ','

    // 迴圈會自動解析每一行: 讀取 int -> 讀取 char(逗號) -> 讀取 int
    while (file >> x >> comma >> y) {
        points.push_back({ (float)x* width_scale, (float)y* height_scale });
    }

    file.close();

    // 簡單的檢查 (非必要，但建議加上)
    if (points.size() != 4) {
        std::cerr << "警告: 預期讀入 4 個點，但實際讀入 " << points.size() << " 個點。" << std::endl;
    }

    return points;
}

#ifdef _BRUCE_C_V2_
int main(int argc, char** argv) 
{
    Config cfg;

    if (!cfg.load("parameter.ini")) {
        std::cout << "Failed to load INI file!\n";
        return -1;
    }
    //偵測動量
    int Image_Width = cfg.getInt("Motion.Image_Width", 1920);
    int Image_Height = cfg.getInt("Motion.Image_Height", 1080);
    float Width_Scale = cfg.getFloat("Motion.Width_Scale", 0.25f);
    float Height_Scale = cfg.getFloat("Motion.Height_Scale", 0.25f);
    int Binarize_Difference_Threshold = cfg.getInt("Motion.Binarize_Difference_Threshold", 30);
    int Block_Horizontal_Number = cfg.getInt("Motion.Block_Horizontal_Number", 16);
    int Block_Vertical_Number = cfg.getInt("Motion.Block_Vertical_Number", 12);
    float Block_Difference_Ratio_Threshold = cfg.getFloat("Motion.Block_Difference_Ratio_Threshold", 0.01);
    int Max_Iterations_DiamondSearch = cfg.getInt("Motion.Max_Iterations_DiamondSearch", 5);
    int Search_Pattern_Define = cfg.getInt("Motion.Search_Pattern_Define", 0);
    //物件
    int Block_Merge_Range = cfg.getInt("Object.Block_Merge_Range", 1);
    int Block_Minimum_Number = cfg.getInt("Object.Block_Minimum_Number", 4);
    //跌倒偵測
    float Fall_Detect_Minimum_Strength = cfg.getFloat("FallDetect.Fall_Detect_Minimum_Strength", 8.0);
    int Fall_Detect_Frame_History_Length = cfg.getInt("FallDetect.Fall_Detect_Frame_History_Length", 10);
    int Fall_Detect_Frame_History_Threshold = cfg.getInt("FallDetect.Fall_Detect_Frame_History_Threshold", 4);
    int Fall_Detect_Dx_Direct_Limit = cfg.getInt("FallDetect.Fall_Detect_Dx_Direct_Limit", 0);
    int Fall_Detect_Dy_Direct_Limit = cfg.getInt("FallDetect.Fall_Detect_Dy_Direct_Limit", -1);
    //bool enableDebug = cfg.getBool("Options.enable_debug", false);
    //std::string logPath = cfg.getString("Options.log_path", "logs/default.txt");
    std::cout << "===== Motion Parameters =====" << std::endl;
    std::cout << "Image_Width                     = " << Image_Width << std::endl;
    std::cout << "Image_Height                    = " << Image_Height << std::endl;
    std::cout << "Width_Scale                     = " << Width_Scale << std::endl;
    std::cout << "Height_Scale                    = " << Height_Scale << std::endl;
    std::cout << "Binarize_Difference_Threshold   = " << Binarize_Difference_Threshold << std::endl;
    std::cout << "Block_Horizontal_Number         = " << Block_Horizontal_Number << std::endl;
    std::cout << "Block_Vertical_Number           = " << Block_Vertical_Number << std::endl;
    std::cout << "Block_Difference_Ratio_Threshold= " << Block_Difference_Ratio_Threshold << std::endl;
    std::cout << "Max_Iterations_DiamondSearch    = " << Max_Iterations_DiamondSearch << std::endl;
    std::cout << "Search_Pattern_Define           = " << Search_Pattern_Define << std::endl;

    std::cout << "\n===== Object Parameters =====" << std::endl;
    std::cout << "Block_Merge_Range               = " << Block_Merge_Range << std::endl;
    std::cout << "Block_Minimum_Number            = " << Block_Minimum_Number << std::endl;

    std::cout << "\n===== Fall Detection Parameters =====" << std::endl;
    std::cout << "Fall_Detect_Minimum_Strength    = " << Fall_Detect_Minimum_Strength << std::endl;
    std::cout << "Fall_Detect_Dx_Direct_Limit     = " << Fall_Detect_Dx_Direct_Limit << std::endl;
    std::cout << "Fall_Detect_Dy_Direct_Limit     = " << Fall_Detect_Dy_Direct_Limit << std::endl;

    std::cout << "=====================================" << std::endl;


    string vType = "NIR";
    string video_pre_str = "C:\\Users\\88880445\\Downloads\\OptFlowFallDetection_Tim\\2025_09_16_長照錄製_檔名標註\\";
    string bed_position_pre_str = "C:\\Users\\88880445\\Downloads\\OptFlowFallDetection_Tim\\bed_position\\";
    string fall_frame_index_pre_str = "C:\\Users\\88880445\\Downloads\\OptFlowFallDetection_Tim\\fall_frame_index\\";


    //string video_name = "20250917_150455_NIR_有護欄_起身到一半滑坐在地.avi";
    //string video_name = "20250917_161106_NIR_有護欄_起身到一半滑坐在地_光線充足.avi";
    //string video_name = "20250917_144242_NIR_有護欄_起身到一半側跌.avi";
    string video_name = "20250917_145644_NIR_有護欄_起身到一半滑坐在地.avi";
    //string video_name = "20250917_145759_NIR_有護欄_起身到一半往前撲倒.avi";
    //string video_name = "20250917_150217_NIR_有護欄_起身到一半側跌.avi";
    //string video_name = "20250917_150545_NIR_有護欄_站起後往前撲倒.avi";
    //string video_name = "20250917_151652_NIR_有護欄_站起後往床頭走後跌倒.avi";
    //string video_name = "20250917_152012_NIR_有護欄_起身到一半往前趴倒.avi";
    //string video_name = "20250917_152434_NIR_無護欄_翻身後滾下床.avi";
    //string video_name = "20250917_152726_NIR_無護欄_上半身起身到一半跌下床.avi";
    //string video_name = "20250917_152923_NIR_無護欄_上半身起身到一半跌下床.avi";
    //string video_name = "20250917_153845_NIR_從房間外走進來_各種跌倒.avi";           //情境較多, 後續考慮給跌倒分級
    //string video_name = "20250917_154154_NIR_從房間外走進來_各種跌倒_Mark.avi";
    //string video_name = "20250917_155239_NIR_有護欄_起身到一半往床頭方向滑坐在地.avi";
    //string video_name = "20250917_161205_NIR_有護欄_起身到一半往床頭方向滑坐在地_光線充足.avi";
    //string video_name = "20250917_161406_NIR_有護欄_起身後滑坐在地_起身後往床頭滑坐_Mark.avi";


    //string video_name = "20250917_154649_NIR_有護欄_起床後用助行器走出房間.avi";    //沒有一幀是跌倒的影片
    //string video_name = "20250917_154910_NIR_有護欄_起床後正常走出房間.avi";          //沒有一幀是跌倒的影片
    //string video_name = "20250917_155037_NIR_有護欄_起床後用拐杖走出房間.avi";     //沒有一幀是跌倒的影片
    //string video_name = "20250917_160643_NIR_床邊坐輪椅.avi";



    std::string parameter_filename = video_name.substr(0, video_name.find_last_of('.'));
    string bed_position_txt_name = bed_position_pre_str + parameter_filename+".txt";
    string fall_frame_index_txt_name = fall_frame_index_pre_str + parameter_filename + ".txt";
    video_name = video_pre_str + video_name;
    std::string first_image_name = parameter_filename + "_first_frame.jpg";



    int video_w = 1920;
    int video_h = 1080;
    int algo_w = 800;
    int algo_h = 450;
    float width_scale = algo_w / (float)video_w;
    float height_scale = algo_h / (float)video_h;
    int h_blocks = 8*2;//8
    int v_blocks = 6*2;//6
    int block_size = 16; // 實際上算法沒用到, 大小僅由上面兩個格數參數決定
    int search_range = 6*4;// 實際上對算法沒影響

    float extract_obj_mov_th = 1.2; //合併block為物件時, Block至少動量要超過的門檻(強度), norm(dx,dy) > th,才有資格跟其他block合併成物件

    int detect_fall_frame_buffer_size = 10; //保存最近幾幀的物件資訊, 會一直更新並將最舊的frame擠出去
    int detect_fall_frame_event_th = 2; //最近M幀內若有N幀都含有動量較大的物件, 則反饋為跌倒(暫定), 後續須加上方向性
    float detect_fall_displacement_th =5;

    //讀取床的範圍資訊
    trapPoints = loadPointsFromTxt(bed_position_txt_name, width_scale, height_scale);
    bedRegion = createTrapezoidMask(algo_w, algo_h, trapPoints);

    //讀取跌倒的時間段
    periods = readFallingFrames(fall_frame_index_txt_name);



    cv::Mat bed = bedRegion.toMat();
    cv::imshow("bed", bed);

    //const char txtName[1024] = "150455_algo_C_800x450_threeStep_output.txt";
    const char txtName[1024] = "161106_algo_C_800x450_diamond_output_new_par.txt";

    //const char txtName[1024] = "161106_algo_C_800x600_threeStep_output.txt";

    ofstream ofs_1(txtName);
    if (!ofs_1.is_open())
    {
        std::cout << "cannot open output.txt" << endl;
        return 1;
    }


    

    // 開啟影片或攝像頭
    VideoCapture cap;
    if (argc > 1) {
        cap.open(video_name);
    }
    else {
        // 預設使用攝像頭
        //cap.open(0);
        cap.open(video_name);
    }

    if (!cap.isOpened()) {
        cout << "錯誤：無法打開影片或攝像頭" << endl;
        return -1;
    }

    // 創建優化的運動估計器
    OptimizedBlockMotionEstimator estimator(h_blocks, v_blocks, block_size, search_range);

    cout << "極速運動向量檢測開始" << endl;
    cout << "優化策略:  + LDSP+SDSP + 整數運算" << endl;
    cout << "按 'q' 鍵退出, 按 'r' 鍵重置" << endl;

    auto last_time = chrono::high_resolution_clock::now();
    int frame_idx = 0;
    std::vector<std::vector<MotionObject>> all_obj;

    int all_changed_num = 0;
    bool detect_fall_wrong = false;
    bool detect_fall = false;
    bool current_frame_fall = false;
    while (true) 
    {
        Mat cv_frame;

        if (!cap.read(cv_frame)) {
            cout << "錯誤：無法讀取影格或影片結束" << endl;
            break;
        }
        
        
        if (frame_idx == 0) cv::imwrite(first_image_name, cv_frame);

        Mat resized_cv_frame;
        cv::resize(cv_frame, resized_cv_frame, Size(algo_w, algo_h));

        // 轉換為自定義Image格式
        Image frame = Image::fromMat(resized_cv_frame);

        Image processed_frame = estimator.preprocessFrame(frame);

        vector<MotionVector> motion_vectors;
        vector<BlockPosition> positions;
        vector<bool> changed_blocks_mask;

        estimator.blockBasedMotionEstimation(processed_frame,
            motion_vectors,
            positions,
            changed_blocks_mask);

        //輸出至txt
        for (size_t ii = 0; ii < positions.size(); ++ii)
        {
            //if (!changed_blocks_mask[ii]) continue;
            if (changed_blocks_mask[ii])   all_changed_num++;

            BlockPosition pos_ = positions[ii];
            MotionVector mov_ = motion_vectors[ii];

            ofs_1 << frame_idx << " " << pos_.j << " " << pos_.i << " " << mov_.dx << " " << mov_.dy << endl;

        }

        //1128 add
        std::vector<MotionObject> obj = extractMotionObjects(motion_vectors, v_blocks, h_blocks, extract_obj_mov_th);

        cv::Mat obj_show = resized_cv_frame.clone();
        drawMotionObjects(obj_show, obj, v_blocks, h_blocks);
        cv::imshow("debug", obj_show);


        all_obj.push_back(obj);
        if (all_obj.size() > detect_fall_frame_buffer_size) all_obj.erase(all_obj.begin());

        //std::cout << "物件數量: " << obj.size() << std::endl;
        detect_fall_wrong = false;
        detect_fall = false;
        current_frame_fall = false;
        std::string show_str;
        detectObjectTemporalMotion(all_obj, detect_fall_displacement_th, detect_fall_frame_buffer_size, detect_fall_frame_event_th, show_str);
        if (show_str != "")
        {
            detect_fall = true;
            //std::cout <<"Frame :"<< frame_idx << ", show_str " << show_str << std::endl;
            bool is_fall = isFrameFalling(frame_idx, periods);
            if (!is_fall)
            {
                std::cout << "誤判 : "<<frame_idx << std::endl;
                detect_fall_wrong = true;
            }
            else
            {
                std::cout << "正解 : " << frame_idx << std::endl;
                //detect_fall_wrong = false;
            }
        }
        



        Mat display_frame = estimator.visualizeOnFrameFast(processed_frame,
            motion_vectors,
            positions,
            changed_blocks_mask);

        // 計算實時 FPS
        auto current_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(
            current_time - last_time).count();
        float fps = duration > 0 ? 1000.0f / duration : 0.0f;
        last_time = current_time;

        string fps_text = "FPS: " + to_string(static_cast<int>(fps));
        //putText(display_frame, fps_text, Point(10, 55),
        //    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 2);

        cv::line(display_frame, cv::Point(trapPoints[0]), cv::Point(trapPoints[1]), cv::Scalar(0, 0, 255), 3);
        cv::line(display_frame, cv::Point(trapPoints[1]), cv::Point(trapPoints[2]), cv::Scalar(0, 255, 0), 3);
        cv::line(display_frame, cv::Point(trapPoints[2]), cv::Point(trapPoints[3]), cv::Scalar(255, 0, 0), 3);
        cv::line(display_frame, cv::Point(trapPoints[3]), cv::Point(trapPoints[0]), cv::Scalar(255, 0, 255), 3);

        //判斷
        if (detect_fall)
        {
            putText(display_frame, "Fall Detect", Point(10, 55), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 2);
        }

        if (detect_fall_wrong && detect_fall)
        {
            putText(display_frame, "Wrong Detect", Point(10, 95), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
        }
        else if(!detect_fall_wrong && detect_fall)
        {
            putText(display_frame, "Correct Detect", Point(10, 95), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 2);
        }

        imshow("Optimized Motion Vector Detection", display_frame);

        char key = waitKey(wait_time);
        if (key == 'q' || key == 'Q') {
            break;
        }
        else if (key == 'r' || key == 'R') {
            estimator.resetPrevFrame();
            cout << "系統已重置" << endl;
        }
        else if (key == 'n')
        {
            frame_idx++;
        }
        else if (key == 'm')
        {
            frame_idx--;
        }
        else
        {
            frame_idx++;
            //std::cout << "frame_count " << frame_count << std::endl;
        }
        if (wait_time == 0)
        {
            std::cout << "frame_count " << frame_idx << std::endl;
        }
        
        //frame_idx++;
    }
    printf("all changed num %d\n", all_changed_num);
    cap.release();
    destroyAllWindows();
    cout << "程式結束" << endl;

    return 0;
}
#endif