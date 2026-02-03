/*
 * 編譯指令 (Linux/Embedded ARM):
 * g++ -O3 -march=armv8-a+simd -o main main.cpp
 *
 * 如果在 x86 測試想要模擬 (需要相容層)，或僅編譯邏輯:
 * g++ -O3 -o main main.cpp (但 NEON 部分需被巨集包覆或移除)
 * *注意：此程式碼包含 ARM NEON 實作，請在支援 NEON 的環境編譯*
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <chrono>

// NEON Header
#include <arm_neon.h>

// --- BMP Helper Functions (No external lib) ---
#pragma pack(push, 1)
struct BMPHeader {
    uint16_t file_type{0x4D42};          // File type always BM which is 0x4D42
    uint32_t file_size{0};               // Size of the file (in bytes)
    uint16_t reserved1{0};               // Reserved, always 0
    uint16_t reserved2{0};               // Reserved, always 0
    uint32_t offset_data{0};             // Start position of pixel data (bytes from the beginning of the file)
    uint32_t size{40};                   // Size of this header (in bytes)
    int32_t width{0};                    // width of bitmap in pixels
    int32_t height{0};                   // height of bitmap in pixels
    uint16_t planes{1};                  // No. of planes for the target device, this is always 1
    uint16_t bit_count{24};              // No. of bits per pixel
    uint32_t compression{0};             // 0 or 3 - uncompressed. THIS PROGRAM CONSIDERS ONLY UNCOMPRESSED BMP images
    uint32_t size_image{0};              // 0 - for uncompressed images
    int32_t x_pixels_per_meter{0};
    int32_t y_pixels_per_meter{0};
    uint32_t colors_used{0};             // No. color indexes in the color table. Use 0 for the max number of colors allowed by bit_count
    uint32_t colors_important{0};        // No. of colors used for displaying the bitmap. If 0 all colors are required
};
#pragma pack(pop)

bool read_bmp(const std::string& filename, int& w, int& h, std::vector<unsigned char>& data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;

    BMPHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (header.file_type != 0x4D42) return false; // "BM" check
    if (header.bit_count != 24) return false;     // Only 24-bit supported for simplicity

    w = header.width;
    h = std::abs(header.height);
    data.resize(w * h * 3);

    file.seekg(header.offset_data, file.beg);
    
    // BMP rows are padded to 4-byte boundary
    int padding = (4 - (w * 3) % 4) % 4;
    std::vector<unsigned char> row_data(w * 3 + padding);

    // Read rows. Determine direction based on height sign
    bool flip = (header.height > 0); // Bottom-up if positive
    
    for (int y = 0; y < h; ++y) {
        file.read(reinterpret_cast<char*>(row_data.data()), row_data.size());
        int target_y = flip ? (h - 1 - y) : y;
        
        // BMP is BGR, we usually want RGB or just keep BGR. 
        // original code converted to RGB (stb default). Let's stick to RGB for consistency with later remap/save.
        for(int x=0; x<w; ++x) {
            int src_idx = x * 3;
            int dst_idx = (target_y * w + x) * 3;
            data[dst_idx + 0] = row_data[src_idx + 2]; // R <-- B
            data[dst_idx + 1] = row_data[src_idx + 1]; // G <-- G
            data[dst_idx + 2] = row_data[src_idx + 0]; // B <-- R
        }
    }
    return true;
}

bool write_bmp(const std::string& filename, int w, int h, const unsigned char* data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;

    int padded_row_size = (w * 3 + 3) & (~3);
    int size = padded_row_size * h;

    BMPHeader header;
    header.file_size = sizeof(BMPHeader) + size;
    header.offset_data = sizeof(BMPHeader);
    header.width = w;
    header.height = -h; // Top-down
    header.bit_count = 24;
    header.size_image = size;

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    std::vector<unsigned char> padded_row(padded_row_size, 0);

    for (int y = 0; y < h; ++y) {
        // Convert RGB back to BGR
        for (int x = 0; x < w; ++x) {
            padded_row[x * 3 + 0] = data[(y * w + x) * 3 + 2];
            padded_row[x * 3 + 1] = data[(y * w + x) * 3 + 1];
            padded_row[x * 3 + 2] = data[(y * w + x) * 3 + 0];
        }
        file.write(reinterpret_cast<const char*>(padded_row.data()), padded_row_size);
    }
    return true;
}

// --- 基礎數學結構 ---
struct Vec3 { float x, y, z; };
struct Mat3x3 { float m[3][3]; }; // Row-major
struct CameraMatrix { float fx, fy, cx, cy; };
struct DistCoeffs { float k1, k2, p1, p2, k3; };

// --- 影像結構 ---
struct Image {
    int w, h, c;
    std::vector<unsigned char> data;
};

// --- 數學工具函式 ---

// Hardcoded Parameters Initialization
void init_hardcoded_params(CameraMatrix& K_th, DistCoeffs& D_th, CameraMatrix& K_ir, DistCoeffs& D_ir, Mat3x3& R, Vec3& T) {
    // Thermal Camera Params
    K_th.fx = 258.32029884179315f; 
    K_th.fy = 257.88064819096905f;
    K_th.cx = 87.64961648850591f;  
    K_th.cy = 127.12974658635059f;
    
    D_th.k1 = -0.27226478496697554f;
    D_th.k2 = 0.030104384408577364f;
    D_th.p1 = 0.002327207502631087f;
    D_th.p2 = -0.0017599189641121363f;
    D_th.k3 = -0.12951497351449237f;

    // IR Camera Params
    K_ir.fx = 1238.8419608138552f;
    K_ir.fy = 1207.3314409965117f;
    K_ir.cx = 993.202792998663f;
    K_ir.cy = 561.5114422758486f;

    D_ir.k1 = -0.38037783018524796f;
    D_ir.k2 = 0.17567569491291468f;
    D_ir.p1 = 0.0001772711227356763f;
    D_ir.p2 = 0.0006384138644647992f;
    D_ir.k3 = -0.04562240573034239f;

    // Stereo Extrinsics
    // R (Row-major)
    R.m[0][0] = 0.9995273141115857f; R.m[0][1] = -0.0079753125405994f; R.m[0][2] = -0.029690785351007064f;
    R.m[1][0] = 0.008358292189878123f; R.m[1][1] = 0.999883177808184f; R.m[1][2] = 0.012797253137928719f;
    R.m[2][0] = 0.029585254714949463f; R.m[2][1] = -0.0130393683162706f; R.m[2][2] = 0.999477207132491f;

    // T
    T.x = 13.839705941498458f;
    T.y = 24.522679079337344f;
    T.z = 3.562802134657886f;
}

// 矩陣轉置 (因為 Python code 用 R.T 作為 R_inv)
Mat3x3 transpose(const Mat3x3& src) {
    Mat3x3 dst;
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            dst.m[i][j] = src.m[j][i];
    return dst;
}

// 3D 點投影 (相當於 cv2.projectPoints 的核心)
void project_point(float X, float Y, float Z, const CameraMatrix& K, const DistCoeffs& D, float& u, float& v) {
    float x = X / Z;
    float y = Y / Z;
    
    float r2 = x*x + y*y;
    float r4 = r2*r2;
    float r6 = r2*r4;
    
    // 徑向畸變
    float radial = 1.0f + D.k1*r2 + D.k2*r4 + D.k3*r6;
    
    // 切向畸變
    float x_d = x * radial + 2.0f*D.p1*x*y + D.p2*(r2 + 2.0f*x*x);
    float y_d = y * radial + D.p1*(r2 + 2.0f*y*y) + 2.0f*D.p2*x*y;
    
    // 投影回像素座標
    u = K.fx * x_d + K.cx;
    v = K.fy * y_d + K.cy;
}


// Helper: Bilinear sample with boundary check
void sample_bilinear(const Image& img, float x, float y, float result[3]) {
    int x0 = (int)x;
    int y0 = (int)y;
    
    // Check bounds. If out of bounds, return black/zero (or handle edge)
    if (x0 < 0 || x0 >= img.w - 1 || y0 < 0 || y0 >= img.h - 1) {
        result[0] = 0; result[1] = 0; result[2] = 0;
        return;
    }

    float dx = x - x0;
    float dy = y - y0;
    float w00 = (1.0f - dx) * (1.0f - dy);
    float w10 = dx * (1.0f - dy);
    float w01 = (1.0f - dx) * dy;
    float w11 = dx * dy;

    int stride = img.w * 3;
    const unsigned char* p00 = &img.data[(y0 * stride + x0 * 3)];
    const unsigned char* p10 = p00 + 3;
    const unsigned char* p01 = p00 + stride;
    const unsigned char* p11 = p01 + 3;

    for (int k = 0; k < 3; ++k) {
        result[k] = p00[k] * w00 + p10[k] * w10 + p01[k] * w01 + p11[k] * w11;
    }
}


// Transform Point from Rectified IR View (Undistorted) to Raw Thermal View (Distorted)
void transform_ir_to_thermal(
    float x_ir, float y_ir,
    const CameraMatrix& new_K_ir, float dist_meters,
    const CameraMatrix& K_th, const DistCoeffs& D_th,
    const Mat3x3& R, const Vec3& T,
    float& u_th, float& v_th
) {
    // 1. Normalized Ideal Coordinates (Target View)
    float x_norm = (x_ir - new_K_ir.cx) / new_K_ir.fx;
    float y_norm = (y_ir - new_K_ir.cy) / new_K_ir.fy;

    // 2. 3D Point in Rectified IR View (at assumed distance)
    float P_ir_x = x_norm * dist_meters;
    float P_ir_y = y_norm * dist_meters;
    float P_ir_z = dist_meters;

    // 3. Transform 3D point from IR Space to Thermal Space
    // P_centered = P_ir - T
    float Pc_x = P_ir_x - T.x;
    float Pc_y = P_ir_y - T.y;
    float Pc_z = P_ir_z - T.z;

    // P_th = (P_ir - T) * R
    float P_th_x = Pc_x * R.m[0][0] + Pc_y * R.m[1][0] + Pc_z * R.m[2][0];
    float P_th_y = Pc_x * R.m[0][1] + Pc_y * R.m[1][1] + Pc_z * R.m[2][1];
    float P_th_z = Pc_x * R.m[0][2] + Pc_y * R.m[1][2] + Pc_z * R.m[2][2];

    // 4. Project to Thermal Image
    project_point(P_th_x, P_th_y, P_th_z, K_th, D_th, u_th, v_th);
}

// Map Rect from IR to Thermal
void map_rect_ir_to_thermal(
    float ir_x1, float ir_y1, float ir_x2, float ir_y2,
    const CameraMatrix& new_K_ir, float dist,
    const CameraMatrix& K_th, const DistCoeffs& D_th,
    const Mat3x3& R, const Vec3& T
) {
    float th_x1, th_y1, th_x2, th_y2;
    transform_ir_to_thermal(ir_x1, ir_y1, new_K_ir, dist, K_th, D_th, R, T, th_x1, th_y1);
    transform_ir_to_thermal(ir_x2, ir_y2, new_K_ir, dist, K_th, D_th, R, T, th_x2, th_y2);
    
    std::cout << "Mapped Rect (IR -> Thermal):" << std::endl;
    std::cout << "  IR: (" << ir_x1 << ", " << ir_y1 << ") -> (" << ir_x2 << ", " << ir_y2 << ")" << std::endl;
    std::cout << "  Thermal: (" << th_x1 << ", " << th_y1 << ") -> (" << th_x2 << ", " << th_y2 << ")" << std::endl;
}

int main() {
    std::string thermal_img_path = "test_image/Thermal_4.bmp";
    std::string ir_img_path = "test_image/IR_4.bmp";
    float ASSUMED_DISTANCE = 1000.0f;

    // 1. 載入參數 (Hardcoded)
    std::cout << "Loading parameters..." << std::endl;
    CameraMatrix K_th, K_ir;
    DistCoeffs D_th, D_ir;
    Mat3x3 R; Vec3 T;

    init_hardcoded_params(K_th, D_th, K_ir, D_ir, R, T);

    // 2. 載入影像
    int w, h;
    std::vector<unsigned char> ir_data, th_data;

    std::cout << "Loading IR image: " << ir_img_path << std::endl;
    if (!read_bmp(ir_img_path, w, h, ir_data)) {
        std::cerr << "Failed to load IR image" << std::endl;
        return -1;
    }
    Image img_ir = {w, h, 3, ir_data};

    int w_th_src, h_th_src;
    std::cout << "Loading Thermal image: " << thermal_img_path << std::endl;
    if (!read_bmp(thermal_img_path, w_th_src, h_th_src, th_data)) {
        std::cerr << "Failed to load Thermal image" << std::endl;
        return -1;
    }
    Image img_th = {w_th_src, h_th_src, 3, th_data};

    // 3. New Camera Matrix (Target matches IR size)
    // We treat the "Rectified IR" view as our target view.
    CameraMatrix new_K_ir = K_ir;
    
    // Remove large result buffer. Use streaming output.
    // std::vector<unsigned char> result_data(w * h * 3); // Removed to save ~6MB
    
    std::cout << "Processing (Streaming to rectified_result.bmp)..." << std::endl;

    // Open Output File
    std::ofstream out_file("rectified_result.bmp", std::ios::binary);
    if (!out_file) {
        std::cerr << "Failed to open output file" << std::endl;
        return -1;
    }

    // Write BMP Header (Top-Down)
    int padded_row_size = (w * 3 + 3) & (~3);
    int file_size = sizeof(BMPHeader) + padded_row_size * h;
    
    BMPHeader header;
    header.file_size = file_size;
    header.offset_data = sizeof(BMPHeader);
    header.width = w;
    header.height = -h; // Top-down
    header.bit_count = 24;
    header.size_image = padded_row_size * h;

    out_file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    
    // Row Buffer (reused)
    std::vector<unsigned char> row_buffer(padded_row_size, 0);

    // Pre-calculate loop invariants
    Mat3x3 R_inv = transpose(R); // R_inv = R.T
    
    auto t_start = std::chrono::high_resolution_clock::now();

    // Iterate over Target Image Pixels (Scanline)
    for (int y = 0; y < h; ++y) {
        // Clear padding bytes if necessary (though 0 init covers it, strictly speaking only end matters)
        
        for (int x = 0; x < w; ++x) {
            
            // --- 1. Compute Source Coordinates ---
            
            // A. Map to Raw IR Image (Distorted) from Rectified View
            // Since Target View == Ideal IR View (just undistorted), 
            // we project 3D point back to Raw IR using Distorted Params.
            // Note: transform_ir_to_thermal does logic for Thermal. For IR we do local logic.
            
            float x_norm = (x - new_K_ir.cx) / new_K_ir.fx;
            float y_norm = (y - new_K_ir.cy) / new_K_ir.fy;
            float P_ir_x = x_norm * ASSUMED_DISTANCE;
            float P_ir_y = y_norm * ASSUMED_DISTANCE;
            float P_ir_z = ASSUMED_DISTANCE;

            float u_ir, v_ir;
            project_point(P_ir_x, P_ir_y, P_ir_z, K_ir, D_ir, u_ir, v_ir);

            // B. Map to Raw Thermal Image
            float u_th, v_th;
            transform_ir_to_thermal((float)x, (float)y, new_K_ir, ASSUMED_DISTANCE, K_th, D_th, R, T, u_th, v_th);

            // --- 2. Sample & Blend ---
            
            // Sample IR
            float rgb_ir[3];
            sample_bilinear(img_ir, u_ir, v_ir, rgb_ir);

            // Sample Thermal
            float rgb_th[3];
            sample_bilinear(img_th, u_th, v_th, rgb_th);

            // Blend (Alpha 0.5)
            int idx = x * 3;
            // BMP is BGR! Our data samples are currently RGB because we swapped in read_bmp?
            // Let's check read_bmp.
            // read_bmp: data[dst_idx + 0] = row_data[src_idx + 2]; // R <-- B
            // So img.data is RGB.
            // But BMP file expects BGR.
            // So we must write BGR.
            
            float r = rgb_ir[0] * 0.5f + rgb_th[0] * 0.5f;
            float g = rgb_ir[1] * 0.5f + rgb_th[1] * 0.5f;
            float b = rgb_ir[2] * 0.5f + rgb_th[2] * 0.5f;
            
            row_buffer[idx + 0] = (unsigned char)std::min(std::max(b, 0.0f), 255.0f); // B
            row_buffer[idx + 1] = (unsigned char)std::min(std::max(g, 0.0f), 255.0f); // G
            row_buffer[idx + 2] = (unsigned char)std::min(std::max(r, 0.0f), 255.0f); // R
        }
        
        // Write Row
        out_file.write(reinterpret_cast<const char*>(row_buffer.data()), padded_row_size);
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Processing Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << " ms" << std::endl;
    
    out_file.close();
    std::cout << "Saved rectified_result.bmp" << std::endl;

    // 8. Demo: Map a specific Rect
    std::cout << "\n--- Demo: Map Rect from IR to Thermal ---" << std::endl;
    // Example: Center 100x100 box
    float cx = w / 2.0f;
    float cy = h / 2.0f;
    map_rect_ir_to_thermal(cx - 50, cy - 50, cx + 50, cy + 50, new_K_ir, ASSUMED_DISTANCE, K_th, D_th, R, T);

    return 0;
}