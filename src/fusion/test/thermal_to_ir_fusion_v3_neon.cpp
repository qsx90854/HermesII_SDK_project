#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <arm_neon.h>
#include <cstring>

// --- BMP Helper (Same) ---
#pragma pack(push, 1)
struct BMPHeader {
    uint16_t file_type{0x4D42};
    uint32_t file_size{0};
    uint16_t reserved1{0};
    uint16_t reserved2{0};
    uint32_t offset_data{0};
    uint32_t size_header{40};
    int32_t width{0};
    int32_t height{0};
    uint16_t planes{1};
    uint16_t bit_count{0};
    uint32_t compression{0};
    uint32_t size_image{0};
    int32_t x_pixels_per_meter{0};
    int32_t y_pixels_per_meter{0};
    uint32_t colors_used{0};
    uint32_t colors_important{0};
};
#pragma pack(pop)

bool read_bmp(const std::string& filename, int& w, int& h, std::vector<unsigned char>& data) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) return false;
    BMPHeader header;
    f.read(reinterpret_cast<char*>(&header), sizeof(header));
    w = header.width;
    h = std::abs(header.height);
    f.seekg(header.offset_data, std::ios::beg);
    data.resize(w * h * 3);
    int padded_row_size = (w * 3 + 3) & (~3);
    std::vector<unsigned char> row(padded_row_size);
    bool flip = (header.height > 0);
    for (int i = 0; i < h; i++) {
        f.read(reinterpret_cast<char*>(row.data()), padded_row_size);
        int y = flip ? (h - 1 - i) : i;
        for (int x = 0; x < w; x++) {
            int idx = (y * w + x) * 3;
            data[idx + 0] = row[x * 3 + 2];
            data[idx + 1] = row[x * 3 + 1];
            data[idx + 2] = row[x * 3 + 0];
        }
    }
    return true;
}

// --- Params ---
struct Vec3 { float x, y, z; };
struct Mat3x3 { float m[3][3]; };
struct CameraMatrix { float fx, fy, cx, cy; };
struct DistCoeffs { float k1, k2, p1, p2, k3; };

void init_hardcoded_params(CameraMatrix& K_th, DistCoeffs& D_th, CameraMatrix& K_ir, DistCoeffs& D_ir, Mat3x3& R, Vec3& T) {
    K_th = {258.32f, 257.88f, 87.64f, 127.12f};
    D_th = {-0.272f, 0.030f, 0.002f, -0.001f, -0.129f};
    K_ir = {1238.84f, 1207.33f, 993.20f, 561.51f};
    D_ir = {-0.380f, 0.175f, 0.0001f, 0.0006f, -0.045f};
    R.m[0][0] = 0.9995f; R.m[0][1] = -0.0079f; R.m[0][2] = -0.0296f;
    R.m[1][0] = 0.0083f; R.m[1][1] = 0.9998f; R.m[1][2] = 0.0127f;
    R.m[2][0] = 0.0295f; R.m[2][1] = -0.0130f; R.m[2][2] = 0.9994f;
    T = {13.83f, 24.52f, 3.56f};
}

void project_point(float X, float Y, float Z, const CameraMatrix& K, const DistCoeffs& D, float& u, float& v) {
    float x = X / Z;
    float y = Y / Z;
    float r2 = x*x + y*y;
    float r4 = r2*r2;
    float r6 = r2*r4;
    float radial = 1.0f + D.k1*r2 + D.k2*r4 + D.k3*r6;
    float x_d = x * radial + 2.0f*D.p1*x*y + D.p2*(r2 + 2.0f*x*x);
    float y_d = y * radial + D.p1*(r2 + 2.0f*y*y) + 2.0f*D.p2*x*y;
    u = K.fx * x_d + K.cx;
    v = K.fy * y_d + K.cy;
}

void transform_ir_to_thermal(
    float x_ir, float y_ir,
    const CameraMatrix& new_K_ir, float dist_meters,
    const CameraMatrix& K_th, const DistCoeffs& D_th,
    const Mat3x3& R, const Vec3& T,
    float& u_th, float& v_th
) {
    float x_norm = (x_ir - new_K_ir.cx) / new_K_ir.fx;
    float y_norm = (y_ir - new_K_ir.cy) / new_K_ir.fy;
    float P_ir_x = x_norm * dist_meters;
    float P_ir_y = y_norm * dist_meters;
    float P_ir_z = dist_meters;
    float Pc_x = P_ir_x - T.x;
    float Pc_y = P_ir_y - T.y;
    float Pc_z = P_ir_z - T.z;
    float P_th_x = Pc_x * R.m[0][0] + Pc_y * R.m[1][0] + Pc_z * R.m[2][0];
    float P_th_y = Pc_x * R.m[0][1] + Pc_y * R.m[1][1] + Pc_z * R.m[2][1];
    float P_th_z = Pc_x * R.m[0][2] + Pc_y * R.m[1][2] + Pc_z * R.m[2][2];
    project_point(P_th_x, P_th_y, P_th_z, K_th, D_th, u_th, v_th);
}

struct GridPoint {
    float u_ir, v_ir;
    float u_th, v_th;
};

// Optimized Bilinear Sample (Integer Math approximation for speed)
// u, v are in pixel coordinates.
inline void sample_fast(const unsigned char* img, int w, int h, float u, float v, uint8_t* out) {
    int x = (int)u;
    int y = (int)v;
    if (x < 0 || x >= w - 1 || y < 0 || y >= h - 1) {
        out[0]=0; out[1]=0; out[2]=0; return;
    }
    
    // Fixed point weights (8-bit precision)
    int fx = (int)((u - x) * 256.0f);
    int fy = (int)((v - y) * 256.0f);
    int ifx = 256 - fx;
    int ify = 256 - fy;
    
    int w00 = (ifx * ify) >> 8;
    int w10 = (fx * ify) >> 8;
    int w01 = (ifx * fy) >> 8;
    int w11 = (fx * fy) >> 8;
    
    int idx = (y * w + x) * 3;
    const unsigned char* p = img + idx;
    int stride = w * 3;
    
    // Unrolling for RGB
    // Row 0
    int r0 = p[0] * w00 + p[3] * w10;
    int g0 = p[1] * w00 + p[4] * w10;
    int b0 = p[2] * w00 + p[5] * w10;
    
    // Row 1
    const unsigned char* p1 = p + stride;
    int r1 = p1[0] * w01 + p1[3] * w11;
    int g1 = p1[1] * w01 + p1[4] * w11;
    int b1 = p1[2] * w01 + p1[5] * w11;
    
    out[0] = (uint8_t)((r0 + r1) >> 8);
    out[1] = (uint8_t)((g0 + g1) >> 8);
    out[2] = (uint8_t)((b0 + b1) >> 8);
}

int main() {
    CameraMatrix K_th, K_ir; DistCoeffs D_th, D_ir; Mat3x3 R; Vec3 T;
    init_hardcoded_params(K_th, D_th, K_ir, D_ir, R, T);
    CameraMatrix new_K_ir = K_ir;
    float dist = 1000.0f;

    std::string th_path = "test_image/Thermal_4.bmp";
    std::string ir_path = "test_image/IR_4.bmp";
    int w, h, w_th, h_th;
    std::vector<unsigned char> img_ir_data, img_th_data;
    if (!read_bmp(ir_path, w, h, img_ir_data)) return -1;
    if (!read_bmp(th_path, w_th, h_th, img_th_data)) return -1;

    // --- Sparse Grid ---
    const int GRID_SIZE = 8; 
    int grid_cols = (w + GRID_SIZE - 1) / GRID_SIZE + 1;
    int grid_rows = (h + GRID_SIZE - 1) / GRID_SIZE + 1;
    
    std::vector<GridPoint> grid(grid_cols * grid_rows);

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int gy = 0; gy < grid_rows; ++gy) {
        for (int gx = 0; gx < grid_cols; ++gx) {
            int x = std::min(gx * GRID_SIZE, w - 1);
            int y = std::min(gy * GRID_SIZE, h - 1);
            float u_ir, v_ir, u_th, v_th;
            
            // IR
            float x_norm = (x - new_K_ir.cx) / new_K_ir.fx;
            float y_norm = (y - new_K_ir.cy) / new_K_ir.fy;
            float P_x = x_norm * dist, P_y = y_norm * dist;
            project_point(P_x, P_y, dist, K_ir, D_ir, u_ir, v_ir);
            
            // Thermal
            transform_ir_to_thermal((float)x, (float)y, new_K_ir, dist, K_th, D_th, R, T, u_th, v_th);
            grid[gy * grid_cols + gx] = {u_ir, v_ir, u_th, v_th};
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Grid Build Time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " us" << std::endl;

    std::ofstream out_file("rectified_result_v3_neon.bmp", std::ios::binary);
    int padded_row_size = (w * 3 + 3) & (~3);
    BMPHeader header; header.file_size = sizeof(BMPHeader) + padded_row_size * h; 
    header.offset_data = sizeof(BMPHeader); header.width = w; header.height = -h; header.bit_count = 24; header.size_image = padded_row_size * h;
    out_file.write((char*)&header, sizeof(header));
    std::vector<unsigned char> row_buffer(padded_row_size);
    unsigned char* ir_ptr = img_ir_data.data();
    unsigned char* th_ptr = img_th_data.data();

    auto t_proc_start = std::chrono::high_resolution_clock::now();

    // Iterate Scanline for Streaming Output

    
    // Correct Loop Structure for Streaming + Optimization:
    // Outer Y (0..h).
    //   Calculate current Grid Row (gy).
    //   Calculate alpha_y, and Left/Right interpolants for ALL columns? No.
    //   Iterate X blocks i.e. gx = 0 .. cols.
    //     Calculate Left/Right for this block.
    //     Iterate x inside block.
    
    for (int y = 0; y < h; ++y) {
        int gy = y / GRID_SIZE;
        if (gy >= grid_rows - 1) gy = grid_rows - 2;
        float alpha_y = (float)(y - gy * GRID_SIZE) / GRID_SIZE;
        
        for (int gx = 0; gx < grid_cols - 1; ++gx) {
            int x_start = gx * GRID_SIZE;
            int x_end = std::min(x_start + GRID_SIZE, w);
            if (x_start >= x_end) continue;

            const GridPoint& p00 = grid[gy * grid_cols + gx];
            const GridPoint& p10 = grid[gy * grid_cols + (gx+1)];
            const GridPoint& p01 = grid[(gy+1) * grid_cols + gx];
            const GridPoint& p11 = grid[(gy+1) * grid_cols + (gx+1)];

            // Optimization: Precompute row ends
            float u_ir_L = p00.u_ir * (1-alpha_y) + p01.u_ir * alpha_y;
            float v_ir_L = p00.v_ir * (1-alpha_y) + p01.v_ir * alpha_y;
            float u_th_L = p00.u_th * (1-alpha_y) + p01.u_th * alpha_y;
            float v_th_L = p00.v_th * (1-alpha_y) + p01.v_th * alpha_y;

            float u_ir_R = p10.u_ir * (1-alpha_y) + p11.u_ir * alpha_y;
            float v_ir_R = p10.v_ir * (1-alpha_y) + p11.v_ir * alpha_y;
            float u_th_R = p10.u_th * (1-alpha_y) + p11.u_th * alpha_y;
            float v_th_R = p10.v_th * (1-alpha_y) + p11.v_th * alpha_y;

            float d_u_ir = (u_ir_R - u_ir_L) / GRID_SIZE;
            float d_v_ir = (v_ir_R - v_ir_L) / GRID_SIZE;
            float d_u_th = (u_th_R - u_th_L) / GRID_SIZE;
            float d_v_th = (v_th_R - v_th_L) / GRID_SIZE;

            float cur_u_ir = u_ir_L;
            float cur_v_ir = v_ir_L;
            float cur_u_th = u_th_L;
            float cur_v_th = v_th_L;

            // Tight inner loop (Incremental)
            for (int x = x_start; x < x_end; ++x) {
                // Sample (Integer Optimized)
                uint8_t rgb_ir[3], rgb_th[3];
                sample_fast(ir_ptr, w, h, cur_u_ir, cur_v_ir, rgb_ir);
                sample_fast(th_ptr, w_th, h_th, cur_u_th, cur_v_th, rgb_th);

                // Blend (Avg) - Scalar (can be NEONized if batched, but sampling dominates)
                // NEON blend: vrhadd_u8 is excellent here. 
                // But for 3 bytes, scalar is fine. Let's try to load 8 bytes if we can. 
                // For structure: we are computing 3 bytes.
                
                int idx = x * 3;
                // BGR output
                row_buffer[idx+0] = (rgb_ir[2] + rgb_th[2]) >> 1; // B
                row_buffer[idx+1] = (rgb_ir[1] + rgb_th[1]) >> 1; // G
                row_buffer[idx+2] = (rgb_ir[0] + rgb_th[0]) >> 1; // R

                // Increment
                cur_u_ir += d_u_ir;
                cur_v_ir += d_v_ir;
                cur_u_th += d_u_th;
                cur_v_th += d_v_th;
            }
        }
        //out_file.write((char*)row_buffer.data(), padded_row_size);
    }

    auto t_proc_end = std::chrono::high_resolution_clock::now();
    std::cout << "Optimized Processing Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_proc_end - t_proc_start).count() << " ms" << std::endl;

    return 0;
}
