#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <algorithm>
#include <chrono>

// --- BMP Helper (Same as V2) ---
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
            data[idx + 0] = row[x * 3 + 2]; // R
            data[idx + 1] = row[x * 3 + 1]; // G
            data[idx + 2] = row[x * 3 + 0]; // B
        }
    }
    return true;
}

// --- Math & Params (Same as V2) ---
struct Vec3 { float x, y, z; };
struct Mat3x3 { float m[3][3]; };
struct CameraMatrix { float fx, fy, cx, cy; };
struct DistCoeffs { float k1, k2, p1, p2, k3; };

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

void sample_bilinear(const std::vector<unsigned char>& img, int w, int h, float u, float v, float* rgb) {
    if (u < 0 || u >= w - 1 || v < 0 || v >= h - 1) { rgb[0]=rgb[1]=rgb[2]=0; return; }
    int x0 = (int)u, y0 = (int)v;
    int x1 = x0 + 1, y1 = y0 + 1;
    float alpha = u - x0, beta = v - y0;
    float w00 = (1 - alpha) * (1 - beta), w10 = alpha * (1 - beta);
    float w01 = (1 - alpha) * beta, w11 = alpha * beta;
    for (int k = 0; k < 3; ++k) {
        rgb[k] = w00 * img[(y0*w + x0)*3 + k] + w10 * img[(y0*w + x1)*3 + k] +
                 w01 * img[(y1*w + x0)*3 + k] + w11 * img[(y1*w + x1)*3 + k];
    }
}

// --- V3 Implementation: Sparse Grid ---
struct GridPoint {
    float u_ir, v_ir; // Mapped coords in Raw IR
    float u_th, v_th; // Mapped coords in Raw Thermal
};

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
    std::cout << "Loaded IR: " << w << "x" << h << ", Thermal: " << w_th << "x" << h_th << std::endl;

    // --- Sparse Grid Config ---
    const int GRID_SIZE = 8; 
    int grid_cols = (w + GRID_SIZE - 1) / GRID_SIZE + 1;
    int grid_rows = (h + GRID_SIZE - 1) / GRID_SIZE + 1;
    
    std::cout << "Building Sparse Grid (" << grid_cols << "x" << grid_rows << ")..." << std::endl;
    std::vector<GridPoint> grid(grid_cols * grid_rows);

    auto t1 = std::chrono::high_resolution_clock::now();

    // 1. Build Grid (Compute Exact Coords at Vertices)
    for (int gy = 0; gy < grid_rows; ++gy) {
        for (int gx = 0; gx < grid_cols; ++gx) {
            int x = std::min(gx * GRID_SIZE, w - 1);
            int y = std::min(gy * GRID_SIZE, h - 1);
            
            // Calc A: Rect IR -> Raw IR
            float x_norm = (x - new_K_ir.cx) / new_K_ir.fx;
            float y_norm = (y - new_K_ir.cy) / new_K_ir.fy;
            float P_ir_x = x_norm * dist;
            float P_ir_y = y_norm * dist;
            float P_ir_z = dist;
            float u_ir, v_ir;
            project_point(P_ir_x, P_ir_y, P_ir_z, K_ir, D_ir, u_ir, v_ir);
            
            // Calc B: Rect IR -> Raw Thermal
            float u_th, v_th;
            transform_ir_to_thermal((float)x, (float)y, new_K_ir, dist, K_th, D_th, R, T, u_th, v_th);
            
            grid[gy * grid_cols + gx] = {u_ir, v_ir, u_th, v_th};
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Grid Build Time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " us" << std::endl;

    // 2. Process Image (Interpolate from Grid)
    std::ofstream out_file("rectified_result_v3.bmp", std::ios::binary);
    int padded_row_size = (w * 3 + 3) & (~3);
    BMPHeader header; header.file_size = sizeof(BMPHeader) + padded_row_size * h; 
    header.offset_data = sizeof(BMPHeader); header.width = w; header.height = -h; header.bit_count = 24; header.size_image = padded_row_size * h;
    out_file.write((char*)&header, sizeof(header));
    std::vector<unsigned char> row_buffer(padded_row_size, 0);

    auto t_proc_start = std::chrono::high_resolution_clock::now();

    for (int y = 0; y < h; ++y) {
        int gy = y / GRID_SIZE;
        int dy = y % GRID_SIZE;
        float alpha_y = (float)dy / GRID_SIZE;
        
        // Handle last row edge case if h not divisible by GRID_SIZE
        if (gy >= grid_rows - 1) { gy = grid_rows - 2; alpha_y = 1.0f; } // Clamp to last valid cell

        for (int x = 0; x < w; ++x) {
            int gx = x / GRID_SIZE;
            int dx = x % GRID_SIZE;
            float alpha_x = (float)dx / GRID_SIZE;
            
            if (gx >= grid_cols - 1) { gx = grid_cols - 2; alpha_x = 1.0f; }

            // Bilinear Interpolation within Grid Cell
            const GridPoint& p00 = grid[gy * grid_cols + gx];
            const GridPoint& p10 = grid[gy * grid_cols + (gx+1)];
            const GridPoint& p01 = grid[(gy+1) * grid_cols + gx];
            const GridPoint& p11 = grid[(gy+1) * grid_cols + (gx+1)];

            // Interpolate u_ir
            float u_ir = (1-alpha_x)*(1-alpha_y)*p00.u_ir + alpha_x*(1-alpha_y)*p10.u_ir +
                         (1-alpha_x)*alpha_y*p01.u_ir + alpha_x*alpha_y*p11.u_ir;
            // Interpolate v_ir
            float v_ir = (1-alpha_x)*(1-alpha_y)*p00.v_ir + alpha_x*(1-alpha_y)*p10.v_ir +
                         (1-alpha_x)*alpha_y*p01.v_ir + alpha_x*alpha_y*p11.v_ir;
            // Interpolate u_th
            float u_th = (1-alpha_x)*(1-alpha_y)*p00.u_th + alpha_x*(1-alpha_y)*p10.u_th +
                         (1-alpha_x)*alpha_y*p01.u_th + alpha_x*alpha_y*p11.u_th;
            // Interpolate v_th
            float v_th = (1-alpha_x)*(1-alpha_y)*p00.v_th + alpha_x*(1-alpha_y)*p10.v_th +
                         (1-alpha_x)*alpha_y*p01.v_th + alpha_x*alpha_y*p11.v_th;

            // Sample & Blend
            float rgb_ir[3], rgb_th[3];
            sample_bilinear(img_ir_data, w, h, u_ir, v_ir, rgb_ir);
            sample_bilinear(img_th_data, w_th, h_th, u_th, v_th, rgb_th);

            int idx = x * 3;
            float r = rgb_ir[0]*0.5f + rgb_th[0]*0.5f;
            float g = rgb_ir[1]*0.5f + rgb_th[1]*0.5f;
            float b = rgb_ir[2]*0.5f + rgb_th[2]*0.5f;
            row_buffer[idx+0] = (unsigned char)std::min(std::max(b,0.f),255.f);
            row_buffer[idx+1] = (unsigned char)std::min(std::max(g,0.f),255.f);
            row_buffer[idx+2] = (unsigned char)std::min(std::max(r,0.f),255.f);
        }
        //out_file.write((char*)row_buffer.data(), padded_row_size);
    }
    auto t_proc_end = std::chrono::high_resolution_clock::now();
    std::cout << "Processing Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_proc_end - t_proc_start).count() << " ms" << std::endl;

    return 0;
}
