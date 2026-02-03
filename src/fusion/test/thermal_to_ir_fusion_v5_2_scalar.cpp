#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstring>
// #include <arm_neon.h> // NEON removed

// --- BMP Helper ---
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

// --- Params ---
struct Vec3 { float x, y, z; };
struct Mat3x3 { float m[3][3]; };
struct CameraMatrix { float fx, fy, cx, cy; };

void init_hardcoded_params(CameraMatrix& K_th, CameraMatrix& K_ir, Mat3x3& R, Vec3& T) {
    K_th = {258.32f, 257.88f, 87.64f, 127.12f};
    K_ir = {1238.84f, 1207.33f, 993.20f, 561.51f};
    R.m[0][0] = 0.9995f; R.m[0][1] = -0.0079f; R.m[0][2] = -0.0296f;
    R.m[1][0] = 0.0083f; R.m[1][1] = 0.9998f; R.m[1][2] = 0.0127f;
    R.m[2][0] = 0.0295f; R.m[2][1] = -0.0130f; R.m[2][2] = 0.9994f;
    T = {13.83f, 24.52f, 3.56f};
}

// Fixed Point Sampler (Scalar)
// Optimized Sampler (Scalar, No Bounds Check, Offset Pre-calculated)
// LUT stores: offset (int32) and weights (packed or separate)
// For simplicity in this step, let's keep weights dynamic but use offset.
// Actually, to fully optimize, we should pre-calc weights too.
// Let's change LUT to store: int32_t offset (TL pixel index), uint8_t w00, w10, w01, w11.
inline void sample_fixed_opt(const unsigned char* img, int32_t offset, uint8_t w00, uint8_t w10, uint8_t w01, uint8_t w11, int stride, uint8_t* out) {
    const unsigned char* p = img + offset;
    
    // R channel only as requested for specific optimization focus (and consistent with previous user edit)
    int r = p[0]*w00 + p[3]*w10 + p[stride]*w01 + p[stride+3]*w11;
    // Note: Inline optimization in main loop makes this function potentially unused in the fast path, 
    // but kept for the residual loop.
    out[0] = r >> 8;  
}

struct LutEntryOpt { 
    int32_t offset; 
    uint8_t w00, w10, w01, w11; 
};
struct PixelLUT { LutEntryOpt th; };

int main() {
    CameraMatrix K_th, K_ir; Mat3x3 R; Vec3 T;
    init_hardcoded_params(K_th, K_ir, R, T);
    CameraMatrix new_K_ir = K_ir;
    float dist = 1000.0f; 

    std::string th_path = "test_image/Thermal_4.bmp";
    std::string ir_path = "test_image/IR_4.bmp";
    int w, h, w_th, h_th;
    std::vector<unsigned char> img_ir_data, img_th_data;
    read_bmp(ir_path, w, h, img_ir_data);
    read_bmp(th_path, w_th, h_th, img_th_data);

    int lut_w = w / 2;
    int lut_h = h / 2;
    
    std::cout << "Building V5 NEON LUT..." << std::endl;
    std::vector<PixelLUT> lut(lut_w * lut_h);
    float fx_ir_inv = 1.0f / new_K_ir.fx;
    float fy_ir_inv = 1.0f / new_K_ir.fy;

    int stride_th = w_th * 3;
    for (int y = 0; y < lut_h; ++y) {
        // ... (Math same as before)
        int real_y = y * 2;
        float y_norm = (real_y - new_K_ir.cy) * fy_ir_inv;
        float Py = y_norm * dist;
        float Pz = dist;
        float Pc_y = Py - T.y;
        float Pc_z = Pz - T.z;
        
        float r00 = R.m[0][0], r01 = R.m[0][1], r02 = R.m[0][2];
        float r10 = R.m[1][0], r11 = R.m[1][1], r12 = R.m[1][2];
        float r20 = R.m[2][0], r21 = R.m[2][1], r22 = R.m[2][2];
        
        float term_x_const = r10*Pc_y + r20*Pc_z - r00*T.x;
        float term_y_const = r11*Pc_y + r21*Pc_z - r01*T.x;
        float term_z_const = r12*Pc_y + r22*Pc_z - r02*T.x;

        for (int x = 0; x < lut_w; ++x) {
            int real_x = x * 2;
            float x_norm = (real_x - new_K_ir.cx) * fx_ir_inv;
            float Px = x_norm * dist;
            float P_th_x = r00 * Px + term_x_const;
            float P_th_y = r01 * Px + term_y_const;
            float P_th_z = r02 * Px + term_z_const;
            float inv_z = 1.0f / P_th_z;
            float u_th_f = K_th.fx * (P_th_x * inv_z) + K_th.cx;
            float v_th_f = K_th.fy * (P_th_y * inv_z) + K_th.cy;
            
            int u_fixed = (int)(u_th_f * 16.0f);
            int v_fixed = (int)(v_th_f * 16.0f);
            
            int x_int = u_fixed >> 4;
            int y_int = v_fixed >> 4;
            
            // Bounds Check & Clamp during LUT Building
            if (x_int < 0) x_int = 0; if (x_int >= w_th - 1) x_int = w_th - 2;
            if (y_int < 0) y_int = 0; if (y_int >= h_th - 1) y_int = h_th - 2;
            
            int x_frac = u_fixed & 0xF; 
            int y_frac = v_fixed & 0xF;
            
            int wf_x = x_frac;
            int wif_x = 16 - x_frac;
            int wf_y = y_frac;
            int wif_y = 16 - y_frac;
            
            lut[y * lut_w + x].th.offset = (y_int * w_th + x_int) * 3;
            lut[y * lut_w + x].th.w00 = (uint8_t)(wif_x * wif_y);
            lut[y * lut_w + x].th.w10 = (uint8_t)(wf_x * wif_y);
            lut[y * lut_w + x].th.w01 = (uint8_t)(wif_x * wf_y);
            lut[y * lut_w + x].th.w11 = (uint8_t)(wf_x * wf_y);
        }
    }
    
    std::ofstream out_file("rectified_result_v5_neon.bmp", std::ios::binary);
    int padded_row_size = (w * 3 + 3) & (~3);
    BMPHeader header; header.file_size = sizeof(BMPHeader) + padded_row_size * h; 
    header.offset_data = sizeof(BMPHeader); header.width = w; header.height = -h; header.bit_count = 24; header.size_image = padded_row_size * h;
    out_file.write((char*)&header, sizeof(header));
    std::vector<uint8_t> fused_r_buffer(w * h);
    const unsigned char* ir_ptr = img_ir_data.data();
    const unsigned char* th_ptr = img_th_data.data();

    double total_us = 0;
    int iterations = 100;
    for (int iter = 0; iter < iterations; ++iter) {
    
    auto t_proc_start = std::chrono::high_resolution_clock::now();
    
    for (int y = 0; y < h; ++y) {
        int lut_y = y >> 1; 
        if (lut_y >= lut_h) lut_y = lut_h - 1;
        const PixelLUT* row_lut = &lut[lut_y * lut_w];
        const unsigned char* row_ir = ir_ptr + y * w * 3;
        uint8_t* row_out_r = fused_r_buffer.data() + y * w;

        // Scalar Loop (Optimized with Offset LUT, Planar Output)
        for (int x = 0; x < w; ++x) {
            int lut_x = x >> 1; 
            if (lut_x >= lut_w) lut_x = lut_w - 1;
            
            // 1. Fetch Thermal Sample (Scalar Optimized)
            const auto& l = row_lut[lut_x].th;
            const unsigned char* p = th_ptr + l.offset;
            const unsigned char* p_row2 = p + stride_th;
            
            // Bilinear Interpolation (Targeting R channel only logic)
            // Note: weights are for the specific LUT point which corresponds to x>>1, y>>1
            // Wait, for V5/V5.2 we reuse the SAME weights for 2x2 block?
            // Yes, lut_x = x >> 1. So x=0,1 use lut_x=0.
            // The weights stored in LUT are calculated based on x_frac from x>>1.
            // So yes, we reuse the exact same sample result for x and x+1?
            // In original code:
            // Neon loop: uses `row_lut[lx]` where `lx = lut_x + i`
            // And stores to `th_samples`.
            // Then duplicates `th_samples[i][0]` to `th_r_arr[i*2]` and `[i*2+1]`.
            // So YES, the fusion value is identical for neighbors.
            
            int r_calc = p[0]*l.w00 + p[3]*l.w10 + p_row2[0]*l.w01 + p_row2[3]*l.w11;
            uint8_t th_r_val = r_calc >> 8;

            // 2. Fetch IR Sample and Fusion
            // IR is BGR (or RGB depending on read_bmp). We want the R channel.
            // read_bmp: 0=R, 1=G, 2=B (actually BGR in standard BMP, but our read_bmp swaps to RGB order?)
            // Let's check read_bmp:
            // data[idx + 0] = row[x * 3 + 2]; // R
            // So data is R, G, B.
            // index for x is x*3.
            // So ir_ptr[x*3 + 0] is R.
            
            uint8_t ir_r = row_ir[x * 3 + 0];
            uint8_t fused_r = (ir_r + th_r_val) >> 1;

            // 3. Store Planar
            row_out_r[x] = fused_r;
        }
    }
    
    auto t_proc_end = std::chrono::high_resolution_clock::now();
    total_us += std::chrono::duration_cast<std::chrono::microseconds>(t_proc_end - t_proc_start).count();
    }
    
    // Post-Benchmark: Write to BMP (Reconstruct BGR) for verification
    // This part is OUTSIDE the benchmark timing loop to isolate fusion performance.
    std::cout << "Avg V5.2 Scalar Processing Time: " << (total_us / 1000.0) / iterations << " ms" << std::endl;

    std::vector<unsigned char> row_bgr(padded_row_size);
    for(int y=0; y<h; ++y) {
        for(int x=0; x<w; ++x) {
             uint8_t val = fused_r_buffer[y*w + x];
             int idx = x * 3;
             row_bgr[idx+0] = val; // B
             row_bgr[idx+1] = val; // G
             row_bgr[idx+2] = val; // R
        }
        out_file.write((char*)row_bgr.data(), padded_row_size);
    }

    return 0;
}
