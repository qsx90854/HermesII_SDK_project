#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <arm_neon.h>

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

        // Optimized NEON Loop: Process 8 pixels (32 bytes input, 4 LUT entries)
        int x = 0;
        int neon_chunk = 8;
        for (; x <= w - neon_chunk; x += neon_chunk) {
            int lut_x = x >> 1; // Start index in LUT
            
            // 1. Fetch 4 Thermal Samples (Scalar) and Pack into uint32
            // We use uint32_t to hold 4 x 8-bit samples: [s3 s2 s1 s0]
            uint32_t packed_r = 0;
            //uint32_t packed_g = 0;
            //uint32_t packed_b = 0;
            
            for(int i=0; i<4; i++) {
                int lx = lut_x + i;
                if(lx >= lut_w) lx = lut_w-1;
                
                const auto& l = row_lut[lx].th;
                uint8_t rgb[3];
                // Direct sample to small buffer (still on stack, but registers likely optimize this)
                // Even better: Inline sample logic here to avoid function call overhead if possible
                
                // Inline Sample Opt
                const unsigned char* p = th_ptr + l.offset;
                int r = p[0]*l.w00 + p[3]*l.w10 + stride_th*l.w01 + p[stride_th+3]*l.w11; // Note: p[stride] used in logic
                // Small fix: logic was p[stride]*w01. 
                const unsigned char* p_row2 = p + stride_th;
                int r_calc = p[0]*l.w00 + p[3]*l.w10 + p_row2[0]*l.w01 + p_row2[3]*l.w11;
                
                uint8_t val_r = r_calc >> 8;
                packed_r |= (val_r << (i * 8));
            }
            
            // 2. Prepare Thermal Vector (Duplicate for 2 pixels each)
            // Create Vector [s3 s2 s1 s0 s3 s2 s1 s0]
            uint8x8_t v_base_r = vcreate_u8((uint64_t)packed_r | ((uint64_t)packed_r << 32));
            
            // Zip with itself to interleave: [s3 s3 s2 s2 s1 s1 s0 s0] ... wait order might differ
            // vzip takes (a, b). 
            // a: 3 2 1 0 3 2 1 0
            // b: 3 2 1 0 3 2 1 0
            // result.val[0] : b0 a0 b1 a1 ... -> 0 0 1 1 2 2 3 3. Correct!
            // Wait, vzip order is low elements first?
            // NEON is Little Endian usually. 
            // packed = 0xS3S2S1S0. 
            // vcreate -> Lane 0=S0, Lane 1=S1...
            // vzip -> Lane 0=a0(S0), Lane 1=b0(S0), Lane 2=a1(S1)...
            // So result is S0 S0 S1 S1 S2 S2 S3 S3. Correct duplication.
            
            uint8x8_t v_base_r_dup = vzip_u8(v_base_r, v_base_r).val[0];
            uint8x8_t v_th_r = v_base_r_dup;
            // uint8x8_t v_th_g... (omitted)
            //uint8x8_t v_th_g = vld1_u8(th_g_arr);
            //uint8x8_t v_th_b = vld1_u8(th_b_arr);
            
            // 3. Load 8 pixels of IR (RGB)
            // vld3 deinterleaves into R, G, B planes
            uint8x8x3_t v_ir = vld3_u8(row_ir + x*3);
            
            // v_ir.val[0] = B, val[1] = G, val[2] = R (Wait, read_bmp fills BGR usually? check read_bmp)
            // read_bmp: data[idx+0]=R, data[idx+1]=G, data[idx+2]=B.
            // vld3 loads into val[0]=R, val[1]=G, val[2]=B. Correct.
            // So v_ir.val[0] is R.
            
            // 4. Blend (Rounded Halving Add)
            uint8x8_t v_res_r = vhadd_u8(v_ir.val[0], v_th_r); // IR R (2) + Thermal R
            
            // 5. Store Planar R (Optimized)
            vst1_u8(row_out_r + x, v_res_r);
        }
        
        // Residuals
        for (; x < w; x+=2) {
             int lut_x = x >> 1; 
             if (lut_x >= lut_w) lut_x = lut_w - 1;
             const auto& l = row_lut[lut_x].th;
             
             // Inline Sample (Scalar)
             const unsigned char* p = th_ptr + l.offset;
             const unsigned char* p_row2 = p + stride_th;
             int r_calc = p[0]*l.w00 + p[3]*l.w10 + p_row2[0]*l.w01 + p_row2[3]*l.w11;
             uint8_t th_r_val = r_calc >> 8;
             
             // Pixel 1
             int idx = x * 3;
             row_out_r[x] = (row_ir[idx+0] + th_r_val) >> 1; // R channel
             
             // Pixel 2
             if (x+1 < w) {
                int idx2 = idx+3;
                row_out_r[x+1] = (row_ir[idx2+0] + th_r_val) >> 1; // R channel shared
             }
        }
    }
    
    auto t_proc_end = std::chrono::high_resolution_clock::now();
    total_us += std::chrono::duration_cast<std::chrono::microseconds>(t_proc_end - t_proc_start).count();
    }
    
    // Post-Benchmark: Write to BMP (Reconstruct BGR) for verification
    // This part is OUTSIDE the benchmark timing loop to isolate fusion performance.
    std::cout << "Avg V5 NEON (R-planar) Processing Time: " << (total_us / 1000.0) / iterations << " ms" << std::endl;

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
