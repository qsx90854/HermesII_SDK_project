#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cstring>

// --- BMP Helper ---
#define ENABLE_SAVE 1

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

// Reads BMP into RGB format (R, G, B order in memory)
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
    bool flip = (header.height > 0); // Standard BMP is bottom-up
    for (int i = 0; i < h; i++) {
        f.read(reinterpret_cast<char*>(row.data()), padded_row_size);
        int y = flip ? (h - 1 - i) : i;
        for (int x = 0; x < w; x++) {
            int idx = (y * w + x) * 3;
            // BMP File is BGR. We convert to RGB.
            data[idx + 0] = row[x * 3 + 2]; // R
            data[idx + 1] = row[x * 3 + 1]; // G
            data[idx + 2] = row[x * 3 + 0]; // B
        }
    }
    return true;
}

// --- Params (No Distortion Config) ---
struct Vec3 { float x, y, z; };
struct Mat3x3 { float m[3][3]; };
struct CameraMatrix { float fx, fy, cx, cy; };

void init_hardcoded_params(CameraMatrix& K_th, CameraMatrix& K_ir, Mat3x3& R, Vec3& T) {
    // Keep internal params, but ignore Distortion D
    K_th = {247.54826f, 237.34992f, 86.711966f, 127.682808f};
    K_ir = {1072.79086f, 1162.81007f, 860.07652f, 540.80523f};
    
    R.m[0][0] = 0.9995f; R.m[0][1] = -0.0079f; R.m[0][2] = -0.0296f;
    R.m[1][0] = 0.0083f; R.m[1][1] = 0.9998f; R.m[1][2] = 0.0127f;
    R.m[2][0] = 0.0295f; R.m[2][1] = -0.0130f; R.m[2][2] = 0.9994f;
    T = {13.83f, 24.52f, 3.56f};
}

// LUT Entry: Fixed Point Q12.4
struct LutEntry {
    int16_t u, v; 
};

// Fast Sampler using Fixed Point coords
inline void sample_fixed(const unsigned char* img, int w, int h, int16_t u_fixed, int16_t v_fixed, uint8_t* out) {
    int x_int = u_fixed >> 4;
    int y_int = v_fixed >> 4;
    int x_frac = u_fixed & 0xF; 
    int y_frac = v_fixed & 0xF;
    
    if (x_int < 0 || x_int >= w - 1 || y_int < 0 || y_int >= h - 1) {
        out[0]=0; out[1]=0; out[2]=0; return;
    }
    
    int wf_x = x_frac;
    int wif_x = 16 - x_frac;
    int wf_y = y_frac;
    int wif_y = 16 - y_frac;
    
    int w00 = wif_x * wif_y; 
    int w10 = wf_x * wif_y;
    int w01 = wif_x * wf_y;
    int w11 = wf_x * wf_y;
    
    int idx = (y_int * w + x_int) * 3;
    const unsigned char* p = img + idx;
    int stride = w * 3;
    
    // img is RGB
    int r = p[0]*w00 + p[3]*w10 + p[stride]*w01 + p[stride+3]*w11;
    int g = p[1]*w00 + p[4]*w10 + p[stride+1]*w01 + p[stride+4]*w11;
    int b = p[2]*w00 + p[5]*w10 + p[stride+2]*w01 + p[stride+5]*w11;
    
    out[0] = r >> 8; // R
    out[1] = g >> 8; // G
    out[2] = b >> 8; // B
    out[0] = r >> 8; // R
    out[1] = g >> 8; // G
    out[2] = b >> 8; // B
}

#define ENABLE_R_ONLY 0 // 1: Output Rectified Thermal (R) only. 0: Output Fused RGB (Alignment Check).

int main() {
    CameraMatrix K_th, K_ir; Mat3x3 R; Vec3 T;
    init_hardcoded_params(K_th, K_ir, R, T);
    CameraMatrix new_K_ir = K_ir;
    float dist = 1000.0f; 

    std::string th_path = "test_image/Thermal_1_undistorted.bmp";
    std::string ir_path = "test_image/IR_1_undistorted.bmp";
    int w, h, w_th, h_th;
    std::vector<unsigned char> img_ir_data, img_th_data;
    if (!read_bmp(ir_path, w, h, img_ir_data)) return -1;
    if (!read_bmp(th_path, w_th, h_th, img_th_data)) return -1;

    // --- V8-1: Partial LUT based on ROI ---
    // ROI Settings (User Controllable)
    int roi_x1 = 500;//672; 
    int roi_x2 = 1200;//1248;
    int roi_y1 = 0;   
    int roi_y2 = h; // Default to full height
    
    // Align ROI to even numbers for 2x downsampling
    if (roi_x1 % 2 != 0) roi_x1--;
    if (roi_y1 % 2 != 0) roi_y1--;
    
    // LUT Dimensions (Reduced Memory)
    int lut_x_start = roi_x1 / 2;
    int lut_y_start = roi_y1 / 2;
    int lut_w = (roi_x2 - roi_x1) / 2;
    int lut_h = (roi_y2 - roi_y1) / 2;
    
    std::cout << "Building V8-1 Partial LUT (" << lut_w << "x" << lut_h << ")... Memory: " 
              << (lut_w * lut_h * sizeof(LutEntry)) / 1024 << " KB" << std::endl;
              
    struct PixelLUT { LutEntry th; }; 
    std::vector<PixelLUT> lut(lut_w * lut_h);
    
    float fx_ir_inv = 1.0f / new_K_ir.fx;
    float fy_ir_inv = 1.0f / new_K_ir.fy;

    auto t1 = std::chrono::high_resolution_clock::now();
    
    for (int y = 0; y < lut_h; ++y) 
    {
        // Optimized Inner Loop Math
        // Note: y here is relative to LUT start.
        // real_y in image = (lut_y_start + y) * 2;
        int real_y = (lut_y_start + y) * 2;
        
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
            int real_x = (lut_x_start + x) * 2;
            
            float x_norm = (real_x - new_K_ir.cx) * fx_ir_inv;
            float Px = x_norm * dist; 
            
            float P_th_x = r00 * Px + term_x_const;
            float P_th_y = r01 * Px + term_y_const;
            float P_th_z = r02 * Px + term_z_const;
            
            float inv_z = 1.0f / P_th_z;
            float u_th_f = K_th.fx * (P_th_x * inv_z) + K_th.cx;
            float v_th_f = K_th.fy * (P_th_y * inv_z) + K_th.cy;
            
            lut[y * lut_w + x].th.u = (int16_t)(u_th_f * 16.0f);
            lut[y * lut_w + x].th.v = (int16_t)(v_th_f * 16.0f);
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "LUT Build Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
    

    // --- Memory Optimization: Line-by-Line benchmark & Save ---
    
    // Padded row size for BMP
    int row_stride = (w * 3 + 3) & (~3);

    // 1. BENCHMARK
    // We use a SINGLE row buffer to simulate writing, to avoid allocating full image
    std::vector<uint8_t> dummy_row_buffer(row_stride); 
    
    // Always get IR ptr
    unsigned char* ir_ptr = img_ir_data.data();
    unsigned char* th_ptr = img_th_data.data();

    double total_us = 0;
    int iterations = 100;
    
    for (int iter = 0; iter < iterations; ++iter) 
    {
        auto t_proc_start = std::chrono::high_resolution_clock::now();
        
        // Process ROI Lines
        for (int y = roi_y1; y < roi_y2; ++y) {
            int lut_y = (y >> 1) - lut_y_start;
            if (lut_y < 0) continue; 
            if (lut_y >= lut_h) lut_y = lut_h - 1;
            
            const PixelLUT* row_lut = &lut[lut_y * lut_w];
            uint8_t* row_out = dummy_row_buffer.data(); // Reuse same small buffer (Write Only Test)
            
            const unsigned char* row_ir = ir_ptr + y * w * 3;
            // Initialize buffer
            std::memcpy(row_out, row_ir, w * 3);
            
            #if !ENABLE_R_ONLY
                 // const unsigned char* row_ir = ir_ptr + y * w * 3;
            #endif

            for (int x = roi_x1; x < roi_x2; x += 2) {
                int lut_x = (x >> 1) - lut_x_start;
                if (lut_x < 0) continue;
                if (lut_x >= lut_w) lut_x = lut_w - 1;
                
                const PixelLUT& entry = row_lut[lut_x];
                
                uint8_t rgb_th[3];
                sample_fixed(th_ptr, w_th, h_th, entry.th.u, entry.th.v, rgb_th);
                
                #if ENABLE_R_ONLY
                    // Dummy write (RGB)
                    int idx = x * 3;
                    row_out[idx+0] = rgb_th[2]; 
                    row_out[idx+1] = rgb_th[1];
                    row_out[idx+2] = rgb_th[0];
                #else
                    int idx = x * 3;
                    // Fusion
                    int ir_b = row_ir[idx+0];
                    int ir_g = row_ir[idx+1];
                    int ir_r = row_ir[idx+2];
                    
                    row_out[idx+0] = (ir_b * 3 + rgb_th[2]) >> 2; 
                    row_out[idx+1] = (ir_g * 3 + rgb_th[1]) >> 2; 
                    row_out[idx+2] = (ir_r * 3 + rgb_th[0]) >> 2; 
                             
                    if (x+1 < w) {
                        int idx2 = (x+1) * 3;
                        row_out[idx2+0] = (row_ir[idx2+0] * 3 + rgb_th[2]) >> 2; 
                        row_out[idx2+1] = (row_ir[idx2+1] * 3 + rgb_th[1]) >> 2; 
                        row_out[idx2+2] = (row_ir[idx2+2] * 3 + rgb_th[0]) >> 2; 
                     }
                #endif
            }
        }
        auto t_proc_end = std::chrono::high_resolution_clock::now();
        total_us += std::chrono::duration_cast<std::chrono::microseconds>(t_proc_end - t_proc_start).count();
    }
    
    std::cout << "Avg Fusion Time (V8-1): " << total_us / iterations << " us" << std::endl;
                 

    // 2. SAVING (Re-run Logic Line-by-Line)
    #if ENABLE_SAVE
    std::ofstream out_file("rectified_result_v8_1_1.bmp", std::ios::binary);
    BMPHeader header; header.file_size = sizeof(BMPHeader) + row_stride * h; 
    header.offset_data = sizeof(BMPHeader); header.width = w; header.height = -h; header.bit_count = 24; header.size_image = row_stride * h;
    out_file.write((char*)&header, sizeof(header));
    
    std::vector<uint8_t> io_row_buffer(row_stride, 0); // Init black
    
    for(int y=0; y<h; ++y) {
        // Reset row buffer to 0/Black for areas outside ROI if needed
        std::fill(io_row_buffer.begin(), io_row_buffer.end(), 0);
        
        // Check ROI Y
        if (y >= roi_y1 && y < roi_y2) {
            int lut_y = (y >> 1) - lut_y_start;
            if (lut_y >= 0 && lut_y < lut_h) {
                const PixelLUT* row_lut = &lut[lut_y * lut_w];
                
                const unsigned char* row_ir = ir_ptr + y * w * 3;
                std::memcpy(io_row_buffer.data(), row_ir, w * 3);
                
                #if !ENABLE_R_ONLY
                    // const unsigned char* row_ir = ir_ptr + y * w * 3;
                #endif
                
                for (int x = roi_x1; x < roi_x2; x += 2) {
                    int lut_x = (x >> 1) - lut_x_start;
                    if (lut_x < 0 || lut_x >= lut_w) continue;
                    
                    const PixelLUT& entry = row_lut[lut_x];
                    uint8_t rgb_th[3];
                    sample_fixed(th_ptr, w_th, h_th, entry.th.u, entry.th.v, rgb_th);
                    
                    #if ENABLE_R_ONLY
                        // Grayscale Output (Replicate or RGB?) -> Use RGB as requested
                        int idx = x * 3;
                        io_row_buffer[idx+0] = rgb_th[2]; 
                        io_row_buffer[idx+1] = rgb_th[1];
                        io_row_buffer[idx+2] = rgb_th[0];
                        
                        // Pixel 2
                        if (x+1 < w) {
                            io_row_buffer[idx+3] = rgb_th[2];
                            io_row_buffer[idx+4] = rgb_th[1];
                            io_row_buffer[idx+5] = rgb_th[0];
                        }
                    #else
                        // Fusion Output
                        int idx = x * 3;
                        io_row_buffer[idx+0] = (row_ir[idx+0] + rgb_th[2]) >> 1; 
                        io_row_buffer[idx+1] = (row_ir[idx+1] + rgb_th[1]) >> 1; 
                        io_row_buffer[idx+2] = (row_ir[idx+2] + rgb_th[0]) >> 1; 
                        
                        if (x + 1 < w) {
                           int idx2 = (x+1) * 3;
                           io_row_buffer[idx2+0] = (row_ir[idx2+0] + rgb_th[2]) >> 1; 
                           io_row_buffer[idx2+1] = (row_ir[idx2+1] + rgb_th[1]) >> 1; 
                           io_row_buffer[idx2+2] = (row_ir[idx2+2] + rgb_th[0]) >> 1; 
                        }
                    #endif
                }
            }
        }
        // Write Line Immediately
        out_file.write((char*)io_row_buffer.data(), row_stride);
    }
    #endif
    
    std::cout << "Done." << std::endl;

    return 0;
}


