#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cstring>

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

// LUT Entry: Fixed Point Q12.4 (1 bit sign, 11 bits int, 4 bits frac)
// Range: -2048 to 2047 pixels
// Precision: 1/16th pixel
struct LutEntry {
    int16_t u, v; 
};

// Fast Sampler using Fixed Point coords
inline void sample_fixed(const unsigned char* img, int w, int h, int16_t u_fixed, int16_t v_fixed, uint8_t* out) {
    // Decode Q4
    int x_int = u_fixed >> 4;
    int y_int = v_fixed >> 4;
    int x_frac = u_fixed & 0xF; // 0..15
    int y_frac = v_fixed & 0xF;
    
    // Bounds Check
    if (x_int < 0 || x_int >= w - 1 || y_int < 0 || y_int >= h - 1) {
        out[0]=0; out[1]=0; out[2]=0; return;
    }
    
    // Weights (0..256 scale)
    // Normalized weight alpha = x_frac / 16.0
    // w1 = alpha, w0 = 1 - alpha.
    // In integers 0..16:
    int wf_x = x_frac; // 0..15
    int wif_x = 16 - x_frac;
    
    int wf_y = y_frac;
    int wif_y = 16 - y_frac;
    
    int w00 = wif_x * wif_y; // Max 16*16 = 256
    int w10 = wf_x * wif_y;
    int w01 = wif_x * wf_y;
    int w11 = wf_x * wf_y;
    
    int idx = (y_int * w + x_int) * 3;
    const unsigned char* p = img + idx;
    int stride = w * 3;
    
    // Bilinear Interpolation
    // p[0] is Blue, p[1] Green, p[2] Red (BMP order in memory? No, read_bmp converts to RGB. p[0]=R)
    // Wait, read_bmp: data[dst_idx + 0] = row[x*3+2] (R). So img data is R,G,B.
    int r = p[0]*w00 + p[3]*w10 + p[stride]*w01 + p[stride+3]*w11;
    int g = p[1]*w00 + p[4]*w10 + p[stride+1]*w01 + p[stride+4]*w11;
    int b = p[2]*w00 + p[5]*w10 + p[stride+2]*w01 + p[stride+5]*w11;
    
    out[0] = r >> 8;
    out[1] = g >> 8;
    out[2] = b >> 8;
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

    // --- V4: Downscaled LUT ---
    // Scale 1/2
    int lut_w = w / 2;
    int lut_h = h / 2;
    
    std::cout << "Building 1/2 Scale LUT (" << lut_w << "x" << lut_h << ")..." << std::endl;
    // Store 2 LUTs interleaved? Or separate? Interleaved is cache friendly for same pixel.
    struct PixelLUT { LutEntry ir; LutEntry th; };
    std::vector<PixelLUT> lut(lut_w * lut_h);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    
    for (int y = 0; y < lut_h; ++y) {
        for (int x = 0; x < lut_w; ++x) {
            // Map LUT pixel to Real Pixel (Center of 2x2 block)
            // x_real = x * 2; y_real = y * 2;
            // Let's sample exactly at x*2, y*2.
            int real_x = x * 2;
            int real_y = y * 2;
            
            // IR Calc
            float x_norm = (real_x - new_K_ir.cx) / new_K_ir.fx;
            float y_norm = (real_y - new_K_ir.cy) / new_K_ir.fy;
            float P_x = x_norm * dist, P_y = y_norm * dist;
            float u_ir_f, v_ir_f;
            project_point(P_x, P_y, dist, K_ir, D_ir, u_ir_f, v_ir_f);
            
            // Thermal Calc
            float u_th_f, v_th_f;
            transform_ir_to_thermal((float)real_x, (float)real_y, new_K_ir, dist, K_th, D_th, R, T, u_th_f, v_th_f);
            
            // Convert to Q4 Fixed Point and Store
            // 16.0f factor
            lut[y * lut_w + x].ir.u = (int16_t)(u_ir_f * 16.0f);
            lut[y * lut_w + x].ir.v = (int16_t)(v_ir_f * 16.0f);
            lut[y * lut_w + x].th.u = (int16_t)(u_th_f * 16.0f);
            lut[y * lut_w + x].th.v = (int16_t)(v_th_f * 16.0f);
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "LUT Build Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
    
    std::ofstream out_file("rectified_result_v4.bmp", std::ios::binary);
    int padded_row_size = (w * 3 + 3) & (~3);
    BMPHeader header; header.file_size = sizeof(BMPHeader) + padded_row_size * h; 
    header.offset_data = sizeof(BMPHeader); header.width = w; header.height = -h; header.bit_count = 24; header.size_image = padded_row_size * h;
    out_file.write((char*)&header, sizeof(header));
    std::vector<unsigned char> row_buffer(padded_row_size);
    unsigned char* ir_ptr = img_ir_data.data();
    unsigned char* th_ptr = img_th_data.data();

    auto t_proc_start = std::chrono::high_resolution_clock::now();
    
    for (int y = 0; y < h; ++y) {
        int lut_y = y >> 1; // Divide by 2
        if (lut_y >= lut_h) lut_y = lut_h - 1;
        const PixelLUT* row_lut = &lut[lut_y * lut_w];
        
        // Optimization: Horizontal Reuse
        // x and x+1 share the same lut_x (x>>1)
        for (int x = 0; x < w; x += 2) {
            int lut_x = x >> 1; // Divide by 2
            if (lut_x >= lut_w) lut_x = lut_w - 1;
            
            const PixelLUT& entry = row_lut[lut_x];
            
            uint8_t rgb_ir[3], rgb_th[3];
            sample_fixed(ir_ptr, w, h, entry.ir.u, entry.ir.v, rgb_ir);
            sample_fixed(th_ptr, w_th, h_th, entry.th.u, entry.th.v, rgb_th);
            
            uint8_t b = (rgb_ir[2] + rgb_th[2]) >> 1;
            uint8_t g = (rgb_ir[1] + rgb_th[1]) >> 1;
            uint8_t r = (rgb_ir[0] + rgb_th[0]) >> 1;
            
            // Pixel x
            int idx = x * 3;
            row_buffer[idx+0] = b; 
            row_buffer[idx+1] = g; 
            row_buffer[idx+2] = r;
            
            // Pixel x+1 (Identical Result)
            if (x + 1 < w) {
                int idx2 = (x + 1) * 3;
                row_buffer[idx2+0] = b; 
                row_buffer[idx2+1] = g; 
                row_buffer[idx2+2] = r;
            }
        }
        //out_file.write((char*)row_buffer.data(), padded_row_size);
    }
    
    auto t_proc_end = std::chrono::high_resolution_clock::now();
    std::cout << "V4 Processing Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_proc_end - t_proc_start).count() << " ms" << std::endl;

    return 0;
}
