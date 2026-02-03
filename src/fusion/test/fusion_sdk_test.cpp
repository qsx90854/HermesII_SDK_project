#include "HermesII_sdk.h"
#include <iostream>
#include <vector>
#include <cstring>

// Stub Image struct usage
// Assuming Image struct in HermesII_sdk.h is compatible (it is).

// BMP Helper (simplest, copy from thermal_to_ir_fusion)
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

void write_bmp(const char* filename, int w, int h, const unsigned char* data) {
    FILE* f = fopen(filename, "wb");
    if(!f) return;
    
    int padded_row_size = (w * 3 + 3) & (~3);
    int file_size = sizeof(BMPHeader) + padded_row_size * h;
    
    BMPHeader header;
    header.file_size = file_size;
    header.offset_data = sizeof(BMPHeader);
    header.width = w;
    header.height = -h; 
    header.bit_count = 24;
    header.size_image = padded_row_size * h;
    
    fwrite(&header, sizeof(header), 1, f);
    
    std::vector<unsigned char> pad(3, 0);
    for(int y=0; y<h; y++) {
        // Assume data is RGB, BMP needs BGR
        // Copy row
        std::vector<unsigned char> row(padded_row_size);
        for(int x=0; x<w; x++) {
            int idx = (y*w+x)*3;
            row[x*3+0] = data[idx+2]; // B
            row[x*3+1] = data[idx+1]; // G
            row[x*3+2] = data[idx+0]; // R
        }
        fwrite(row.data(), padded_row_size, 1, f);
    }
    fclose(f);
}

// Dummy Read BMP
bool read_bmp(const char* filename, int& w, int& h, std::vector<unsigned char>& data) {
    FILE* f = fopen(filename, "rb");
    if(!f) return false;
    BMPHeader header;
    fread(&header, sizeof(header), 1, f);
    w = header.width;
    h = abs(header.height); // Standard BMP
    
    fseek(f, header.offset_data, SEEK_SET);
    
    data.resize(w*h*3);
    int padded_row_size = (w * 3 + 3) & (~3);
    std::vector<unsigned char> row(padded_row_size);
    
    // If height is negative, it's top-down. If positive, bottom-up.
    bool flip = (header.height > 0);
    
    for(int i=0; i<h; i++) {
        fread(row.data(), padded_row_size, 1, f);
        int y = flip ? (h-1-i) : i;
        for(int x=0; x<w; x++) {
            int idx = (y*w+x)*3;
            data[idx+0] = row[x*3+2]; // R
            data[idx+1] = row[x*3+1]; // G
            data[idx+2] = row[x*3+0]; // B
        }
    }
    fclose(f);
    return true;
}

int main() {
    VisionSDK::VisionSDK sdk;
    
    // 1. Prepare Params
    VisionSDK::FusionParams p;
    // Hardcode same params as test
    p.K_th = {258.32f, 257.88f, 87.64f, 127.12f};
    p.D_th = {-0.272f, 0.030f, 0.002f, -0.001f, -0.129f};
    p.K_ir = {1238.84f, 1207.33f, 993.20f, 561.51f};
    p.D_ir = {-0.380f, 0.175f, 0.0001f, 0.0006f, -0.045f};
    
    // R (Row-major)
    p.Extrinsics.R[0] = 0.9995f; p.Extrinsics.R[1] = -0.0079f; p.Extrinsics.R[2] = -0.0296f;
    p.Extrinsics.R[3] = 0.0083f; p.Extrinsics.R[4] = 0.9998f;  p.Extrinsics.R[5] = 0.0127f;
    p.Extrinsics.R[6] = 0.0295f; p.Extrinsics.R[7] = -0.0130f; p.Extrinsics.R[8] = 0.9994f;
    
    p.Extrinsics.T[0] = 13.83f;
    p.Extrinsics.T[1] = 24.52f;
    p.Extrinsics.T[2] = 3.56f;
    
    p.assumed_distance_mm = 1000.0f;
    
    // 2. Load Images
    int w, h;
    std::vector<unsigned char> ir_data, th_data;
    if(!read_bmp("IR_4.bmp", w, h, ir_data)) { std::cerr << "Err IR" << std::endl; return -1; }
    if(!read_bmp("Thermal_4.bmp", w, h, th_data)) { std::cerr << "Err Th" << std::endl; return -1; } // Re-using w,h assumption
    
    VisionSDK::Image img_ir; 
    img_ir.width = w; img_ir.height = h; img_ir.channels = 3; img_ir.data = ir_data.data();
    
    VisionSDK::Image img_th; 
    img_th.width = w; img_th.height = h; img_th.channels = 3; img_th.data = th_data.data();
    
    // 3. Alloc Output
    std::vector<unsigned char> out_data(w*h*3);
    VisionSDK::Image img_out;
    img_out.width = w; img_out.height = h; img_out.channels = 3; img_out.data = out_data.data();
    
    // 4. Run FuseV2
    std::cout << "Testing FuseImagesV2..." << std::endl;
    if(sdk.FuseImagesV2(img_ir, img_th, p, img_out) != VisionSDK::StatusCode::OK) {
        std::cerr << "Fusion Failed" << std::endl;
        return -1;
    }
    write_bmp("sdk_fused.bmp", w, h, out_data.data());
    std::cout << "Saved sdk_fused.bmp" << std::endl;
    
    // 5. Test MapPointV2
    float ix = 910, iy=490, tx, ty;
    sdk.MapPointV2(ix, iy, p, tx, ty);
    std::cout << "Point Mapped: " << ix << "," << iy << " -> " << tx << "," << ty << std::endl;
    
    return 0;
}
