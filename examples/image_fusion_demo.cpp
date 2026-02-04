#include "HermesII_sdk.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <map>
#include <string>
#include <algorithm>

using namespace VisionSDK;

// Minimal Config Loader
class ConfigLoader {
    std::map<std::string, std::string> data;
    std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\r\n");
        if (std::string::npos == first) return str;
        size_t last = str.find_last_not_of(" \t\r\n");
        return str.substr(first, (last - first + 1));
    }
public:
    bool load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) return false;
        std::string line, section;
        while (std::getline(file, line)) {
            line = trim(line);
            if (line.empty() || line[0] == ';' || line[0] == '#') continue;
            if (line[0] == '[') {
                size_t end = line.find(']');
                if (end != std::string::npos) section = trim(line.substr(1, end - 1));
            } else {
                size_t eq = line.find('=');
                if (eq != std::string::npos) {
                    std::string key = trim(line.substr(0, eq));
                    std::string val = trim(line.substr(eq + 1));
                    if (!section.empty()) key = section + "." + key;
                    data[key] = val;
                }
            }
        }
        return true;
    }
    float getFloat(const std::string& key, float def) {
        if (data.count(key)) try { return std::stof(data[key]); } catch(...) {}
        return def;
    }
};

void fillChecker(Image& img, int size, int offX) {
    for(int y=0; y<img.height; ++y) {
        for(int x=0; x<img.width; ++x) {
            int cx = (x + offX) / size;
            int cy = y / size;
            if ((cx + cy) % 2 == 0) {
                img.data[(y*img.width+x)*3 + 0] = 255;
                img.data[(y*img.width+x)*3 + 1] = 255;
                img.data[(y*img.width+x)*3 + 2] = 255;
            } else {
                 img.data[(y*img.width+x)*3 + 0] = 0;
                img.data[(y*img.width+x)*3 + 1] = 0;
                img.data[(y*img.width+x)*3 + 2] = 0;
            }
        }
    }
}

// Draw a simple circle or X
void drawTarget(Image& img, int cx, int cy, uint8_t r, uint8_t g, uint8_t b) {
    int rad = 50;
    for(int y=cy-rad; y<=cy+rad; ++y) {
        for(int x=cx-rad; x<=cx+rad; ++x) {
            if (x>=0 && x<img.width && y>=0 && y<img.height) {
                if ((x-cx)*(x-cx) + (y-cy)*(y-cy) <= rad*rad) {
                     img.data[(y*img.width+x)*3 + 0] = r;
                     img.data[(y*img.width+x)*3 + 1] = g;
                     img.data[(y*img.width+x)*3 + 2] = b;
                }
            }
        }
    }
}

// BMP Saver
void saveBMP(const std::string& filename, const Image& img) {
    FILE* f = fopen(filename.c_str(), "wb");
    if (!f) return;
    int w = img.width, h = img.height;
    int filesize = 54 + 3 * w * h;
    uint8_t header[54] = {0x42,0x4D, 0,0,0,0, 0,0,0,0, 54,0,0,0, 40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
    header[2] = (uint8_t)(filesize); header[3] = (uint8_t)(filesize>>8); header[4] = (uint8_t)(filesize>>16); header[5] = (uint8_t)(filesize>>24);
    header[18] = (uint8_t)(w); header[19] = (uint8_t)(w>>8); header[20] = (uint8_t)(w>>16); header[21] = (uint8_t)(w>>24);
    header[22] = (uint8_t)(h); header[23] = (uint8_t)(h>>8); header[24] = (uint8_t)(h>>16); header[25] = (uint8_t)(h>>24);
    fwrite(header, 1, 54, f);
    uint8_t pad[3] = {0,0,0};
    int padSize = (4 - (w*3)%4)%4;
    for(int y=h-1; y>=0; --y) {
        for(int x=0; x<w; ++x) {
            uint8_t r = img.data[(y*w+x)*3];
            uint8_t g = img.data[(y*w+x)*3+1];
            uint8_t b = img.data[(y*w+x)*3+2];
            uint8_t bgr[] = {b,g,r};
            fwrite(bgr, 1, 3, f);
        }
        fwrite(pad, 1, padSize, f);
    }
    fclose(f);
}

int main() {
    ConfigLoader cfg;
    if (!cfg.load("camera_params.ini")) {
        std::cerr << "Failed to load camera_params.ini" << std::endl;
        return 1;
    }
    
    VisionSDK::VisionSDK sdk;
    VisionSDK::Config config;
    sdk.Init(config);

    CameraIntrinsics camA, camB;
    CameraExtrinsics extA, extB;
    
    // Load A
    camA.fx = cfg.getFloat("CameraA.fx", 800);
    camA.fy = cfg.getFloat("CameraA.fy", 800);
    camA.cx = cfg.getFloat("CameraA.cx", 400);
    camA.cy = cfg.getFloat("CameraA.cy", 300);
    
    extA.translation[0] = cfg.getFloat("Extrinsics.T_A_0", 0);
    extA.translation[1] = cfg.getFloat("Extrinsics.T_A_1", 0);
    extA.translation[2] = cfg.getFloat("Extrinsics.T_A_2", 1000);
    for(int i=0; i<9; ++i) extA.rotation[i] = cfg.getFloat("Extrinsics.R_A_" + std::to_string(i), (i%4==0)?1.0f:0.0f);

    // Load B
    camB.fx = cfg.getFloat("CameraB.fx", 800);
    camB.fy = cfg.getFloat("CameraB.fy", 800);
    camB.cx = cfg.getFloat("CameraB.cx", 400);
    camB.cy = cfg.getFloat("CameraB.cy", 300);

    extB.translation[0] = cfg.getFloat("Extrinsics.T_B_0", 0);
    extB.translation[1] = cfg.getFloat("Extrinsics.T_B_1", 0);
    extB.translation[2] = cfg.getFloat("Extrinsics.T_B_2", 1000);
    for(int i=0; i<9; ++i) extB.rotation[i] = cfg.getFloat("Extrinsics.R_B_" + std::to_string(i), (i%4==0)?1.0f:0.0f);

    // Create Images
    int W = 800, H = 600;
    std::vector<uint8_t> bufA(W*H*3);
    std::vector<uint8_t> bufB(W*H*3);
    std::vector<uint8_t> bufOut(W*H*3);
    
    Image imgA = {bufA.data(), W, H, 3, 0};
    Image imgB = {bufB.data(), W, H, 3, 0};
    Image imgOut = {bufOut.data(), W, H, 3, 0};
    
    // Fill A (Source) with Red Target
    std::fill(bufA.begin(), bufA.end(), 0);
    drawTarget(imgA, 400, 300, 255, 0, 0); // Red Circle at Center of A
    
    // Fill B (Dest) with Checkerboard
    fillChecker(imgB, 50, 0);
    
    // Fusion
    StatusCode status = sdk.FuseImages3D(imgA, camA, extA, imgB, camB, extB, imgOut);
    if (status != StatusCode::OK) {
        std::cerr << "Fusion Failed!" << std::endl;
        return 1;
    }
    
    // Save
    saveBMP("fusion_out.bmp", imgOut);
    saveBMP("fusion_src_a.bmp", imgA);
    saveBMP("fusion_src_b.bmp", imgB);
    
    std::cout << "Fusion Complete. Saved fusion_out.bmp" << std::endl;
    return 0;
}
