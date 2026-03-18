/**
 * MapPointV2_example.cpp
 *
 * Demonstrates MapPointV2: given an ROI in the IR image (ir_test2.jpg),
 * maps the four corners to thermal image coordinates and saves the
 * cropped thermal ROI as a JPEG using stb_image / stb_image_write.
 *
 * Camera params source:
 *   IR  : 20260316_refer/camera_params_ir_ori.json
 *   TH  : 20260316_refer/camera_params_th_ori.json
 *   Ext : src/fusion/test/stereo_extrinsics.json
 *
 * NOTE: The thermal image used here is thermal_test2.jpg (original orientation).
 *       The Python reference (20260309.py) uses thermal_test2_f.jpg which is
 *       thermal_test2.jpg rotated 180° counter-clockwise.
 *       Therefore, after MapPointV2 we flip the coordinates:
 *           x_orig = th_W - 1 - x_flipped
 *           y_orig = th_H - 1 - y_flipped
 *
 * Build: see makefile2 target (add as needed, links against libHermesII_sdk.so)
 */

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "HermesII_sdk.h"

#include <iostream>
#include <algorithm>
#include <cmath>

// ============================================================
// Camera Parameters (hardcoded from JSON files)
// ============================================================

// --- IR intrinsics (camera_params_ir_ori.json) ---
// camera_matrix[0][0], [1][1], [0][2], [1][2]
static const float IR_FX = 758.7997f;
static const float IR_FY = 765.2671f;
static const float IR_CX = 923.8562f;
static const float IR_CY = 525.8359f;
// dist_coeff: k1, k2, p1, p2, k3
static const float IR_K1 =  0.134144f;
static const float IR_K2 = -0.471027f;
static const float IR_P1 = -0.019565f;
static const float IR_P2 = -0.013385f;
static const float IR_K3 =  0.378201f;

// --- Thermal intrinsics (camera_params_th_ori.json) ---
static const float TH_FX = 155.8987f;
static const float TH_FY = 155.8953f;
static const float TH_CX =  62.2165f;
static const float TH_CY =  80.9219f;
static const float TH_K1 = -0.367410f;
static const float TH_K2 =  0.264578f;
static const float TH_P1 =  0.000880f;
static const float TH_P2 =  0.001003f;
static const float TH_K3 = -0.022196f;

// --- Extrinsics (stereo_extrinsics.json) ---
// NOTE: note says "Transformation from Thermal (Cam1) to IR (Cam2)"
// MapPointV2 / TransformPointV2 goes IR -> Thermal by:
//   P_cam = (P_ir - T) , then rotated by R
// R stored row-major
static const float EXT_R[9] = {
     0.9992100830694601f, 0.0045686807425445045f, 0.03947577799860541f,
     -0.007593586059857406f, 0.9970160400652578f, 0.07682026622802023f,
     -0.03900701658771169f, -0.07705934731663037f,  0.996263172885589f
};
static const float EXT_T[3] = {
    6.308468115270685f,
    49.09361311464897f,
    17.417980590779344f
};
static const float ASSUMED_DISTANCE_MM = 1000.0f;

// ============================================================
// Coordinate flip helper
// thermal_test2.jpg = thermal_test2_f.jpg rotated 180 deg CCW
// So mapped point (u,v) in the "flipped" space maps to:
//   u_orig = TH_W - 1 - u
//   v_orig = TH_H - 1 - v
// ============================================================

int main() {
    // --------------------------------------------------------
    // 1. Build FusionParams
    // --------------------------------------------------------
    VisionSDK::FusionParams p;

    p.K_ir.fx = IR_FX;  p.K_ir.fy = IR_FY;
    p.K_ir.cx = IR_CX;  p.K_ir.cy = IR_CY;
    p.D_ir.k1 = IR_K1;  p.D_ir.k2 = IR_K2;
    p.D_ir.p1 = IR_P1;  p.D_ir.p2 = IR_P2;
    p.D_ir.k3 = IR_K3;

    p.K_th.fx = TH_FX;  p.K_th.fy = TH_FY;
    p.K_th.cx = TH_CX;  p.K_th.cy = TH_CY;
    p.D_th.k1 = TH_K1;  p.D_th.k2 = TH_K2;
    p.D_th.p1 = TH_P1;  p.D_th.p2 = TH_P2;
    p.D_th.k3 = TH_K3;

    for (int i = 0; i < 9; ++i) p.Extrinsics.R[i] = EXT_R[i];
    for (int i = 0; i < 3; ++i) p.Extrinsics.T[i] = EXT_T[i];

    p.assumed_distance_mm = ASSUMED_DISTANCE_MM;

    // --------------------------------------------------------
    // 2. Init SDK and store camera params (call once)
    // --------------------------------------------------------
    VisionSDK::VisionSDK sdk;
    sdk.Init();
    sdk.SetFusionCameraParams(p);   // store params — no need to pass on every call

    // --------------------------------------------------------
    // 3. Load thermal image (thermal_test2.jpg - original orient.)
    // --------------------------------------------------------
    const char* th_path = "20260316_refer/thermal_test2.jpg";
    int th_w, th_h, th_ch;
    unsigned char* th_data = stbi_load(th_path, &th_w, &th_h, &th_ch, 3);
    if (!th_data) {
        std::cerr << "[ERROR] Cannot load: " << th_path << std::endl;
        return -1;
    }
    std::cout << "Thermal image: " << th_w << "x" << th_h << std::endl;

    // --------------------------------------------------------
    // 4. Define IR ROI (x, y, w, h)
    // --------------------------------------------------------
    const float roi_x = 912.f;
    const float roi_y = 302.f;
    const float roi_w = 148.f;
    const float roi_h = 192.f;

    // --------------------------------------------------------
    // 5. Map IR ROI -> Thermal bounding box via MapROI()
    //    Returns the bounding box in thermal_test2_f coordinate space.
    //    Since thermal_test2.jpg is thermal_test2_f.jpg rotated 180 deg,
    //    we flip the resulting bounding box.
    // --------------------------------------------------------
    float tx, ty, tw, th_roi;
    if (sdk.MapROI(roi_x, roi_y, roi_w, roi_h, tx, ty, tw, th_roi) != VisionSDK::StatusCode::OK) {
        std::cerr << "[ERROR] MapROI failed\n";
        stbi_image_free(th_data);
        return -1;
    }
    std::cout << "IR  ROI       : (" << roi_x << "," << roi_y << ") " << roi_w << "x" << roi_h << "\n";
    std::cout << "TH  ROI(pre-flip) : (" << tx << "," << ty << ") " << tw << "x" << th_roi << "\n";

    // Flip bounding box: (tx,ty,tw,th) in the flipped image becomes:
    //   new_x = th_W - tx - tw
    //   new_y = th_H - ty - th_roi
    float ftx = (float)th_w - tx - tw;
    float fty = (float)th_h - ty - th_roi;
    std::cout << "TH  ROI(post-flip): (" << ftx << "," << fty << ") " << tw << "x" << th_roi << "\n";

    // --------------------------------------------------------
    // 6. Crop the thermal ROI (with boundary clamp)
    // --------------------------------------------------------
    int ix1 = std::max(0,    (int)std::floor(ftx));
    int iy1 = std::max(0,    (int)std::floor(fty));
    int ix2 = std::min(th_w, (int)std::ceil(ftx + tw));
    int iy2 = std::min(th_h, (int)std::ceil(fty + th_roi));
    int crop_w = ix2 - ix1;
    int crop_h = iy2 - iy1;

    if (crop_w <= 0 || crop_h <= 0) {
        std::cerr << "[ERROR] Mapped ROI is outside the thermal image.\n";
        stbi_image_free(th_data);
        return -1;
    }

    std::vector<unsigned char> roi_buf(crop_w * crop_h * 3);
    for (int row = 0; row < crop_h; ++row) {
        int src_y = iy1 + row;
        for (int col = 0; col < crop_w; ++col) {
            int src_x = ix1 + col;
            int si = (src_y * th_w + src_x) * 3;
            int di = (row * crop_w + col) * 3;
            roi_buf[di+0] = th_data[si+0];
            roi_buf[di+1] = th_data[si+1];
            roi_buf[di+2] = th_data[si+2];
        }
    }

    // --------------------------------------------------------
    // 7. Save result using stb_image_write (JPEG)
    // --------------------------------------------------------
    const char* out_path = "thermal_roi_result2.jpg";
    if (stbi_write_jpg(out_path, crop_w, crop_h, 3, roi_buf.data(), 90)) {
        std::cout << "Saved -> " << out_path << "  (" << crop_w << "x" << crop_h << ")\n";
    } else {
        std::cerr << "[ERROR] Failed to write " << out_path << "\n";
    }

    stbi_image_free(th_data);
    return 0;
}
