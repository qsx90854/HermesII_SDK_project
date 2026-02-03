#include "image_fusion.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

namespace VisionSDK {

bool ImageFusion::Fuse(const Image& img1, const Image& img2, Image& output) {
    if (img1.width != img2.width || img1.height != img2.height) {
        std::cerr << "[Fusion] Image dimensions do not match!" << std::endl;
        return false;
    }
    std::cout << "[Fusion] Fusing images..." << std::endl;
    output = img1; 
    return true;
}
// --------------------------------------------------------
// Fusion V2 Implementation
// --------------------------------------------------------

// Helper: 3D Projection
static void project_point_v2(float X, float Y, float Z, const FusionIntrinsics& K, const FusionDistCoeffs& D, float& u, float& v) {
    float x = X / Z;
    float y = Y / Z;
    
    float r2 = x*x + y*y;
    float r4 = r2*r2;
    float r6 = r2*r4;
    
    // Radial
    float radial = 1.0f + D.k1*r2 + D.k2*r4 + D.k3*r6;
    
    // Tangential
    float x_d = x * radial + 2.0f*D.p1*x*y + D.p2*(r2 + 2.0f*x*x);
    float y_d = y * radial + D.p1*(r2 + 2.0f*y*y) + 2.0f*D.p2*x*y;
    
    // Project
    u = K.fx * x_d + K.cx;
    v = K.fy * y_d + K.cy;
}

// Helper: Bilinear Sample
static void sample_bilinear_v2(const Image& img, float u, float v, float* rgb) {
    if (u < 0 || u >= img.width - 1 || v < 0 || v >= img.height - 1) {
        rgb[0] = rgb[1] = rgb[2] = 0.0f;
        return;
    }
    
    int x0 = (int)u;
    int y0 = (int)v;
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    float alpha = u - x0;
    float beta = v - y0;
    
    int w = img.width;
    int c = img.channels; // Assuming 3 for RGB/BGR
    if (c < 3) { // Fallback for gray
         rgb[0]=rgb[1]=rgb[2]=0; return; 
    }
    
    unsigned char* data = img.data;
    
    // Weightings
    float w00 = (1.0f - alpha) * (1.0f - beta);
    float w10 = alpha * (1.0f - beta);
    float w01 = (1.0f - alpha) * beta;
    float w11 = alpha * beta;
    
    for (int k = 0; k < 3; ++k) {
        float val = w00 * data[(y0*w + x0)*c + k] +
                    w10 * data[(y0*w + x1)*c + k] +
                    w01 * data[(y1*w + x0)*c + k] +
                    w11 * data[(y1*w + x1)*c + k];
        rgb[k] = val;
    }
}

void ImageFusion::TransformPointV2(float x_ir, float y_ir, const FusionParams& p, float& u_th, float& v_th) {
    // 1. Normalized Ideal Coordinates (IR Rectified View)
    // We treat Rectified IR as having same Intrinsics as Raw IR (for target view definition)
    float x_norm = (x_ir - p.K_ir.cx) / p.K_ir.fx;
    float y_norm = (y_ir - p.K_ir.cy) / p.K_ir.fy;

    // 2. 3D Point in Rectified IR View
    float dist = p.assumed_distance_mm;
    float P_ir_x = x_norm * dist;
    float P_ir_y = y_norm * dist;
    float P_ir_z = dist;

    // 3. Transform to Thermal Space
    // P_th = (P_ir - T) * R
    // Note: R in params is likely R_ir_to_th or similar? 
    // In thermal_to_ir_fusion.cpp: P_th = (P_ir - T) * R => This implies T is translation of Thermal in IR frame?
    // Wait. Let's check the math in previous code.
    // Pc = P_ir - T; P_th = Pc * R. 
    // This implies P_ir = T + P_th * R_inv? 
    // It depends on definition. Let's stick to the ported exact math.
    
    const float* R = p.Extrinsics.R;
    const float* T = p.Extrinsics.T;

    float Pc_x = P_ir_x - T[0];
    float Pc_y = P_ir_y - T[1];
    float Pc_z = P_ir_z - T[2];

    float val_x = Pc_x * R[0] + Pc_y * R[3] + Pc_z * R[6];
    float val_y = Pc_x * R[1] + Pc_y * R[4] + Pc_z * R[7];
    float val_z = Pc_x * R[2] + Pc_y * R[5] + Pc_z * R[8];
    
    project_point_v2(val_x, val_y, val_z, p.K_th, p.D_th, u_th, v_th);
}

bool ImageFusion::FuseV2(const Image& img_ir, const Image& img_th, const FusionParams& p, Image& dst) {
    if (img_ir.width != img_th.width || img_ir.height != img_th.height) { // Assuming same resolution input preferred? 
        // Or if we resize output to IR size?
        // Let's assume output size = IR size.
    }
    
    int w = img_ir.width;
    int h = img_ir.height;
    
    // Prepare Output
    dst.width = w;
    dst.height = h;
    dst.channels = 3; 
    // Assumes dst.data is externally allocated OR we need to verify.
    // Usually SDK FuseImages allocates output?
    // Let's assume caller provides buffer or we check.
    if (!dst.data) return false;

    // Scanline Loop
    // Reuse TransformPointV2 logic?
    // TransformPointV2 calculates u_th, v_th.
    // We also need u_ir, v_ir (Distorted IR).
    
    float dist = p.assumed_distance_mm;
    
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            
            // 1. Calculate Source Coordinates
            
            // A. Rectified IR -> Raw IR (Distorted)
            // Normalized
            float x_norm = (x - p.K_ir.cx) / p.K_ir.fx;
            float y_norm = (y - p.K_ir.cy) / p.K_ir.fy;
            
            float P_ir_x = x_norm * dist;
            float P_ir_y = y_norm * dist;
            float P_ir_z = dist;
            
            float u_ir, v_ir;
            project_point_v2(P_ir_x, P_ir_y, P_ir_z, p.K_ir, p.D_ir, u_ir, v_ir);
            
            // B. Rectified IR -> Raw Thermal
            float u_th, v_th;
            TransformPointV2((float)x, (float)y, p, u_th, v_th);
            
            // 2. Sample
            float rgb_ir[3], rgb_th[3];
            sample_bilinear_v2(img_ir, u_ir, v_ir, rgb_ir);
            sample_bilinear_v2(img_th, u_th, v_th, rgb_th);
            
            // 3. Blend (50/50)
            int idx = (y * w + x) * 3;
            for(int k=0; k<3; ++k) {
                float val = rgb_ir[k] * 0.5f + rgb_th[k] * 0.5f;
                dst.data[idx + k] = (uint8_t)std::min(std::max(val, 0.0f), 255.0f);
            }
        }
    }
    return true;
}


// --------------------------------------------------------
// Matrix Helpers
// --------------------------------------------------------
ImageFusion::Matrix3x3 ImageFusion::Multiply(const Matrix3x3& A, const Matrix3x3& B) {
    Matrix3x3 C;
    for(int i=0; i<3; ++i) {
        for(int j=0; j<3; ++j) {
            float sum = 0;
            for(int k=0; k<3; ++k) sum += A.data[i*3+k] * B.data[k*3+j];
            C.data[i*3+j] = sum;
        }
    }
    return C;
}

ImageFusion::Matrix3x3 ImageFusion::Invert(const Matrix3x3& A) {
    float det = A.data[0] * (A.data[4] * A.data[8] - A.data[7] * A.data[5]) -
                A.data[1] * (A.data[3] * A.data[8] - A.data[6] * A.data[5]) +
                A.data[2] * (A.data[3] * A.data[7] - A.data[6] * A.data[4]);
                
    if (std::abs(det) < 1e-9) return Matrix3x3::Identity();
    float invDet = 1.0f / det;
    
    Matrix3x3 R;
    R.data[0] = (A.data[4] * A.data[8] - A.data[5] * A.data[7]) * invDet;
    R.data[1] = (A.data[2] * A.data[7] - A.data[1] * A.data[8]) * invDet;
    R.data[2] = (A.data[1] * A.data[5] - A.data[2] * A.data[4]) * invDet;
    R.data[3] = (A.data[5] * A.data[6] - A.data[3] * A.data[8]) * invDet;
    R.data[4] = (A.data[0] * A.data[8] - A.data[2] * A.data[6]) * invDet;
    R.data[5] = (A.data[2] * A.data[3] - A.data[0] * A.data[5]) * invDet;
    R.data[6] = (A.data[3] * A.data[7] - A.data[4] * A.data[6]) * invDet;
    R.data[7] = (A.data[1] * A.data[6] - A.data[0] * A.data[7]) * invDet;
    R.data[8] = (A.data[0] * A.data[4] - A.data[1] * A.data[3]) * invDet;
    return R;
}

ImageFusion::Matrix3x3 ImageFusion::ComputeHomographyFromParams(
    const CameraIntrinsics& camA, const CameraExtrinsics& extA,
    const CameraIntrinsics& camB, const CameraExtrinsics& extB) 
{
    // 1. Construct H_W_to_Cam (3x3 Matrix mapping World (X,Y,1) to Image Homogeneous (u,v,w))
    // P_cam = R * P_world + T
    // World is Z=0. So P_world = [X, Y, 0]^T
    // P_cam = r1*X + r2*Y + T
    //       = [r1 r2 T] * [X, Y, 1]^T
    // P_pix = K * P_cam = K * [r1 r2 T] * [X, Y, 1]^T
    // So H_W_to_Cam = K * [r1 r2 T]
    
    auto getHWorldToCam = [](const CameraIntrinsics& K, const CameraExtrinsics& E) -> Matrix3x3 {
        // K Matrix
        // fx  0 cx
        //  0 fy cy
        //  0  0  1
        float kmat[9] = {K.fx, 0, K.cx, 0, K.fy, K.cy, 0, 0, 1};
        
        // RT Matrix (3x3 part: r1, r2, t)
        // r1 = column 0 of R
        // r2 = column 1 of R
        // t  = T vector
        float rt[9] = {
            E.rotation[0], E.rotation[1], E.translation[0], 
            E.rotation[3], E.rotation[4], E.translation[1],
            E.rotation[6], E.rotation[7], E.translation[2]
        };
        
        // Multiply K * RT
        Matrix3x3 M;
        for(int i=0; i<3; ++i) {
            for(int j=0; j<3; ++j) {
                float sum = 0;
                for(int k=0; k<3; ++k) sum += kmat[i*3+k] * rt[k*3+j];
                M.data[i*3+j] = sum;
            }
        }
        return M;
    };

    Matrix3x3 H_WA = getHWorldToCam(camA, extA);
    Matrix3x3 H_WB = getHWorldToCam(camB, extB);

    // We want Mapping from B to A (for warping into B coordinate space, we need A source pixel for B dst pixel)
    // To scan B pixels (dst) and find A pixels (src):
    // p_A = H * p_B
    // p_B -> P_W -> p_A
    // P_W = H_WB_inv * p_B
    // p_A = H_WA * P_W = H_WA * H_WB_inv * p_B
    
    Matrix3x3 H_WB_inv = Invert(H_WB);
    return Multiply(H_WA, H_WB_inv);
}

// --------------------------------------------------------
// Homography Solver (Gaussian Elimination)
// --------------------------------------------------------
static bool Solve8x8(double A[8][8], double B[8], double X[8]) {
    int n = 8;
    // Pivot
    for (int i = 0; i < n; i++) {
        // Find pivot
        double maxEl = std::abs(A[i][i]);
        int maxRow = i;
        for (int k = i + 1; k < n; k++) {
            if (std::abs(A[k][i]) > maxEl) {
                maxEl = std::abs(A[k][i]);
                maxRow = k;
            }
        }

        // Swap
        for (int k = i; k < n; k++) std::swap(A[maxRow][k], A[i][k]);
        std::swap(B[maxRow], B[i]);

        if (std::abs(A[i][i]) < 1e-9) return false; // Singular

        // Eliminate
        for (int k = i + 1; k < n; k++) {
            double c = -A[k][i] / A[i][i];
            for (int j = i; j < n; j++) {
                if (i == j) A[k][j] = 0;
                else A[k][j] += c * A[i][j];
            }
            B[k] += c * B[i];
        }
    }

    // Back substitution
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        for (int j = i + 1; j < n; j++) {
            sum += A[i][j] * X[j];
        }
        X[i] = (B[i] - sum) / A[i][i];
    }
    return true;
}

ImageFusion::Matrix3x3 ImageFusion::ComputeHomography(const std::vector<Point2f>& src, const std::vector<Point2f>& dst) {
    if (src.size() < 4 || dst.size() < 4) return Matrix3x3::Identity();

    double A[8][8] = {0};
    double B[8] = {0};

    // Use first 4 points
    for (int i = 0; i < 4; i++) {
        double u = src[i].x;
        double v = src[i].y;
        double x = dst[i].x;
        double y = dst[i].y;

        // Equation 1: x = (h0*u + h1*v + h2) / (h6*u + h7*v + 1)
        // u*h0 + v*h1 + h2 - u*x*h6 - v*x*h7 = x
        A[2*i][0] = u;
        A[2*i][1] = v;
        A[2*i][2] = 1;
        A[2*i][3] = 0;
        A[2*i][4] = 0;
        A[2*i][5] = 0;
        A[2*i][6] = -u * x;
        A[2*i][7] = -v * x;
        B[2*i] = x;

        // Equation 2: y = (h3*u + h4*v + h5) / (h6*u + h7*v + 1)
        // u*h3 + v*h4 + h5 - u*y*h6 - v*y*h7 = y
        A[2*i+1][0] = 0;
        A[2*i+1][1] = 0;
        A[2*i+1][2] = 0;
        A[2*i+1][3] = u;
        A[2*i+1][4] = v;
        A[2*i+1][5] = 1;
        A[2*i+1][6] = -u * y;
        A[2*i+1][7] = -v * y;
        B[2*i+1] = y;
    }

    double h[8];
    if (!Solve8x8(A, B, h)) {
        std::cerr << "[Fusion] Failed to solve Homography (singular)." << std::endl;
        return Matrix3x3::Identity();
    }

    Matrix3x3 H;
    H.data[0] = (float)h[0]; H.data[1] = (float)h[1]; H.data[2] = (float)h[2];
    H.data[3] = (float)h[3]; H.data[4] = (float)h[4]; H.data[5] = (float)h[5];
    H.data[6] = (float)h[6]; H.data[7] = (float)h[7]; H.data[8] = 1.0f;
    return H;
}

void ImageFusion::BuildWarpLUT(const Matrix3x3& H_inv, int w, int h) {
    lutWidth = w;
    lutHeight = h;
    tableX.resize(w * h);
    tableY.resize(w * h);

    const float* m = H_inv.data;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            // Apply H_inv * [x, y, 1]
            // u = (h0*x + h1*y + h2) / (h6*x + h7*y + h8)
            // v = (h3*x + h4*y + h5) / (h6*x + h7*y + h8)
            
            float denom = m[6] * x + m[7] * y + m[8];
            if (std::abs(denom) < 1e-6) denom = 1e-6f;

            float u = (m[0] * x + m[1] * y + m[2]) / denom;
            float v = (m[3] * x + m[4] * y + m[5]) / denom;

            tableX[y * w + x] = u;
            tableY[y * w + x] = v;
        }
    }
}

bool ImageFusion::Warp(const Image& src, Image& dst) {
    if (dst.width != lutWidth || dst.height != lutHeight) {
        std::cerr << "[Fusion] Dst image size does not match LUT!" << std::endl;
        return false;
    }

    int dstW = dst.width;
    int dstH = dst.height;
    int srcW = src.width;
    int srcH = src.height;
    int channels = src.channels;

    uint8_t* dstData = dst.data;
    const uint8_t* srcData = src.data;

    for (int y = 0; y < dstH; y++) {
        for (int x = 0; x < dstW; x++) {
            int idx = y * dstW + x;
            float u = tableX[idx];
            float v = tableY[idx];

            // Bilinear Interpolation
            int u0 = (int)std::floor(u);
            int v0 = (int)std::floor(v);
            int u1 = u0 + 1;
            int v1 = v0 + 1;

            float alpha = u - u0;
            float beta = v - v0;
            float w00 = (1 - alpha) * (1 - beta);
            float w10 = alpha * (1 - beta);
            float w01 = (1 - alpha) * beta;
            float w11 = alpha * beta;

            for (int c = 0; c < channels; c++) {
                float val = 0;
                // Boundary check (clamp edge)
                auto getVal = [&](int px, int py) -> float {
                    if (px < 0) px = 0;
                    if (px >= srcW) px = srcW - 1;
                    if (py < 0) py = 0;
                    if (py >= srcH) py = srcH - 1;
                    return (float)srcData[(py * srcW + px) * channels + c];
                };

                val += w00 * getVal(u0, v0);
                val += w10 * getVal(u1, v0);
                val += w01 * getVal(u0, v1);
                val += w11 * getVal(u1, v1);

                dstData[idx * channels + c] = (uint8_t)std::min(255.0f, std::max(0.0f, val));
            }
        }
    }
    return true;
}

}
