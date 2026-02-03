#ifndef IMAGE_FUSION_H
#define IMAGE_FUSION_H

#include "HermesII_sdk.h"

namespace VisionSDK {

class ImageFusion {
public:
    bool Fuse(const Image& img1, const Image& img2, Image& output);

    // --- Fusion V2 ---
    // Note: We use HermesII_sdk's FusionParams but need to forward declare or include it.
    // Ideally ImageFusion should be independent, but for simplicity we'll assume access to the struct definitions from HermesII_sdk.h
    // Since HermesII_sdk.h includes this file? No, usually cpp includes both or sdk includes this.
    // Let's check include order. HermesII_sdk.h includes "HermesII_sdk.h" (itself? no) -> "HermesII_sdk.h" lines 4 includes "HermesII_sdk.h" ? Wait.
    // ImageFusion.h included "HermesII_sdk.h" at line 4 (in previous view). Correct.
    
    bool FuseV2(const Image& img_ir, const Image& img_th, const FusionParams& params, Image& dst);
    void TransformPointV2(float x_ir, float y_ir, const FusionParams& params, float& u_th, float& v_th);


    // ==========================================
    // Homography & Warping (No OpenCV)
    // ==========================================
    struct Point2f {
        float x, y;
    };

    struct Matrix3x3 {
        float data[9]; // Row-major: 0,1,2; 3,4,5; 6,7,8
        
        static Matrix3x3 Identity() {
            return {{1,0,0, 0,1,0, 0,0,1}};
        }
    };

    // Solve for H such that dst = H * src
    // Need at least 4 points each.
    static Matrix3x3 ComputeHomography(const std::vector<Point2f>& srcPoints, const std::vector<Point2f>& dstPoints);

    // Matrix Operations
    static Matrix3x3 Multiply(const Matrix3x3& A, const Matrix3x3& B);
    static Matrix3x3 Invert(const Matrix3x3& A);

    // Compute Homography from Camera Parameters (Planar Assumption Z=0)
    // Returns H that maps Image B pixels to Image A pixels
    static Matrix3x3 ComputeHomographyFromParams(
        const CameraIntrinsics& camA, const CameraExtrinsics& extA,
        const CameraIntrinsics& camB, const CameraExtrinsics& extB);

    // Build Lookup Table for warping B -> A
    // H should map A (dst) pixels -> B (src) coordinates (Inverse Mapping)
    // Or if H maps src->dst, we need H_inv.
    // Usually we construct H_inv directly or invert H.
    // We will assume input H is the Transformation Matrix we want to use to FIND source pixels.
    // i.e., src = H * dst.
    void BuildWarpLUT(const Matrix3x3& H_inv, int w, int h);

    // Apply Warping using LUT
    // dst must be allocated
    bool Warp(const Image& src, Image& dst);

private:
    std::vector<float> tableX;
    std::vector<float> tableY;
    int lutWidth = 0;
    int lutHeight = 0;
};

}

#endif
