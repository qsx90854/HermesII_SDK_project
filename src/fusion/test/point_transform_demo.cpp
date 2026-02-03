#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>

// --- 基礎數學結構 ---
struct Vec3 { float x, y, z; };
struct Mat3x3 { float m[3][3]; }; // Row-major
struct CameraMatrix { float fx, fy, cx, cy; };
struct DistCoeffs { float k1, k2, p1, p2, k3; };

// --- 數學工具函式 ---

// 3D 點投影 (相當於 cv2.projectPoints 的核心)
void project_point(float X, float Y, float Z, const CameraMatrix& K, const DistCoeffs& D, float& u, float& v) {
    float x = X / Z;
    float y = Y / Z;
    
    float r2 = x*x + y*y;
    float r4 = r2*r2;
    float r6 = r2*r4;
    
    // 徑向畸變
    float radial = 1.0f + D.k1*r2 + D.k2*r4 + D.k3*r6;
    
    // 切向畸變
    float x_d = x * radial + 2.0f*D.p1*x*y + D.p2*(r2 + 2.0f*x*x);
    float y_d = y * radial + D.p1*(r2 + 2.0f*y*y) + 2.0f*D.p2*x*y;
    
    // 投影回像素座標
    u = K.fx * x_d + K.cx;
    v = K.fy * y_d + K.cy;
}

// Transform Point from Rectified IR View (Undistorted) to Raw Thermal View (Distorted)
void transform_ir_to_thermal(
    float x_ir, float y_ir,
    const CameraMatrix& new_K_ir, float dist_meters,
    const CameraMatrix& K_th, const DistCoeffs& D_th,
    const Mat3x3& R, const Vec3& T,
    float& u_th, float& v_th
) {
    // 1. Normalized Ideal Coordinates (Target View)
    float x_norm = (x_ir - new_K_ir.cx) / new_K_ir.fx;
    float y_norm = (y_ir - new_K_ir.cy) / new_K_ir.fy;

    // 2. 3D Point in Rectified IR View (at assumed distance)
    float P_ir_x = x_norm * dist_meters;
    float P_ir_y = y_norm * dist_meters;
    float P_ir_z = dist_meters;

    // 3. Transform 3D point from IR Space to Thermal Space
    // P_centered = P_ir - T
    float Pc_x = P_ir_x - T.x;
    float Pc_y = P_ir_y - T.y;
    float Pc_z = P_ir_z - T.z;

    // P_th = (P_ir - T) * R
    float P_th_x = Pc_x * R.m[0][0] + Pc_y * R.m[1][0] + Pc_z * R.m[2][0];
    float P_th_y = Pc_x * R.m[0][1] + Pc_y * R.m[1][1] + Pc_z * R.m[2][1];
    float P_th_z = Pc_x * R.m[0][2] + Pc_y * R.m[1][2] + Pc_z * R.m[2][2];

    // 4. Project to Thermal Image
    project_point(P_th_x, P_th_y, P_th_z, K_th, D_th, u_th, v_th);
}

// Hardcoded Parameters Initialization
void init_hardcoded_params(CameraMatrix& K_th, DistCoeffs& D_th, CameraMatrix& K_ir, DistCoeffs& D_ir, Mat3x3& R, Vec3& T) {
    // Thermal Camera Params
    K_th.fx = 258.32029884179315f; 
    K_th.fy = 257.88064819096905f;
    K_th.cx = 87.64961648850591f;  
    K_th.cy = 127.12974658635059f;
    
    D_th.k1 = -0.27226478496697554f;
    D_th.k2 = 0.030104384408577364f;
    D_th.p1 = 0.002327207502631087f;
    D_th.p2 = -0.0017599189641121363f;
    D_th.k3 = -0.12951497351449237f;

    // IR Camera Params
    K_ir.fx = 1238.8419608138552f;
    K_ir.fy = 1207.3314409965117f;
    K_ir.cx = 993.202792998663f;
    K_ir.cy = 561.5114422758486f;

    D_ir.k1 = -0.38037783018524796f;
    D_ir.k2 = 0.17567569491291468f;
    D_ir.p1 = 0.0001772711227356763f;
    D_ir.p2 = 0.0006384138644647992f;
    D_ir.k3 = -0.04562240573034239f;

    // Stereo Extrinsics
    // R (Row-major)
    R.m[0][0] = 0.9995273141115857f; R.m[0][1] = -0.0079753125405994f; R.m[0][2] = -0.029690785351007064f;
    R.m[1][0] = 0.008358292189878123f; R.m[1][1] = 0.999883177808184f; R.m[1][2] = 0.012797253137928719f;
    R.m[2][0] = 0.029585254714949463f; R.m[2][1] = -0.0130393683162706f; R.m[2][2] = 0.999477207132491f;

    // T
    T.x = 13.839705941498458f;
    T.y = 24.522679079337344f;
    T.z = 3.562802134657886f;
}

int main() {
    // 1. Initialize Parameters
    CameraMatrix K_th, K_ir;
    DistCoeffs D_th, D_ir;
    Mat3x3 R; Vec3 T;
    init_hardcoded_params(K_th, D_th, K_ir, D_ir, R, T);
    
    // We treat the "Rectified IR" view as our target view.
    // Assuming same K as IR for the rectified view (as per original logic)
    CameraMatrix new_K_ir = K_ir;
    float assumed_dist = 1000.0f; // mm

    // 2. Define Input Points (Rectified IR coordinates)
    // Example: Two points (x1, y1) and (x2, y2)
    float ir_pts[2][2] = {
        {910.0f, 490.0f}, // Point 1
        {1010.0f, 590.0f} // Point 2
    };

    std::cout << "--- Coordinate Transform Demo (No Map Construction) ---\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 2; ++i) {
        float ir_x = ir_pts[i][0];
        float ir_y = ir_pts[i][1];
        float th_x, th_y;

        // Perform single point transformation
        transform_ir_to_thermal(ir_x, ir_y, new_K_ir, assumed_dist, K_th, D_th, R, T, th_x, th_y);

        std::cout << "Point " << i+1 << ":\n";
        std::cout << "  IR (Rectified): (" << ir_x << ", " << ir_y << ")\n";
        std::cout << "  Thermal (Raw) : (" << th_x << ", " << th_y << ")\n";
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end_time - start_time;
    std::cout << "Transformation time (2 points): " << elapsed.count() << " us" << std::endl;
    std::cout << "Average time per point: " << elapsed.count() / 2.0 << " us" << std::endl;

    return 0;
}
