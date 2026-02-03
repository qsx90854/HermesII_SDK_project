#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include <vector>
#include <cmath>

namespace VisionSDK {

class KalmanFilter {
public:
    // State: [x, y, vx, vy]
    // Measurement: [x, y]
    
    // F: State Transition Matrix (4x4)
    // H: Measurement Matrix (2x4)
    // Q: Process Noise Covariance (4x4)
    // R: Measurement Noise Covariance (2x2)
    // P: Error Covariance (4x4)
    // K: Kalman Gain (4x2)
    
    float x[4];     // State vector
    float P[4][4];  // Covariance matrix
    
    // Constants
    float dt; 
    float q_pos; // Process noise for position
    float q_vel; // Process noise for velocity
    float r_pos; // Measurement noise

    KalmanFilter(float initial_x, float initial_y) {
        dt = 1.0f;
        q_pos = 1.0f;
        q_vel = 1.0f;
        r_pos = 10.0f;

        // Init State
        x[0] = initial_x;
        x[1] = initial_y;
        x[2] = 0;
        x[3] = 0;

        // Init Covariance (Identity * large value)
        for(int i=0; i<4; ++i)
            for(int j=0; j<4; ++j) P[i][j] = (i==j) ? 1000.0f : 0.0f;
    }

    // Predict state for next step
    void Predict() {
        // x = F * x
        // F = [1 0 dt 0]
        //     [0 1 0 dt]
        //     [0 0 1  0]
        //     [0 0 0  1]
        x[0] += x[2] * dt;
        x[1] += x[3] * dt;
        
        // P = F * P * F^T + Q
        // Simplified update for performance (assuming simple independent Q)
        // This is a naive implementation manually expanded for 4x4
        
        float F[4][4] = {
            {1, 0, dt, 0},
            {0, 1, 0, dt},
            {0, 0, 1, 0},
            {0, 0, 0, 1}
        };
        
        float FP[4][4] = {0};
        for(int i=0; i<4; ++i)
            for(int j=0; j<4; ++j)
                for(int k=0; k<4; ++k)
                    FP[i][j] += F[i][k] * P[k][j];
                    
        float P_new[4][4] = {0};
        for(int i=0; i<4; ++i)
            for(int j=0; j<4; ++j) {
                for(int k=0; k<4; ++k)
                    P_new[i][j] += FP[i][k] * F[j][k]; // F[j][k] is F^T[k][j]
            }
            
        // Add Q
        P_new[0][0] += q_pos; P_new[1][1] += q_pos;
        P_new[2][2] += q_vel; P_new[3][3] += q_vel;
        
        // Copy back
        for(int i=0; i<4; ++i)
            for(int j=0; j<4; ++j) P[i][j] = P_new[i][j];
    }
    
    // Update with measurement [mx, my]
    void Update(float mx, float my) {
        // y = z - H * x (Residual)
        // H = [1 0 0 0]
        //     [0 1 0 0]
        float y_res[2];
        y_res[0] = mx - x[0];
        y_res[1] = my - x[1];
        
        // S = H * P * H^T + R
        // H * P selects top-left 2x4 of P
        // (H * P) * H^T selects top-left 2x2 of P
        float S[2][2];
        S[0][0] = P[0][0] + r_pos;
        S[0][1] = P[0][1];
        S[1][0] = P[1][0];
        S[1][1] = P[1][1] + r_pos;
        
        // K = P * H^T * inv(S)
        // K is 4x2
        // det(S)
        float det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
        float invS[2][2];
        invS[0][0] = S[1][1] / det;
        invS[0][1] = -S[0][1] / det;
        invS[1][0] = -S[1][0] / det;
        invS[1][1] = S[0][0] / det;
        
        float PHt[4][2]; // P * H^T (First 2 cols of P)
        for(int i=0; i<4; ++i) {
            PHt[i][0] = P[i][0];
            PHt[i][1] = P[i][1];
        }
        
        float K[4][2];
        for(int i=0; i<4; ++i) {
            K[i][0] = PHt[i][0] * invS[0][0] + PHt[i][1] * invS[1][0];
            K[i][1] = PHt[i][0] * invS[0][1] + PHt[i][1] * invS[1][1];
        }
        
        // x = x + K * y
        x[0] += K[0][0] * y_res[0] + K[0][1] * y_res[1];
        x[1] += K[1][0] * y_res[0] + K[1][1] * y_res[1];
        x[2] += K[2][0] * y_res[0] + K[2][1] * y_res[1];
        x[3] += K[3][0] * y_res[0] + K[3][1] * y_res[1];
        
        // P = (I - K * H) * P
        // I - KH
        float IKH[4][4];
        for(int i=0; i<4; ++i)
            for(int j=0; j<4; ++j) IKH[i][j] = (i==j ? 1.0f : 0.0f);
            
        // K * H (4x2 * 2x4 -> 4x4)
        // H has 1 at (0,0) and (1,1)
        for(int i=0; i<4; ++i) {
            IKH[i][0] -= K[i][0];
            IKH[i][1] -= K[i][1];
        }
        
        float P_new[4][4] = {0};
        for(int i=0; i<4; ++i)
            for(int j=0; j<4; ++j)
                for(int k=0; k<4; ++k)
                    P_new[i][j] += IKH[i][k] * P[k][j];
                    
        for(int i=0; i<4; ++i)
            for(int j=0; j<4; ++j) P[i][j] = P_new[i][j];
    }
    
    // Get predicted position
    void GetState(float& out_x, float& out_y) const {
        out_x = x[0];
        out_y = x[1];
    }
};

} // VisionSDK

#endif
