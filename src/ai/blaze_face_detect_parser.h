#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// BlazeFace Model Configuration (Short Range)
#define FACE_NUM_CLASSES 1 // Face or not
#define FACE_NUM_ANCHORS 896
#define FACE_CONF_THRESHOLD 0.5f  // Confidence threshold
#define FACE_IOU_THRESHOLD 0.3f   // NMS IoU threshold
#define FACE_MAX_OBJS 100         // Max output detections

/**
 * @brief BlazeFace detection result structure
 */
typedef struct {
    float x1, y1, x2, y2;  // Bounding box (pixels)
    float score;           // Confidence score [0, 1]
    
    // 6 Keypoints (x, y)
    // 0: Right Eye, 1: Left Eye, 2: Nose Tip, 3: Mouth Center, 4: Right Ear, 5: Left Ear
    struct {
        float x;
        float y;
    } keypoints[6];
    
} FaceDetection;

/**
 * @brief Parse BlazeFace model output
 * 
 * @param regressors Pointer to regression output [1, 896, 16] (16 floats per anchor)
 *                   Format per anchor: [dy, dx, dh, dw, k1y, k1x, k2y, k2x, ... k6y, k6x] (Note: often y,x order in Mediapipe)
 * @param classifiers Pointer to classification output [1, 896, 1] (1 float per anchor)
 * @param output Output buffer for detection results
 * @param output_num Pointer to store number of detections
 * @param input_width Input image width used for normalization (typically 128)
 * @param input_height Input image height used for normalization (typically 128)
 * @return int 0 on success, < 0 on error
 */
int blaze_face_parse(float *regressors, float *classifiers, 
                     FaceDetection *output, int *output_num, 
                     int input_width, int input_height);

#ifdef __cplusplus
}
#endif
