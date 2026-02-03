#include "blaze_face_detect_parser.h"

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Internal struct for Anchors
typedef struct {
    float y_center;
    float x_center;
    float h;
    float w;
} Anchor;

// Global anchors cache (simple implementation, ideally should be passed in or context-aware)
static Anchor g_anchors[FACE_NUM_ANCHORS];
static int g_anchors_init = 0;

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Generate anchors for BlazeFace (Short Range)
// 128x128 input
// Strides: 8, 16
// 8x8 map (stride 16) -> 2 anchors
// 16x16 map (stride 8) -> 2 anchors
// Total: 16*16*2 + 8*8*6 (Wait, standard BlazeFace is slightly different, let's follow standard impl)
// Standard BlazeFace (Back) might handle different. 
// For Short Range (Front):
// Stride 8: 16x16 grid, 2 anchors per cell -> 512 anchors
// Stride 16: 8x8 grid, 6 anchors per cell -> 384 anchors
// Total: 896 anchors
// Generate anchors for BlazeFace (Short Range)
// 128x128 input
static void generate_anchors(int input_w, int input_h) {
    if (g_anchors_init) return;

    int anchor_idx = 0;
    
    // Layer 1: Stride 8
    int stride1 = 8;
    int grid_rows1 = (input_h + stride1 - 1) / stride1;
    int grid_cols1 = (input_w + stride1 - 1) / stride1;
    
    for (int y = 0; y < grid_rows1; ++y) {
        for (int x = 0; x < grid_cols1; ++x) {
            float y_center = (y + 0.5f) * stride1 / input_h;
            float x_center = (x + 0.5f) * stride1 / input_w;
            
            // 2 anchors per cell, fixed size = stride (common baseline)
            for(int k=0; k<2; k++) {
                g_anchors[anchor_idx].y_center = y_center;
                g_anchors[anchor_idx].x_center = x_center;
                g_anchors[anchor_idx].h = 1.0f; // placeholder, not used if we assume dw is absolute
                g_anchors[anchor_idx].w = 1.0f; 
                // However, for decoding cx/cy we typically use 'anchor scale' which is 1.0 if we assume dx/dy are normalized by image size?
                // NO. Standard is dx/dy are normalized by stride (or anchor size). 
                // Let's set w/h to normalized stride for now to be safe if we change decoding.
                // But simplified MP decoding often uses just image_size normalization. 
                // Let's TRY interpreting regressors as RAW normalized coords first? No user says it's wrong.
                
                // HYPOTHESIS: Model outputs are [x, y, w, h] normalized with respect to anchor size (stride).
                g_anchors[anchor_idx].h = (float)stride1 / input_h;
                g_anchors[anchor_idx].w = (float)stride1 / input_w;

                anchor_idx++;
            }
        }
    }
    
    // Layer 2: Stride 16
    int stride2 = 16;
    int grid_rows2 = (input_h + stride2 - 1) / stride2;
    int grid_cols2 = (input_w + stride2 - 1) / stride2;
    
     for (int y = 0; y < grid_rows2; ++y) {
        for (int x = 0; x < grid_cols2; ++x) {
            float y_center = (y + 0.5f) * stride2 / input_h;
            float x_center = (x + 0.5f) * stride2 / input_w;
            
            for (int k=0; k<6; k++) {
                g_anchors[anchor_idx].y_center = y_center;
                g_anchors[anchor_idx].x_center = x_center;
                g_anchors[anchor_idx].h = (float)stride2 / input_h;
                g_anchors[anchor_idx].w = (float)stride2 / input_w;
                anchor_idx++;
            }
        }
    }
    g_anchors_init = 1;
}

// Reuse IOU from yolov8 parser logic
static float bbox_iou(FaceDetection *box1, FaceDetection *box2) {
    float ix1 = fmaxf(box1->x1, box2->x1);
    float iy1 = fmaxf(box1->y1, box2->y1);
    float ix2 = fminf(box1->x2, box2->x2);
    float iy2 = fminf(box1->y2, box2->y2);

    float iw = fmaxf(0, ix2 - ix1);
    float ih = fmaxf(0, iy2 - iy1);
    float inter = iw * ih;

    float area1 = (box1->x2 - box1->x1) * (box1->y2 - box1->y1);
    float area2 = (box2->x2 - box2->x1) * (box2->y2 - box2->y1);

    return inter / (area1 + area2 - inter);
}

static void obj_qsort_desc(FaceDetection **data, int left, int right) {
    if (left >= right) return;
    int l = left;
    int r = right;
    FaceDetection *mid = data[left];
    while (l < r) {
        while (l < r && data[r]->score <= mid->score) r--;
        data[l] = data[r];
        while (l < r && data[l]->score > mid->score) l++;
        data[r] = data[l];
    }
    data[l] = mid;
    obj_qsort_desc(data, left, l - 1);
    obj_qsort_desc(data, l + 1, right);
}

static int non_max_suppression(FaceDetection *nms_cache, int num_nms, 
                              FaceDetection *results, int *num_results) {
    if (num_nms <= 0) {
        *num_results = 0;
        return 0;
    }

    FaceDetection *plist[FACE_NUM_ANCHORS];
    for (int i = 0; i < num_nms; i++) {
        plist[i] = &nms_cache[i];
    }

    obj_qsort_desc(plist, 0, num_nms - 1);

    *num_results = 0;
    for (int i = 0; i < num_nms; i++) {
        if (plist[i] == NULL) continue;
        FaceDetection *obj1 = plist[i];
        
        for (int j = i + 1; j < num_nms; j++) {
            if (plist[j] == NULL) continue;
            FaceDetection *obj2 = plist[j];
            if (bbox_iou(obj1, obj2) >= FACE_IOU_THRESHOLD) {
                plist[j] = NULL;
            }
        }
        
        if (*num_results < FACE_MAX_OBJS) {
            memcpy(&results[(*num_results)++], obj1, sizeof(FaceDetection));
        } else {
            break;
        }
    }
    return *num_results;
}

int blaze_face_parse(float *regressors, float *classifiers, 
                     FaceDetection *output, int *output_num, 
                     int input_width, int input_height) {
    if (!regressors || !classifiers || !output || !output_num) {
        return -EINVAL;
    }

    // Ensure anchors are generated
    generate_anchors(input_width, input_height);

    FaceDetection nms_cache[FACE_NUM_ANCHORS];
    int num_candidates = 0;

    for (int i = 0; i < FACE_NUM_ANCHORS; i++) {
        // 1. Decode Score
        float raw_score = classifiers[i];
        float score = sigmoid(raw_score); 

        if (score < FACE_CONF_THRESHOLD) {
            continue;
        }

        // 2. Decode Box & Keypoints
        // Regressor shape [896, 16]
        // 2. Decode Box & Keypoints
        // Regressor shape [896, 16]
        // Standard TF/MediaPipe order is usually [y, x, h, w]
        
        int offset = i * 16;
        float dy = regressors[offset + 0]; 
        float dx = regressors[offset + 1]; 
        float dh = regressors[offset + 2];
        float dw = regressors[offset + 3];

        Anchor anchor = g_anchors[i];
        
        // Decoding:
        // cx = anchor.x + dx * anchor.w 
        // cy = anchor.y + dy * anchor.h
        
        // float cx = anchor.x_center + dx * anchor.w;
        // float cy = anchor.y_center + dy * anchor.h;
        // float w = dw * anchor.w; 
        // float h = dh * anchor.h;
        float cx = anchor.x_center *128 + dx ;
        float cy = anchor.y_center *128 + dy ;
        float w = dw ; 
        float h = dh ;
        
        FaceDetection *det = &nms_cache[num_candidates];
        det->score = score;
        // det->x1 = (cx - w * 0.5f) * input_width;
        // det->y1 = (cy - h * 0.5f) * input_height;
        // det->x2 = (cx + w * 0.5f) * input_width;
        // det->y2 = (cy + h * 0.5f) * input_height;
        det->x1 = (cx - w * 0.5f) ;
        det->y1 = (cy - h * 0.5f) ;
        det->x2 = (cx + w * 0.5f) ;
        det->y2 = (cy + h * 0.5f) ;


        if (score > 0.5f) { // Debug print for high confidence
             printf("Debug Parse: Anch[%.2f, %.2f, %.2f] Raw[%.2f, %.2f, %.2f, %.2f] -> Box[%.2f, %.2f, %.2f, %.2f]\n",
                    anchor.x_center, anchor.y_center, anchor.w,
                    dx, dy, dw, dh,
                    det->x1, det->y1, det->x2, det->y2);
        }

        
        
        // Keypoints: 6 pairs, usually [y, x] in TF
        for (int k = 0; k < 6; k++) {
            float kdy = regressors[offset + 4 + k * 2];
            float kdx = regressors[offset + 4 + k * 2 + 1];
            
            float kx = anchor.x_center + kdx * anchor.w;
            float ky = anchor.y_center + kdy * anchor.h;
            
            det->keypoints[k].x = kx * input_width;
            det->keypoints[k].y = ky * input_height;
        }

        num_candidates++;
        if (num_candidates >= FACE_MAX_OBJS) break;
    }

    int ret = non_max_suppression(nms_cache, num_candidates, output, output_num);
    if (ret >= 0) return 0;
    return ret;
}
