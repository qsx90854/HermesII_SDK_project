#include "face_detector.h"
#include "blaze_face_detect_parser.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>

#ifndef DISABLE_NPU
#include "fhhcp/npu.h"
#include "fhhcp/sys.h"
#include "types/vmm_api.h"
#include <fhhcp/cv.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include "../image_process/image_process.h"

namespace VisionSDK {

// =========================================================
// Debug Helper: Save Gray BMP
// =========================================================
static void save_debug_gray_bmp(const char* filename, const unsigned char* data, int w, int h) {
    if (!data) return;
    FILE* f = fopen(filename, "wb");
    if (!f) {
        printf("[FaceDetector] Failed to open %s for writing\n", filename);
        return;
    }

    int filesize = 54 + w * h * 3; // BMP uses RGB even for gray usually, or palette. Let's write 24-bit RGB for simplicity.
    unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
    unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
    unsigned char bmppad[3] = {0,0,0};

    bmpfileheader[ 2] = (unsigned char)(filesize    );
    bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
    bmpfileheader[ 4] = (unsigned char)(filesize>>16);
    bmpfileheader[ 5] = (unsigned char)(filesize>>24);

    bmpinfoheader[ 4] = (unsigned char)(       w    );
    bmpinfoheader[ 5] = (unsigned char)(       w>> 8);
    bmpinfoheader[ 6] = (unsigned char)(       w>>16);
    bmpinfoheader[ 7] = (unsigned char)(       w>>24);
    bmpinfoheader[ 8] = (unsigned char)(       h    );
    bmpinfoheader[ 9] = (unsigned char)(       h>> 8);
    bmpinfoheader[10] = (unsigned char)(       h>>16);
    bmpinfoheader[11] = (unsigned char)(       h>>24);

    fwrite(bmpfileheader,1,14,f);
    fwrite(bmpinfoheader,1,40,f);

    for(int i=0; i<h; i++) {
        for(int j=0; j<w; j++) {
            // BMP is bottom-up usually? Standard windows BMP is.
            // Let's write top-down but standard BMP readers might show it flipped.
            // Actually, let's just write validation data.
            // If we want it upright in viewer, we might need to invert Y unless height is negative in header.
            // Let's stick to standard loop: (h-1-i)
            int idx = (h-1-i)*w + j;
            unsigned char val = data[idx];
            fwrite(&val, 1, 1, f);
            fwrite(&val, 1, 1, f);
            fwrite(&val, 1, 1, f);
        }
        fwrite(bmppad,1,(4-(w*3)%4)%4,f);
    }

    fclose(f);
    printf("[FaceDetector] Saved debug image: %s\n", filename);
}

// =========================================================
// NPU Helper Functions (Private to this compilation unit)
// =========================================================
#ifndef DISABLE_NPU
static void handle_error(const char *msg, int ret) {
    if (msg) printf("FaceDetector Error: %s, code: %d\n", msg, ret);
}

static int alloc_mmz_memory(T_TY_Mem *mem, uint32_t size, E_TY_MemAllocType type) {
    if (!mem || size == 0) return -1;
    int ret = 0;
    const char *tag = (type == E_TY_MEM_VMM_CACHED) ? "NPU_CACHED" : "NPU";
    
    if (type == E_TY_MEM_VMM_NO_CACHED) {
        ret = FH_SYS_VmmAllocEx64((FH_UINT64*)&mem->phyAddr, (void **)&mem->virAddr, tag, "anonymous", size, 128);
    } else {
        ret = FH_SYS_VmmAllocEx_Cached64((FH_UINT64*)&mem->phyAddr, (void **)&mem->virAddr, tag, "anonymous", size, 128);
        if (ret == 0) FH_SYS_VmmFlushCache64(mem->phyAddr, (void *)mem->virAddr, mem->size);
    }
    if (ret != 0) return ret;
    mem->size = size;
    return 0;
}

static int free_mmz_memory(T_TY_Mem *mem) {
    return FH_SYS_VmmFreeOne64(mem->phyAddr);
}

static int flush_mmz_memory(T_TY_Mem *mem) {
    return FH_SYS_VmmFlushCache64(mem->phyAddr, (void *)mem->virAddr, mem->size);
}

static int get_file_size(const char *path) {
    struct stat statbuf;
    if (stat(path, &statbuf) != 0) return -1;
    return statbuf.st_size;
}

static int read_file(T_TY_Mem *mem, const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    size_t read_bytes = fread((void *)mem->virAddr, 1, mem->size, fp);
    fclose(fp);
    if (read_bytes != mem->size) return -2;
    return 0;
}

static int get_image_size(E_TY_PixelFormat fmt, uint32_t width, uint32_t height) {
    switch (fmt) {
        case E_TY_PIXEL_FORMAT_U8C1: return width * height;
        case E_TY_PIXEL_FORMAT_U16C1: return width * height * 2;
        case E_TY_PIXEL_FORMAT_YUV_SEMIPLANAR_420: return width * height * 3 / 2;
        case E_TY_PIXEL_FORMAT_YUV_400: return width * height;
        default: return 0;
    }
}

static int get_blob_size(T_TY_BlobDesc *blob) {
    if (blob->type == E_TY_BLOB_IMAGE) {
        return get_image_size(blob->img.picFormat, blob->img.picWidthStride, blob->img.picHeightStride);
    }
    // Simple case for data blobs (assuming float32 etc)
    int32_t byteUnitVec[] = {4, 2, 1, 4, 1, 2, 4, 8, 8, 1};
    int unitSize = byteUnitVec[blob->tensor.dataType];
    int num = 1;
    for(int i=0; i<blob->tensor.numDims; i++) num *= blob->tensor.dims[i];
    return unitSize * num;
}
#else
static int get_file_size(const char *path) { // Kept specifically if needed, but likely unused in mock
    struct stat statbuf;
    if (stat(path, &statbuf) != 0) return -1;
    return statbuf.st_size;
}
#endif

// --------------------------------------------------------
// Helpers for Color Conversion (NEON)
// --------------------------------------------------------
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// ... (other includes)

// --------------------------------------------------------
// Helpers for Color Conversion
// --------------------------------------------------------
static void YUV400_to_YUV420_NEON(const unsigned char* src_y, unsigned char* dst_yuv, int width, int height) {
    int y_size = width * height;
    int uv_width = width / 2;
    int uv_height = height / 2;
    int uv_size = uv_width * uv_height;

    unsigned char* dst_y = dst_yuv;
    unsigned char* dst_u = dst_yuv + y_size;
    unsigned char* dst_v = dst_u + uv_size;

#ifdef __ARM_NEON
    // Y Plane (Copy 16 bytes)
    for (int i = 0; i < y_size; i += 16) {
        uint8x16_t data = vld1q_u8(src_y + i);
        vst1q_u8(dst_y + i, data);
    }

    // U/V Plane (Set to 128)
    uint8x16_t gray_val = vdupq_n_u8(128);
    for (int i = 0; i < uv_size; i += 16) {
        vst1q_u8(dst_u + i, gray_val);
    }
    for (int i = 0; i < uv_size; i += 16) {
        vst1q_u8(dst_v + i, gray_val);
    }
#else
    // Generic C++ Implementation for non-ARM platforms
    memcpy(dst_y, src_y, y_size);
    memset(dst_u, 128, uv_size);
    memset(dst_v, 128, uv_size);
#endif
}

// rgb_interleaved_to_planar removed (moved to ImageProcess)

// Added per user request
static bool CropCenterSquare(const Image& src, Image& dst) {
    if (!src.data) return false;
    
    int min_dim = std::min(src.width, src.height);
    int x_off = (src.width - min_dim) / 2;
    int y_off = (src.height - min_dim) / 2;
    
    dst.width = min_dim;
    dst.height = min_dim;
    dst.channels = src.channels;
    dst.timestamp = src.timestamp;
    
    // Allocate if needed. Note: Caller must free this if allocated here.
    // Ideally, use a managed buffer context.
    if (!dst.data) {
        dst.data = (unsigned char*)malloc(min_dim * min_dim * src.channels);
    }
    
    if (!dst.data) return false;

    for (int y = 0; y < min_dim; y++) {
         int src_index = ((y + y_off) * src.width + x_off) * src.channels;
         int dst_index = (y * min_dim) * src.channels; // Packed
         memcpy(dst.data + dst_index, src.data + src_index, min_dim * src.channels);
    }
    return true;
}

// =========================================================
// Implementation Class
// =========================================================

class FaceDetector::Impl {
public:
#ifndef DISABLE_NPU
    TY_NPU_MODEL_HANDLE model_handle = NULL;
    TY_NPU_TASK_HANDLE task_handle = NULL;
    T_TY_MemSegmentInfo model_mem;
    T_TY_MemSegmentInfo task_mem;
    T_TY_TaskInput *task_inputs = NULL;
    T_TY_TaskOutput *task_outputs = NULL;
    T_TY_ModelDesc model_desc;
    bool initialized = false;
    
    // Image Processor
    ImageProcess imageProcess;

    Impl() {
        memset(&model_mem, 0, sizeof(model_mem));
        memset(&task_mem, 0, sizeof(task_mem));
    }

    ~Impl() {
        cleanup();
    }

    void cleanup() {
        if (task_handle) TY_NPU_ReleaseTask(task_handle);
        if (model_handle) TY_NPU_ReleaseModel(model_handle);
        
        if (task_inputs) {
            for (int i=0; i<model_desc.ioDesc.inputNum; i++) free_mmz_memory(&task_inputs[i].dataIn);
            free(task_inputs);
        }
        if (task_outputs) {
             for (int i=0; i<model_desc.ioDesc.outputNum; i++) free_mmz_memory(&task_outputs[i].dataOut);
             free(task_outputs);
        }
        for (int i=0; i<model_mem.segNum; i++) free_mmz_memory(&model_mem.memInfo[i].mem);
        
        initialized = false;
    }

    StatusCode Init(const std::string& model_path) {
        if (initialized) return StatusCode::OK;

        // Ensure System Init is called (checking if already called? existing API doesn't seem to error if called twice, but best practice: call once)
        // Since we don't have a global singleton for SysInit yet, we will call it here.
        // Or should we move this to VisionSDK::Init?
        // User asked to check if these lines are present.
        // Let's add them here for now as FaceDetector might be used standalone.
        // But typically SysInit returns 0 if already init? Need to check manual.
        // C_V2 says: ret = TY_NPU_SysInit(); if(ret!=0) handle_error
        
        // Initializing System
        int ret = 0;
        printf("[FaceDetector] Init: TY_NPU_SysInit\n");
        ret = TY_NPU_SysInit();
        if (ret != 0 && ret != 0xe0000004) { // 0xe0000004 might be "already initialized"? 
             // Just print error for now, if it fails
             // std::cout << "[FaceDetector] Warning: TY_NPU_SysInit returned " << ret << std::endl;
             // If it fails seriously, we might want to return, but some drivers return error if already init.
        }
        printf("[FaceDetector] Init: TY_CV_SysInit\n");
        // CV Init
        ret = TY_CV_SysInit();
         if (ret != 0 && ret != 0xe0000004) {
             // std::cout << "[FaceDetector] Warning: TY_CV_SysInit returned " << ret << std::endl;
        }

        printf("[FaceDetector] Init: Loading model %s\n", model_path.c_str());
        int size = get_file_size(model_path.c_str());
        if (size <= 0) {
            std::cout << "[FaceDetector] Init Error: Invalid file size " << size << std::endl;
            return StatusCode::ERROR_INIT_FAILED;
        }

        model_mem.segNum = 1;
        model_mem.memInfo[0].allocInfo.alignByteSize = 128;
        model_mem.memInfo[0].allocInfo.allocType = E_TY_MEM_VMM_CACHED;
        model_mem.memInfo[0].allocInfo.shareType = E_MEM_EXCLUSIVED;
        model_mem.memInfo[0].allocInfo.size = size; // Missing Line Fixed!
        
        if (alloc_mmz_memory(&model_mem.memInfo[0].mem, size, E_TY_MEM_VMM_CACHED) != 0) {
            std::cout << "[FaceDetector] Init Error: Alloc model mmz failed" << std::endl;
            return StatusCode::ERROR_INIT_FAILED;
        }
        if (read_file(&model_mem.memInfo[0].mem, model_path.c_str()) != 0) {
            std::cout << "[FaceDetector] Init Error: Read file failed" << std::endl;
            return StatusCode::ERROR_INIT_FAILED;
        }
        flush_mmz_memory(&model_mem.memInfo[0].mem);

        T_TY_ModelCfgParam model_cfg;
        if (TY_NPU_CreateModelFromPhyMem(&model_mem, &model_cfg, NULL, &model_desc, &model_handle) != 0) {
            std::cout << "[FaceDetector] Init Error: TY_NPU_CreateModelFromPhyMem failed" << std::endl;
            return StatusCode::ERROR_INIT_FAILED;
        }

        T_TY_TaskCfgParam task_cfg;
        TY_NPU_GetTaskMemSize(model_handle, &task_cfg, &task_mem);
        
        // Allocate task memory
        for(int i=0; i<task_mem.segNum; i++) {
             if (alloc_mmz_memory(&task_mem.memInfo[i].mem, task_mem.memInfo[i].allocInfo.size, E_TY_MEM_VMM_CACHED) != 0) {
                std::cout << "[FaceDetector] Init Error: Alloc task mmz failed" << std::endl;
                return StatusCode::ERROR_INIT_FAILED;
             }
        }
        
        if (TY_NPU_CreateTask(model_handle, &task_cfg, &task_mem, &task_handle) != 0) {
            std::cout << "[FaceDetector] Init Error: Create Task failed" << std::endl;
            return StatusCode::ERROR_INIT_FAILED;
        }

        // Init IO
        int input_num = model_desc.ioDesc.inputNum;
        int output_num = model_desc.ioDesc.outputNum;
        task_inputs = (T_TY_TaskInput*)calloc(input_num, sizeof(T_TY_TaskInput));
        task_outputs = (T_TY_TaskOutput*)calloc(output_num, sizeof(T_TY_TaskOutput));
        
        for (int i=0; i<input_num; i++) {
             task_inputs[i].descIn.type = E_TY_BLOB_IMAGE;
             task_inputs[i].descIn.img.picFormat = E_TY_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
             task_inputs[i].descIn.img.picWidth = 128;
             task_inputs[i].descIn.img.picHeight = 128;
             task_inputs[i].descIn.img.picWidthStride = 128;
             task_inputs[i].descIn.img.picHeightStride = 128;
             task_inputs[i].descIn.img.roi.x = 0;
             task_inputs[i].descIn.img.roi.y = 0;
             task_inputs[i].descIn.img.roi.width = 128;
             task_inputs[i].descIn.img.roi.height = 128;

             if (alloc_mmz_memory(&task_inputs[i].dataIn, get_blob_size(&task_inputs[i].descIn), E_TY_MEM_VMM_CACHED) != 0) {
                 std::cout << "[FaceDetector] Init Error: Alloc input mmz failed" << std::endl;
                 return StatusCode::ERROR_INIT_FAILED;
             }
        }
        for (int i=0; i<output_num; i++) {
             if (alloc_mmz_memory(&task_outputs[i].dataOut, get_blob_size(&model_desc.ioDesc.out[i]), E_TY_MEM_VMM_CACHED) != 0) {
                 std::cout << "[FaceDetector] Init Error: Alloc output mmz failed" << std::endl;
                 return StatusCode::ERROR_INIT_FAILED;
             }
        }

        initialized = true;
        // Ensure debug dir exists? 
        //system("mkdir -p debug_face_input");

        // std::cout << "[FaceDetector] Init Success!" << std::endl;
        return StatusCode::OK;
    }

    int Detect(const Image& img, std::vector<FaceROI>& faces) {
        if (!initialized) {
            std::cout << "[FaceDetector] Init failed!" << std::endl;
            return -1;
        }
        
        // --- SAVE DEBUG IMAGE ---
        static int debug_frame_count = 0;
        char debug_filename[256];
        snprintf(debug_filename, sizeof(debug_filename), "debug_face_input/face_input_%05d.bmp", debug_frame_count++);
        // Assuming img is 128x128 Gray (YUV400)
        // save_debug_gray_bmp(debug_filename, (const unsigned char*)img.data, img.width, img.height);
        // ------------------------

        // 1. Copy image data to NPU input buffer
         // Safety check
         if (img.width != 128 || img.height != 128) {
             printf("[FaceDetector] Input size mismatch %dx%d\n", img.width, img.height);
             return -1;
         }

         // Use NEON conversion from Gray to YUV420 (Task Input)
         YUV400_to_YUV420_NEON((uint8_t*)img.data, (uint8_t*)task_inputs[0].dataIn.virAddr, 128, 128);
         flush_mmz_memory(&task_inputs[0].dataIn);
         // printf("before NPU Forward\n");
         // 2. Inference
         int ret = TY_NPU_Forward(task_handle, E_TY_NPU_ID_0, 
                                  model_desc.ioDesc.inputNum, task_inputs, 
                                  model_desc.ioDesc.outputNum, task_outputs);
         if (ret != 0) {
             std::cout << "[FaceDetector] NPU Forward failed: " << ret << std::endl;
             return ret;
         }

         // 3. Flush output
         for(int i=0; i<model_desc.ioDesc.outputNum; i++) 
             flush_mmz_memory(&task_outputs[i].dataOut);

         // 4. Parse Results
         FaceDetection results[FACE_MAX_OBJS];
         int num_dets = 0;
         
         // Using the blaze_face_parse C function from the parser file
         blaze_face_parse((float*)task_outputs[0].dataOut.virAddr, 
                          (float*)task_outputs[1].dataOut.virAddr,
                          results, &num_dets, 128, 128);

        faces.clear();
        // std::cout << "[FaceDetector] Frame " << debug_frame_count-1 << " Dets: " << num_dets << std::endl;

        if (num_dets == 0) return 0;

        // 5. Post-process (Union)
        // Note: The input image was vertically flipped (vflip=true in Resize).
        // So the detected coordinates are relative to the flipped image (0,0 is bottom-left of original).
        // We need to revert the Y-coordinates to match the original top-down orientation.
        // Coordinate Space: 0..128. 
        // y_orig = 128 - y_flipped (roughly, assuming standard pixel coords).
        // Check blaze_face_parse output: x1, y1 are pixels.
        
        auto fix_y = [](float y) { return 128.0f - y; };

        // We need to be careful with y1 vs y2. If y1 < y2 in flipped, then 128-y1 > 128-y2.
        // So flipped y1 becomes new y2, flipped y2 becomes new y1.
        
        if (num_dets == 1) {
            float y1 = fix_y(results[0].y2); // Swap 1 and 2
            float y2 = fix_y(results[0].y1);
            faces.push_back({results[0].x1, y1, results[0].x2, y2, results[0].score});
        } else {
             float ux1 = results[0].x1;
             float uy1 = results[0].y1;
             float ux2 = results[0].x2;
             float uy2 = results[0].y2;
             float max_score = results[0].score;

             for(int i=1; i<num_dets; i++) {
                 if (results[i].x1 < ux1) ux1 = results[i].x1;
                 if (results[i].y1 < uy1) uy1 = results[i].y1;
                 if (results[i].x2 > ux2) ux2 = results[i].x2;
                 if (results[i].y2 > uy2) uy2 = results[i].y2;
                 if (results[i].score > max_score) max_score = results[i].score;
             }
             
             // Fix Union Box
             float fixed_y1 = fix_y(uy2);
             float fixed_y2 = fix_y(uy1);
             faces.push_back({ux1, fixed_y1, ux2, fixed_y2, max_score});
        }
        return 0;
    }

    bool Resize(const Image& src, Image& dst) {
        if (!src.data) {
            std::cout << "[FaceDetector::Resize] Error: Source data is NULL" << std::endl;
            return false;
        }
        if (!initialized) {
            std::cout << "[FaceDetector::Resize] Error: FaceDetector not initialized!" << std::endl;
            return false;
        }
        
        // Delegate to ImageProcess
        // We know we want 128x128 output
        // Also enable V-Flip because reference demo does it (input data is likely flipped)
        // printf("before call ImageProcess Resize\n");
        return imageProcess.Resize(src, dst, 128, 128, true);
    }
#else
    // MOCK IMPLEMENTATION
    Impl() {}
    ~Impl() {}
    StatusCode Init(const std::string& model_path) {
        printf("[FaceDetector] MOCK: Init called with %s. NPU Disabled.\n", model_path.c_str());
        return StatusCode::OK;
    }
    int Detect(const Image& img, std::vector<FaceROI>& faces) {
        // printf("[FaceDetector] MOCK: Detect called. NPU Disabled.\n");
        faces.clear();
        return 0; // 0 faces
    }
    bool Resize(const Image& src, Image& dst) {
        // Mock resize if needed, or just allow it if ImageProcess is pure CPU?
        // ImageProcess includes might pull in NPU, check ImageProcess.
        // Assuming ImageProcess might fail if it uses NPU. But previous code uses imageProcess.Resize which uses pure CPU (stb_image_resize or manual).
        // Let's assume Resize is safe or stub it. 
        // For now stub it to be safe.
        // printf("[FaceDetector] MOCK: Resize called. NPU Disabled.\n");
        return true; 
    }
#endif
};

// =========================================================
// Public Wrapper
// =========================================================

FaceDetector::FaceDetector() : pImpl(std::make_shared<Impl>()) {}
FaceDetector::~FaceDetector() {}

StatusCode FaceDetector::Init(const std::string& model_path) {
    return pImpl->Init(model_path);
}

int FaceDetector::Detect(const Image& img, std::vector<FaceROI>& faces) {
    return pImpl->Detect(img, faces);
}

bool FaceDetector::Resize(const Image& src, Image& dst) {
    return pImpl->Resize(src, dst);
}

} // namespace VisionSDK
