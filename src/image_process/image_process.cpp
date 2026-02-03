#include "image_process.h"
#ifndef DISABLE_NPU
#include <fhhcp/cv.h>
#endif
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>

namespace VisionSDK {

// Helper: Convert Interleaved to Planar
static void rgb_interleaved_to_planar(const uint8_t* src, uint8_t* dst, int w, int h) {
    int plane_size = w * h;
    for (int i = 0; i < plane_size; i++) {
        dst[i] = src[3 * i];                // R
        dst[plane_size + i] = src[3 * i + 1]; // G
        dst[plane_size * 2 + i] = src[3 * i + 2]; // B
    }
}

// Helper to flip vertical gray
static void flip_vertical_gray(uint8_t *img, int w, int h) {
    uint8_t *row_buf = (uint8_t*)malloc(w);
    if (!row_buf) return;
    for (int y = 0; y < h / 2; y++) {
        uint8_t *row_top = img + y * w;
        uint8_t *row_bot = img + (h - 1 - y) * w;
        // Swap
        memcpy(row_buf, row_top, w);
        memcpy(row_top, row_bot, w);
        memcpy(row_bot, row_buf, w);
    }
    free(row_buf);
}

// Helper to flip RGB Planar vertical
static void flip_vertical_planar_rgb(uint8_t *img, int w, int h) {
    int plane_size = w * h;
    // Flip R
    flip_vertical_gray(img, w, h);
    // Flip G
    flip_vertical_gray(img + plane_size, w, h);
    // Flip B
    flip_vertical_gray(img + plane_size * 2, w, h);
}


#ifndef DISABLE_NPU
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
    if (mem->phyAddr) return FH_SYS_VmmFreeOne64(mem->phyAddr);
    return 0;
}

static int flush_mmz_memory(T_TY_Mem *mem) {
    return FH_SYS_VmmFlushCache64(mem->phyAddr, (void *)mem->virAddr, mem->size);
}
#endif

class ImageProcess::Impl {
public:
#ifndef DISABLE_NPU
    T_TY_Mem src_mem;
    T_TY_Mem dst_mem;
    bool initialized = false;

    Impl() {
        memset(&src_mem, 0, sizeof(src_mem));
        memset(&dst_mem, 0, sizeof(dst_mem));
    }

    ~Impl() {
        free_mmz_memory(&src_mem);
        free_mmz_memory(&dst_mem);
    }
    
    bool Resize(const Image& src, Image& dst, int dst_w, int dst_h, bool vflip) {
        if (!src.data) return false;

        //std::cout << "[ImageProcess] Resizing " << src.width << "x" << src.height << " (" << src.channels << "ch) to " << dst_w << "x" << dst_h << " Flip:" << vflip << std::endl;

        // 1. Prepare Source Memory (RGB Interleaved -> Planar)
        int src_size = src.width * src.height * 3;
        
        if (src_mem.size < src_size) {
            free_mmz_memory(&src_mem);
            if (alloc_mmz_memory(&src_mem, src_size, E_TY_MEM_VMM_CACHED) != 0) {
                 std::cout << "[ImageProcess] Failed to alloc src mmz" << std::endl;
                 return false;
            }
        }
        // printf("before rgb_interleaved_to_planar. src.data=%p, virAddr=%llx, w=%d, h=%d\n", src.data, src_mem.virAddr, src.width, src.height);
        // if (src.data == nullptr) {
        //     printf("[ImageProcess] Error: src.data is NULL!\n");
        //     return false;
        // }
        // if (src_mem.virAddr == 0) {
        //     printf("[ImageProcess] Error: src_mem.virAddr is 0!\n");
        //     return false;
        // }

        // Convert and Copy
        // Convert and Copy
        if (src.channels == 1) {
             // Gray to Planar RGB (Replicate)
             uint8_t* pR = (uint8_t*)src_mem.virAddr;
             uint8_t* pG = pR + src.width * src.height;
             uint8_t* pB = pG + src.width * src.height;
             memcpy(pR, src.data, src.width * src.height);
             memcpy(pG, src.data, src.width * src.height);
             memcpy(pB, src.data, src.width * src.height);
        } else {
            rgb_interleaved_to_planar((const uint8_t*)src.data, (uint8_t*)src_mem.virAddr, src.width, src.height);
        }
        // printf("after rgb_interleaved_to_planar\n");
        // V-Flip if requested (In-place on Planar Buffer)
        if (vflip) {
            // printf("before flip_vertical_planar_rgb\n");
            flip_vertical_planar_rgb((uint8_t*)src_mem.virAddr, src.width, src.height);
        }
        // printf("after flush_mmz\n");
        // 2. Prepare Dest Memory
        int dst_size = dst_w * dst_h; // Gray = 1 byte per pixel
        if (dst_mem.size < dst_size) {
            free_mmz_memory(&dst_mem);
            if (alloc_mmz_memory(&dst_mem, dst_size, E_TY_MEM_VMM_CACHED) != 0) {
                 std::cout << "[ImageProcess] Failed to alloc dst mmz" << std::endl;
                 return false;
            }
        }
        // printf("before src_ty\n");
        // 3. TY CV Resize
        T_TY_Image src_ty;
        src_ty.mem = src_mem;
        src_ty.desc.picFormat = E_TY_PIXEL_FORMAT_RGB_888_PLANAR;
        src_ty.desc.picWidth = src.width;
        src_ty.desc.picHeight = src.height;
        src_ty.desc.picWidthStride = src.width;
        src_ty.desc.picHeightStride = src.height;
        src_ty.desc.roi.x = 0;
        src_ty.desc.roi.y = 0;
        src_ty.desc.roi.width = src.width;
        src_ty.desc.roi.height = src.height;
        
        T_TY_Image dst_ty;
        dst_ty.mem = dst_mem;
        dst_ty.desc.picFormat = E_TY_PIXEL_FORMAT_YUV_400; // Output Gray
        dst_ty.desc.picWidth = dst_w;
        dst_ty.desc.picHeight = dst_h;
        dst_ty.desc.picWidthStride = dst_w;
        dst_ty.desc.picHeightStride = dst_h;
        dst_ty.desc.roi.x = 0;
        dst_ty.desc.roi.y = 0;
        dst_ty.desc.roi.width = dst_w;
        dst_ty.desc.roi.height = dst_h;
        
        // printf("before TY_CV_CvtResize\n");
        int ret = TY_CV_CvtResize(&src_ty, &dst_ty, 1, NULL, 1);
        if (ret != 0) {
             std::cout << "[ImageProcess] TY_CV_CvtResize failed: " << ret << std::endl;
             return false;
        }
        
        flush_mmz_memory(&dst_mem);
        
        // 4. Set Output
        dst.width = dst_w;
        dst.height = dst_h;
        dst.channels = 1;
        dst.data = (unsigned char*)dst_mem.virAddr; 
        
        //std::cout << "[ImageProcess] Resize success. Dst width=" << dst.width << " height=" << dst.height << std::endl;

        return true;
    }
#else
    // Mock Implementation
    Impl() {}
    ~Impl() {}
    bool Resize(const Image& src, Image& dst, int dst_w, int dst_h, bool vflip) {
        printf("[ImageProcess] MOCK: Resize called. NPU Disabled.\n");
        // Minimal stub: Allocate dummy data if needed to prevent crashes, or just return false
        // Caller might expect dst.data to be valid. 
        // Let's allocate on heap? But memory model is "managed" by Impl/NPU usually.
        // Assuming user just wants to run Fall Logic, which doesn't use Resize unless Face is on.
        return true; 
    }
#endif
};

ImageProcess::ImageProcess() : pImpl(std::make_shared<Impl>()) {}
ImageProcess::~ImageProcess() {}

bool ImageProcess::Resize(const Image& src, Image& dst, int dst_w, int dst_h, bool vflip) {
    return pImpl->Resize(src, dst, dst_w, dst_h, vflip);
}

} // namespace VisionSDK
