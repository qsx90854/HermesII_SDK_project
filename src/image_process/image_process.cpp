#include "image_process.h"
#ifndef DISABLE_NPU
#include <fhhcp/cv.h>
#endif
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>

#include <mutex>
#include <vector>
#include <chrono>

namespace VisionSDK {

std::mutex g_cv_resize_mutex;
bool g_cv_sys_initialized = false;

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
        if (ret == 0) {
            int flush_ret = FH_SYS_VmmFlushCache64(mem->phyAddr, (void *)mem->virAddr, size);
            if (flush_ret != 0) {
                printf("[MMZ] Alloc (ImageProcess) Error: flush cache failed, ret=%d\n", flush_ret);
                FH_SYS_VmmFreeOne64(mem->phyAddr);
                mem->phyAddr = 0;
                mem->virAddr = 0;
                mem->size = 0;
                return -3;
            }
        }
    }
    if (ret != 0) return ret;

    if (mem->phyAddr == 0 || mem->virAddr == 0) {
        printf("[MMZ] Alloc (ImageProcess) Error: invalid address after alloc (phyAddr=0x%llx virAddr=0x%llx)\n",
               (unsigned long long)mem->phyAddr, (unsigned long long)mem->virAddr);
        mem->size = 0;
        return -2;
    }

    mem->size = size;
    printf("[MMZ] Alloc (ImageProcess) Size: %u bytes, PhyAddr: 0x%llx, VirAddr: 0x%llx\n", size, (unsigned long long)mem->phyAddr, (unsigned long long)mem->virAddr);
    return 0;
}

static int free_mmz_memory(T_TY_Mem *mem) {
    if (mem && mem->phyAddr) {
        printf("[MMZ] Free (ImageProcess) Size: %u bytes, PhyAddr: 0x%llx, VirAddr: 0x%llx\n", mem->size, (unsigned long long)mem->phyAddr, (unsigned long long)mem->virAddr);
        int ret = FH_SYS_VmmFreeOne64(mem->phyAddr);
        mem->phyAddr = 0;
        mem->virAddr = 0;
        mem->size = 0;
        return ret;
    }
    return 0;
}

static int flush_mmz_memory(T_TY_Mem *mem) {
    return FH_SYS_VmmFlushCache64(mem->phyAddr, (void *)mem->virAddr, mem->size);
}

static void save_bmp_gray(const char* filename, const uint8_t* data, int width, int height) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;

    int row_size = (width + 3) & ~3; // 4 bytes alignment
    int data_size = row_size * height;
    int palette_size = 256 * 4;
    int file_size = 14 + 40 + palette_size + data_size;

    uint8_t file_header[14] = {
        'B', 'M',
        (uint8_t)(file_size & 0xFF),
        (uint8_t)((file_size >> 8) & 0xFF),
        (uint8_t)((file_size >> 16) & 0xFF),
        (uint8_t)((file_size >> 24) & 0xFF),
        0, 0, 0, 0,
        (uint8_t)((14 + 40 + palette_size) & 0xFF),
        (uint8_t)(((14 + 40 + palette_size) >> 8) & 0xFF),
        (uint8_t)(((14 + 40 + palette_size) >> 16) & 0xFF),
        (uint8_t)(((14 + 40 + palette_size) >> 24) & 0xFF)
    };

    uint8_t info_header[40] = {
        40, 0, 0, 0,
        (uint8_t)(width & 0xFF),
        (uint8_t)((width >> 8) & 0xFF),
        (uint8_t)((width >> 16) & 0xFF),
        (uint8_t)((width >> 24) & 0xFF),
        (uint8_t)(height & 0xFF),
        (uint8_t)((height >> 8) & 0xFF),
        (uint8_t)((height >> 16) & 0xFF),
        (uint8_t)((height >> 24) & 0xFF),
        1, 0,
        8, 0, // 8 bits per pixel
        0, 0, 0, 0,
        (uint8_t)(data_size & 0xFF),
        (uint8_t)((data_size >> 8) & 0xFF),
        (uint8_t)((data_size >> 16) & 0xFF),
        (uint8_t)((data_size >> 24) & 0xFF),
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    };

    fwrite(file_header, 1, 14, f);
    fwrite(info_header, 1, 40, f);

    for (int i = 0; i < 256; i++) {
        uint8_t rgba[4] = { (uint8_t)i, (uint8_t)i, (uint8_t)i, 0 };
        fwrite(rgba, 1, 4, f);
    }

    uint8_t* padding = (uint8_t*)calloc(4, 1);
    for (int y = height - 1; y >= 0; y--) {
        fwrite(data + y * width, 1, width, f);
        if (row_size > width) {
            fwrite(padding, 1, row_size - width, f);
        }
    }
    free(padding);
    fclose(f);
}
#endif

class ImageProcess::Impl {
public:
    // Pure-CPU resize path: used by both the NPU and DISABLE_NPU builds,
    // so it must live outside the #ifndef DISABLE_NPU block below.
    std::vector<unsigned char> resize2_dst_buf;

    bool PrepareDstMemory(Image& dst, int dst_w, int dst_h) {
        int dst_size = dst_w * dst_h; 
        if (resize2_dst_buf.size() < dst_size) {
            resize2_dst_buf.resize(dst_size);
        }
        dst.data = resize2_dst_buf.data();
        return true;
    }

    bool Resize2(const Image& src, Image& dst, int dst_w, int dst_h, bool vflip, int crop_x, int crop_y, int crop_w, int crop_h) {
        if (!src.data || !dst.data) return false;

        int real_crop_x = crop_x;
        int real_crop_y = crop_y;
        int real_crop_w = crop_w;
        int real_crop_h = crop_h;

        if (real_crop_w <= 0 || real_crop_h <= 0) {
            real_crop_x = 0;
            real_crop_y = 0;
            real_crop_w = src.width;
            real_crop_h = src.height;
        }

        // Precalculate lookup tables for X and Y mapping to remove float math from inner loop
        std::vector<int> sx_map(dst_w);
        float scale_x = (float)real_crop_w / dst_w;
        for (int dx = 0; dx < dst_w; ++dx) {
            float src_x_f = real_crop_x + (dx + 0.5f) * scale_x - 0.5f;
            int sx = (int)(src_x_f + 0.5f);
            if (sx < 0) sx = 0;
            if (sx >= src.width) sx = src.width - 1;
            sx_map[dx] = sx;
        }

        std::vector<int> sy_map(dst_h);
        float scale_y = (float)real_crop_h / dst_h;
        for (int dy = 0; dy < dst_h; ++dy) {
            int target_dy = vflip ? (dst_h - 1 - dy) : dy;
            float src_y_f = real_crop_y + (target_dy + 0.5f) * scale_y - 0.5f;
            int sy = (int)(src_y_f + 0.5f);
            if (sy < 0) sy = 0;
            if (sy >= src.height) sy = src.height - 1;
            sy_map[dy] = sy;
        }

        // Loop over pixels with optimized fixed-point conversion
        for (int dy = 0; dy < dst_h; ++dy) {
            int sy = sy_map[dy];
            int sy_offset = sy * src.width;
            uint8_t* dst_line = &dst.data[dy * dst_w];

            if (src.channels == 3) {
                for (int dx = 0; dx < dst_w; ++dx) {
                    int sx = sx_map[dx];
                    int src_idx = (sy_offset + sx) * 3;
                    uint8_t r = src.data[src_idx];
                    uint8_t g = src.data[src_idx + 1];
                    uint8_t b = src.data[src_idx + 2];
                    // Fixed-point gray conversion: (R*77 + G*150 + B*29) >> 8
                    dst_line[dx] = (uint8_t)((r * 77 + g * 150 + b * 29) >> 8);
                }
            } else {
                for (int dx = 0; dx < dst_w; ++dx) {
                    int sx = sx_map[dx];
                    dst_line[dx] = src.data[sy_offset + sx];
                }
            }
        }

        // Set output Image parameters
        dst.width = dst_w;
        dst.height = dst_h;
        dst.channels = 1;

        return true;
    }

#ifndef DISABLE_NPU
    T_TY_Mem src_mem;
    T_TY_Mem dst_mem;
    bool initialized = false;

    int last_src_w = 0;
    int last_src_h = 0;
    int last_dst_w = 0;
    int last_dst_h = 0;

    Impl() {
        memset(&src_mem, 0, sizeof(src_mem));
        memset(&dst_mem, 0, sizeof(dst_mem));
    }

    ~Impl() {
        free_mmz_memory(&src_mem);
        free_mmz_memory(&dst_mem);
    }

    bool Resize(const Image& src, Image& dst, int dst_w, int dst_h, bool vflip, int crop_x = 0, int crop_y = 0, int crop_w = 0, int crop_h = 0) {
        std::cout<<"BIG ERROR, GO OLD RESIZE"<<std::endl;
        if (&src == nullptr || &dst == nullptr) {
            std::cout << "[ImageProcess] Error: src or dst is null reference!" << std::endl;
            return false;
        }

        // std::cout << "[ImageProcess] Resize Enter. src: " << src.width << "x" << src.height 
        //           << ", dst_w: " << dst_w << ", dst_h: " << dst_h << ", vflip: " << vflip 
        //           << ", src_mem virAddr: " << (void*)src_mem.virAddr << " phyAddr: 0x" << std::hex << src_mem.phyAddr 
        //           << ", dst_mem virAddr: " << (void*)dst_mem.virAddr << " phyAddr: 0x" << dst_mem.phyAddr << std::dec << std::endl;

        struct ExitPrinter {
            const T_TY_Mem& src;
            const T_TY_Mem& dst;
            ExitPrinter(const T_TY_Mem& s, const T_TY_Mem& d) : src(s), dst(d) {}
            ~ExitPrinter() {
                //std::cout << "[ImageProcess] Resize Exit. src_mem virAddr: " << (void*)src.virAddr 
                //          << " phyAddr: 0x" << std::hex << src.phyAddr 
                //          << ", dst_mem virAddr: " << (void*)dst.virAddr 
                //          << " phyAddr: 0x" << dst.phyAddr << std::dec << std::endl;
            }
        } exit_printer(src_mem, dst_mem);

        if (dst_mem.virAddr != 0) {
            FH_UINT64 phyAddr = 0;
            int ret = FH_SYS_VmmGetPaddr_ByVaddr64(&phyAddr, (void*)dst_mem.virAddr);
            if (ret != 0) {
                std::cout << "[ImageProcess] Warning: dst_mem.virAddr (0x" << std::hex << dst_mem.virAddr << std::dec << ") is invalid at start of Resize! Code: " << ret << std::endl;
            }
        }



        if (!src.data) return false;

        extern bool g_cv_sys_initialized;
        if (!g_cv_sys_initialized) {
            int ret = TY_CV_SysInit();
            if (ret == 0)
            {
                 g_cv_sys_initialized = true;
            }
            else if (ret == 0xA01D8002 || ret == 0xe0000004)
            {
                std::cout << "[ImageProcess] Warning: TY_CV_SysInit returned " << ret << ", but it's ok." << std::endl;
                 g_cv_sys_initialized = true;
            } 
            else {
                 std::cout << "[ImageProcess] Warning: TY_CV_SysInit returned " << ret << std::endl;
            }
        }
        initialized = true;

        //std::cout << "[ImageProcess] Resizing " << src.width << "x" << src.height << " (" << src.channels << "ch) to " << dst_w << "x" << dst_h << " Flip:" << vflip << std::endl;

        // 1. Prepare Source Memory (RGB Interleaved -> Planar)
        int src_size = src.width * src.height * 3;
        
        if (src.width != last_src_w || src.height != last_src_h || src_mem.virAddr == 0) {
            free_mmz_memory(&src_mem);
            if (alloc_mmz_memory(&src_mem, src_size, E_TY_MEM_VMM_NO_CACHED) != 0) {
                 std::cout << "[ImageProcess] Failed to alloc src mmz" << std::endl;
                 return false;
            }
            last_src_w = src.width;
            last_src_h = src.height;
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
        // Optimization: Do NOT flip the large source image (640x480) here. 
        // We will flip the resized smaller image (128x128) at the end, which is much faster.
        /*
        if (vflip) {
            // printf("before flip_vertical_planar_rgb\n");
            flip_vertical_planar_rgb((uint8_t*)src_mem.virAddr, src.width, src.height);
        }
        */
        // flush_mmz_memory(&src_mem); // Ensure CPU writes are flushed to DDR for TY_CV CvtResize (No flush needed for E_TY_MEM_VMM_NO_CACHED)

        // printf("after flush_mmz\n");
        // 2. Prepare Dest Memory
        int dst_size = dst_w * dst_h; // Gray = 1 byte per pixel
        if (dst_w != last_dst_w || dst_h != last_dst_h || dst_mem.virAddr == 0) {
            free_mmz_memory(&dst_mem);
            if (alloc_mmz_memory(&dst_mem, dst_size, E_TY_MEM_VMM_NO_CACHED) != 0) {
                 std::cout << "[ImageProcess] Failed to alloc dst mmz" << std::endl;
                 return false;
            }
            last_dst_w = dst_w;
            last_dst_h = dst_h;
        }
        // printf("before src_ty\n");
        // 3. TY CV Resize
        T_TY_Image src_ty;
        T_TY_Image dst_ty;
        memset(&src_ty, 0, sizeof(T_TY_Image));
        memset(&dst_ty, 0, sizeof(T_TY_Image));
        
        src_ty.mem = src_mem;
        src_ty.desc.picFormat = E_TY_PIXEL_FORMAT_RGB_888_PLANAR;
        src_ty.desc.picWidth = src.width;
        src_ty.desc.picHeight = src.height;
        src_ty.desc.picWidthStride = src.width;
        src_ty.desc.picHeightStride = src.height;
        if (crop_w > 0 && crop_h > 0) {
            src_ty.desc.roi.x = crop_x;
            src_ty.desc.roi.y = crop_y; // Since the source image is not flipped, the ROI y coordinate is directly crop_y.
            src_ty.desc.roi.width = crop_w;
            src_ty.desc.roi.height = crop_h;
        } else {
            src_ty.desc.roi.x = 0;
            src_ty.desc.roi.y = 0;
            src_ty.desc.roi.width = src.width;
            src_ty.desc.roi.height = src.height;
        }
        
        
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
        int ret = 0;
        {
            std::lock_guard<std::mutex> lock(g_cv_resize_mutex);
            ret = TY_CV_CvtResize(&src_ty, &dst_ty, 1, NULL, 1); // disable for debug
        }
        if (ret != 0) {
             std::cout << "[ImageProcess] TY_CV_CvtResize failed: " << ret << std::endl;
             return false;
        }
        
        // V-Flip the resized smaller gray output image (128x128) if requested.
        if (vflip) {
            flip_vertical_gray((uint8_t*)dst_mem.virAddr, dst_w, dst_h);
        }
        

        
        // flush_mmz_memory(&dst_mem); // No flush needed for E_TY_MEM_VMM_NO_CACHED
        
        {
            //static int bmp_counter = 1;
            //char filename[64];
            //snprintf(filename, sizeof(filename), "%d.bmp", bmp_counter);
            //save_bmp_gray(filename, (const uint8_t*)dst_mem.virAddr, dst_w, dst_h);
            //bmp_counter = bmp_counter % 5 + 1;
        }
        
        // 4. Set Output
        dst.width = dst_w;
        dst.height = dst_h;
        dst.channels = 1;
        dst.data = (unsigned char*)dst_mem.virAddr; 
        
        if (dst_mem.virAddr != 0) {
            FH_UINT64 phyAddr = 0;
            int ret = FH_SYS_VmmGetPaddr_ByVaddr64(&phyAddr, (void*)dst_mem.virAddr);
            if (ret != 0) {
                std::cout << "[ImageProcess] Warning: dst_mem.virAddr (0x" << std::hex << dst_mem.virAddr << std::dec << ") is invalid at end of Resize! Code: " << ret << std::endl;
            }
        }

        //std::cout << "[ImageProcess] Resize success. Dst width=" << dst.width << " height=" << dst.height << std::endl;

        return true;
    }
#else
    // Mock Implementation
    Impl() {}
    ~Impl() {}
    bool Resize(const Image& src, Image& dst, int dst_w, int dst_h, bool vflip, int crop_x = 0, int crop_y = 0, int crop_w = 0, int crop_h = 0) {
        printf("[ImageProcess] MOCK: Resize called. NPU Disabled.\n");

        return true; 
    }
#endif
};

ImageProcess::ImageProcess() : pImpl(std::make_shared<Impl>()) {}
ImageProcess::~ImageProcess() {}

bool ImageProcess::Resize(const Image& src, Image& dst, int dst_w, int dst_h, bool vflip, int crop_x, int crop_y, int crop_w, int crop_h) {
    auto t0 = std::chrono::high_resolution_clock::now();
    if (!pImpl->PrepareDstMemory(dst, dst_w, dst_h)) {
        return false;
    }
    bool ret = pImpl->Resize2(src, dst, dst_w, dst_h, vflip, crop_x, crop_y, crop_w, crop_h);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    //std::cout << "[ImageProcess] Resize time: " << duration << " us" << std::endl;
    return ret;
}

} // namespace VisionSDK
