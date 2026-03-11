#ifndef VIDEO_READER_H
#define VIDEO_READER_H

#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <stdexcept>

class VideoReader {
private:
    FILE* pipe;
    int width;
    int height;
    int channels;
    size_t frameSize;

public:
    // 預設建構子，方便後續呼叫 openRtsp 等方法
    VideoReader(int targetWidth = 800, int targetHeight = 450) 
        : pipe(nullptr), width(targetWidth), height(targetHeight), channels(3) {
        frameSize = width * height * channels;
    }

    // 建構子：開啟影片並自動設定 Resize 尺寸 (預設 800x450)
    VideoReader(const std::string& videoPath, int targetWidth = 800, int targetHeight = 450) 
        : pipe(nullptr), width(targetWidth), height(targetHeight), channels(3) {
        
        frameSize = width * height * channels;
        
        std::string cmd;
        if (videoPath.find("rtsp://") == 0) {
            // RTSP 串流自動套用 -rtsp_transport tcp 參數
            cmd = "ffmpeg -rtsp_transport tcp -i \"" + videoPath + "\" -f image2pipe -pix_fmt rgb24 -s " + 
                  std::to_string(width) + "x" + std::to_string(height) + 
                  " -vcodec rawvideo - 2>/dev/null";
        } else {
            // 組裝 FFmpeg 指令
            // -i: 輸入檔案
            // -f image2pipe: 輸出連續影像
            // -pix_fmt rgb24: 確保輸出格式為 RGB (對應 channels=3)
            // -s: 強制 Resize 解析度
            // -vcodec rawvideo: 輸出未壓縮的 raw data
            // 2>/dev/null: 隱藏 FFmpeg 的終端機輸出訊息 (若需 debug 可移除此段)
            cmd = "ffmpeg -i \"" + videoPath + "\" -f image2pipe -pix_fmt rgb24 -s " + 
                  std::to_string(width) + "x" + std::to_string(height) + 
                  " -vcodec rawvideo - 2>/dev/null";
        }
        
        // 開啟 pipe
        pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            throw std::runtime_error("無法開啟 FFmpeg pipe！請確認系統已安裝 ffmpeg。");
        }
    }

    // 專門開啟 RTSP 串流的 Function
    bool openRtsp(const std::string& rtspUrl) {
        if (pipe) {
            pclose(pipe);
            pipe = nullptr;
        }
        std::string cmd = "ffmpeg -rtsp_transport tcp -i \"" + rtspUrl + "\" -f image2pipe -pix_fmt rgb24 -s " + 
                          std::to_string(width) + "x" + std::to_string(height) + 
                          " -vcodec rawvideo - 2>/dev/null";
        
        pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            std::cerr << "無法開啟 FFmpeg RTSP pipe！請確認系統已安裝 ffmpeg。\n";
            return false;
        }
        return true;
    }

    // 解構子：安全關閉 pipe
    ~VideoReader() {
        if (pipe) {
            pclose(pipe);
            pipe = nullptr;
        }
    }

    // 讀取下一張 Frame
    // 成功回傳 true，影片結束或失敗回傳 false
    bool readFrame(std::vector<uint8_t>& frameData) {
        if (!pipe) return false;
        
        // 確保接收陣列的 size 正確
        if (frameData.size() != frameSize) {
            frameData.resize(frameSize);
        }
        
        // 從 pipe 讀取剛好一張 Frame 大小的 raw data
        size_t bytesRead = fread(frameData.data(), 1, frameSize, pipe);
        
        // 如果讀取到的 byte 數等於預期，代表成功讀取一張完整的 Frame
        return bytesRead == frameSize;
    }
};

#endif // VIDEO_READER_H