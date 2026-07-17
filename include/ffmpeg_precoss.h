#ifndef VIDEO_READER_H
#define VIDEO_READER_H

#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <algorithm>

class VideoReader {
private:
    FILE* pipe;
    int width;
    int height;
    int channels;
    size_t frameSize;

    // Async thread stuff for RTSP
    bool is_async;
    std::thread worker;
    std::mutex mtx;
    std::vector<uint8_t> latest_frame;
    bool new_frame_ready;
    std::atomic<bool> running;
    std::atomic<bool> stopped;

    void startAsync() {
        is_async = true;
        running = true;
        stopped = false;
        new_frame_ready = false;
        if (latest_frame.size() != frameSize) latest_frame.resize(frameSize);
        
        worker = std::thread([this]() {
            std::vector<uint8_t> buf(frameSize);
            while (running) {
                size_t bytesRead = fread(buf.data(), 1, frameSize, pipe);
                if (bytesRead == frameSize) {
                    std::lock_guard<std::mutex> lock(mtx);
                    std::copy(buf.begin(), buf.end(), latest_frame.begin());
                    new_frame_ready = true;
                } else {
                    break;
                }
            }
            stopped = true;
        });
    }

    void stopAsync() {
        running = false;
        if (worker.joinable()) {
            worker.join();
        }
        if (pipe) {
            pclose(pipe);
            pipe = nullptr;
        }
    }

public:
    // 預設建構子
    VideoReader(int targetWidth = 800, int targetHeight = 450, const std::string& pix_fmt = "rgb24") 
        : pipe(nullptr), width(targetWidth), height(targetHeight), is_async(false), new_frame_ready(false), running(false), stopped(true) {
        channels = (pix_fmt == "gray") ? 1 : 3;
        frameSize = width * height * channels;
    }

    // 建構子
    VideoReader(const std::string& videoPath, int targetWidth = 800, int targetHeight = 450, const std::string& pix_fmt = "rgb24") 
        : pipe(nullptr), width(targetWidth), height(targetHeight), is_async(false), new_frame_ready(false), running(false), stopped(true) {
        
        channels = (pix_fmt == "gray") ? 1 : 3;
        frameSize = width * height * channels;
        
        if (videoPath.find("rtsp://") == 0) {
            if (!openRtsp(videoPath, pix_fmt)) {
                throw std::runtime_error("無法開啟 FFmpeg RTSP pipe！");
            }
        } else {
            std::string cmd = "ffmpeg -i \"" + videoPath + "\" -f image2pipe -pix_fmt " + pix_fmt + " -s " + 
                              std::to_string(width) + "x" + std::to_string(height) + 
                              " -vcodec rawvideo - 2>/dev/null";
            pipe = popen(cmd.c_str(), "r");
            if (!pipe) {
                throw std::runtime_error("無法開啟 FFmpeg pipe！");
            }
        }
    }

        // 專門開啟 RTSP 串流的 Function
    bool openRtsp(const std::string& rtspUrl, const std::string& pix_fmt = "rgb24") {
        if (is_async) {
            stopAsync();
        } else if (pipe) {
            pclose(pipe);
            pipe = nullptr;
        }

        std::string cmd = "ffmpeg -rtsp_transport tcp -i \"" + rtspUrl + "\" -f image2pipe -pix_fmt " + pix_fmt + " -s " + 
                          std::to_string(width) + "x" + std::to_string(height) + 
                          " -vcodec rawvideo - 2>/dev/null";
        
        pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            std::cerr << "無法開啟 FFmpeg RTSP pipe！請確認系統已安裝 ffmpeg。\n";
            return false;
        }

        // RTSP 啟動非同步讀取
        startAsync();
        return true;
    }

    // 解構子
    ~VideoReader() {
        if (is_async) {
            stopAsync();
        } else if (pipe) {
            pclose(pipe);
            pipe = nullptr;
        }
    }

    // 讀取下一張 Frame
    bool readFrame(std::vector<uint8_t>& frameData) {
        if (frameData.size() != frameSize) {
            frameData.resize(frameSize);
        }

        if (is_async) {
            if (stopped && !new_frame_ready) return false;

            while (!new_frame_ready && running && !stopped) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }

            if (new_frame_ready) {
                std::lock_guard<std::mutex> lock(mtx);
                std::copy(latest_frame.begin(), latest_frame.end(), frameData.begin());
                new_frame_ready = false;
                return true;
            }
            return false;
            
        } else {
            if (!pipe) return false;
            size_t bytesRead = fread(frameData.data(), 1, frameSize, pipe);
            return bytesRead == frameSize;
        }
    }
};

#endif // VIDEO_READER_H