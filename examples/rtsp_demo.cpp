#include "HermesII_sdk.h"
#include "ffmpeg_precoss.h"
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    // Default RTSP URL or from arguments
    std::string rtsp_url = "rtsp://10.0.1.105:1254/live";
    if (argc > 1) {
        rtsp_url = argv[1];
    }
    
    std::cout << "Starting RTSP streaming from: " << rtsp_url << std::endl;

    // Initialize SDK
    VisionSDK::VisionSDK sdk;
    VisionSDK::StatusCode status = sdk.Init("models/yolo.tflite");
    if (status != VisionSDK::StatusCode::OK) {
        std::cerr << "Failed to init SDK" << std::endl;
        return -1;
    }

    // Register Callback
    sdk.RegisterVisionSDKCallback([](const VisionSDK::VisionSDKEvent& event) {
        std::cout << "[Callback] Frame: " << event.frame_index 
                  << " Confidence: " << event.confidence 
                  << " Fall: " << (event.is_fall_detected ? "YES" : "NO") 
                  << " Strong: " << (event.is_strong ? "YES" : "NO") << std::endl;
    });

    int width = 800; // Resize width
    int height = 450; // Resize height

    // Open RTSP stream using the newly updated VideoReader logic
    VideoReader reader(width, height);
    if (!reader.openRtsp(rtsp_url)) {
        std::cerr << "Failed to open RTSP stream. Check URL and ffmpeg." << std::endl;
        return -1;
    }

    std::vector<uint8_t> frame_data;
    int frame_count = 0;

    // Keep pulling frames from the stream
    while (reader.readFrame(frame_data)) {
        // Feed into SDK
        sdk.SetInputMemory(frame_data.data(), width, height, 3);
        sdk.ProcessNextFrame();
        
        frame_count++;
        if (frame_count % 30 == 0) {
            std::cout << "[Demo] Processed " << frame_count << " frames..." << std::endl;
        }
    }

    std::cout << "RTSP streaming finished. Processed " << frame_count << " frames." << std::endl;

    return 0;
}
