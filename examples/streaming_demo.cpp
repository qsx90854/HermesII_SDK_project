#include "HermesII_sdk.h"
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <vector>
#include <cstring>
#include <atomic>

// Simulated Shared Memory Buffer
struct SharedBuffer {
    std::vector<unsigned char> data;
    int width;
    int height;
    int channels;
    std::mutex mtx;
    std::condition_variable cv;
    bool new_frame_ready = false;
    bool stop = false;
};

// Producer: Simulates a camera driver or external process writing to shared memory
void CameraProducer(SharedBuffer& shared_mem) {
    int frame_count = 0;
    while (!shared_mem.stop) {
        {
            std::lock_guard<std::mutex> lock(shared_mem.mtx);
            
            // Simulate writing pixel data (e.g., changing color over time)
            // Just fill with a dummy pattern for this example
            int value = frame_count % 255;
            std::fill(shared_mem.data.begin(), shared_mem.data.end(), static_cast<unsigned char>(value));
            
            // Notify that data is ready
            shared_mem.new_frame_ready = true;
        }
        
        // Notify the consumer (Trigger)
        shared_mem.cv.notify_one();
        
        std::cout << "[CameraProducer] Updated frame " << frame_count << std::endl;
        
        frame_count++;
        // Simulate ~30 FPS
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }
}

int main() {
    // 1. Initialize Shared Memory Layout
    // Assume 640x480 RGB image
    int width = 640;
    int height = 480;
    int channels = 3;
    
    SharedBuffer shared_mem;
    shared_mem.width = width;
    shared_mem.height = height;
    shared_mem.channels = channels;
    shared_mem.data.resize(width * height * channels);
    
    // 2. Initialize SDK
    VisionSDK::VisionSDK sdk;
    // Config logic removed or moved to defaults
    
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

    // Set Input Buffer (Shared Memory)
    sdk.SetInputMemory(shared_mem.data.data(), width, height, channels);
    
    // 3. Start Camera Producer Thread
    std::thread producer(CameraProducer, std::ref(shared_mem));
    
    // 4. Consumer Loop (Main Application / Algorithm)
    // Wait for trigger, then process
    int processed_count = 0;
    const int MAX_FRAMES = 10; // Run for a limited time for demo
    
    while (processed_count < MAX_FRAMES) {
        std::unique_lock<std::mutex> lock(shared_mem.mtx);
        
        // Wait for the trigger (Condition Variable)
        shared_mem.cv.wait(lock, [&shared_mem] { return shared_mem.new_frame_ready; });
        
        std::cout << "[Consumer] Received trigger. Processing frame..." << std::endl;
        
        // Trigger SDK Processing
        // The data pointer is already set via SetInputMemory
        sdk.ProcessNextFrame(); 
        
        // Verify pixel value for demo
        std::cout << "[Consumer] First pixel value: " << (int)shared_mem.data[0] << std::endl;
        
        // Reset flag
        shared_mem.new_frame_ready = false;
        
        lock.unlock(); // Release lock while processing (if copying data) or keep if processing directly
        
        processed_count++;
    }
    
    // Cleanup
    shared_mem.stop = true;
    if (producer.joinable()) {
        producer.join();
    }
    
    std::cout << "Streaming demo finished." << std::endl;
    return 0;
}
