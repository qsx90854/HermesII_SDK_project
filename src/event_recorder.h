#ifndef EVENT_RECORDER_H
#define EVENT_RECORDER_H

#include <atomic>
#include <cstdint>
#include <vector>
#include <deque>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <utility>
#include "HermesII_sdk.h"

namespace VisionSDK {

// Event-triggered raw frame recorder.
// Continuously spools every frame fed to the SDK into an mmap-backed ring file
// on the SD card. When a fall / bed-exit event fires, the window
// [event - pre_frames, event + post_frames] is materialized into a standalone
// .raw file (plus background snapshot, .meta.json and a global .jsonl log)
// by a background writer thread. Frames never accumulate in RAM.
//
// Threading contract: Configure/PushFrame/OnEvent/Shutdown may be called from
// different threads; they are serialized by api_mtx_, so Shutdown() (from
// Release() or the destructor, possibly on another thread than the one feeding
// frames) can never unmap the ring while a PushFrame is copying into it. After
// Shutdown() every entry point becomes a no-op. IsCapturing() is lock-free.
class EventRecorder {
public:
    // Decision-relevant per-frame snapshot, built by the SDK glue right after
    // Detect(). Kept in a RAM ring aligned with the frame ring; dumped to
    // <event>.analysis.jsonl at finalize so one can later inspect WHY a
    // fall / bed-exit was decided (object blocks, momentum, areas, ...).
    struct FrameAnalysisObject {
        int id = -1;
        float cx = 0, cy = 0;              // center in grid coords
        float dx = 0, dy = 0;              // avg momentum vector
        float strength = 0;                // momentum magnitude
        float acceleration = 0;
        int pixel_count = 0;
        float safe_area_ratio = 0;
        float direction_variance = 0;
        bool in_observation = false;       // Case5 observation phase
        std::vector<uint16_t> blocks;      // grid block indices of this object
    };
    struct FrameAnalysis {
        uint64_t seq = UINT64_MAX;         // filled in by RecordFrameAnalysis
        uint64_t timestamp_ms = 0;
        uint32_t process_time_us = 0;      // Detect() duration for this frame
        int grid_cols = 0, grid_rows = 0, block_size = 0;
        int total_fg_pixels = 0;
        std::vector<FrameAnalysisObject> objects;
    };

    EventRecorder();
    ~EventRecorder();

    // Called from SetConfig(EventRecording_v1). Safe to call multiple times.
    void Configure(bool enable, int pre_frames, int post_frames);

    // Called once per ProcessNextFrame(), before Detect(). Copies the frame
    // into the ring file. Never throws; on storage failure disables itself.
    void PushFrame(const Image& img);

    // Called from the SDK callback wrapper when is_fall_detected/is_bed_exit.
    // bg_image is only consumed on the first trigger (IDLE -> CAPTURING);
    // re-triggers during an active capture are merged into the trigger list.
    void OnEvent(const VisionSDKEvent& event, std::vector<uint8_t>&& bg_image);

    // Test hook: starts one capture exactly as a real event would (same
    // finalize path, files marked event_type "self_test"). No-op while a
    // capture is active. Driven by the sdk.ini key event_record_self_test.
    void TriggerSelfTest(std::vector<uint8_t>&& bg_image);

    // Called after Detect() returns: back-fills the processing time of the
    // most recently pushed frame into its ring slot (PushFrame runs before
    // Detect, so the duration is not known at push time).
    void RecordProcessTime(uint64_t process_time_us);

    // Called after Detect() returns: stores this frame's analysis snapshot.
    void RecordFrameAnalysis(FrameAnalysis&& fa);

    // Cheap gate for the glue: skip building analysis when recording is off.
    bool IsActive() const { return enabled_ && !failed_; }

    // Called from VisionSDK::SetBedRegion: the polygon (input-image pixel
    // coords) in effect at event time is written into the meta json.
    void SetBedRegion(const std::vector<std::pair<int, int>>& points);

    // Called from VisionSDK::SetConfig() each time the corresponding versioned
    // config struct is set: keeps a last-known snapshot so it can be written
    // into <event>.meta.json's "config" object at finalize time (what
    // parameters were actually in effect when this decision was made).
    // EventRecorder does not interpret these values itself. EventRecording_v1
    // itself needs no separate setter: enabled_/pre_frames_/post_frames_
    // already hold it (see Configure()).
    void SetMotionConfig(const MotionEstimation_v1& c);
    void SetObjectConfig(const ObjectExtraction_v1& c);
    void SetFallConfig(const FallDetection_v3& c);
    void SetImageConfig(const ImageRelated_v1& c);

    // True while a capture window is collecting its post frames. The glue
    // layer uses this to grab the background image only at the event start.
    bool IsCapturing() const { return capturing_.load(std::memory_order_relaxed); }

    // Flush in-flight capture (as partial), join writer thread, unmap ring.
    void Shutdown();

private:
    struct Trigger {
        uint64_t seq;
        uint64_t timestamp_ms;   // input timestamp of the frame that triggered
        int frame_index;
        bool is_fall;
        bool is_bed_exit;
        bool is_self_test;
        float confidence;
    };

    struct FinalizeJob {
        uint64_t first_seq;
        uint64_t event_seq;
        uint64_t last_seq;
        bool complete;                 // false when flushed early at Shutdown
        std::vector<Trigger> triggers;
        std::vector<uint8_t> bg_image;
        std::vector<FrameAnalysis> analysis;  // snapshots for the window
        std::vector<std::pair<int, int>> bed_region;  // polygon at event time
        // Config snapshot at event time (see cfg_motion_ etc. below).
        MotionEstimation_v1 cfg_motion;
        ObjectExtraction_v1 cfg_object;
        FallDetection_v3 cfg_fall;
        ImageRelated_v1 cfg_image;
        bool has_motion_cfg, has_object_cfg, has_fall_cfg, has_image_cfg;
        bool event_recording_enabled;
        int event_recording_pre_frames, event_recording_post_frames;
    };

    bool InitRing(int width, int height, int channels);
    void StartWriterLocked();   // caller must hold api_mtx_
    void CloseRing();
    void QueueFinalize(bool complete);
    void WriterLoop();
    void RunFinalize(const FinalizeJob& job);
    void Fail(const std::string& why);

    // --- configuration (SDK thread only) ---
    // Default-on for field testing: an old binary running against the new .so
    // records without any SetConfig call. Disable via
    // SetConfig(EventRecording_v1{enable=false}).
    bool enabled_ = true;
    int pre_frames_ = 300;
    int post_frames_ = 300;

    // --- ring file (SDK thread writes, writer thread reads) ---
    int fd_ = -1;
    uint8_t* map_ = nullptr;
    size_t map_size_ = 0;
    uint8_t* slots_ = nullptr;       // first slot inside map_
    struct RingSlotIndex* index_ = nullptr;
    struct RingFileHeader* hdr_ = nullptr;
    int width_ = 0, height_ = 0, channels_ = 0;
    uint32_t frame_size_ = 0;
    uint32_t slot_size_ = 0;         // frame_size_ rounded up to page size
    uint32_t capacity_ = 0;          // pre + 1 + post + margin
    uint64_t write_seq_ = 0;         // frames written so far; slot = seq % capacity
    bool last_push_written_ = false; // false when the last frame was dropped
    bool ring_ready_ = false;
    bool failed_ = false;

    // Timestamp of the most recently pushed frame (write_seq_ - 1). Kept as a
    // plain field rather than re-reading it back out of index_[] so OnEvent/
    // TriggerSelfTest work identically whether or not the mmap ring exists
    // (see EVENT_RECORDER_SKIP_RAW_STORAGE in event_recorder.cpp).
    uint64_t last_frame_ts_ = 0;

    // RAM ring of per-frame analysis snapshots, same slot layout as the frame
    // ring (~1-2 MB total). Guarded by api_mtx_. Lost on restart (best effort).
    std::vector<FrameAnalysis> analysis_ring_;

    // Bed region polygon currently in effect (guarded by api_mtx_).
    std::vector<std::pair<int, int>> bed_region_;

    // Last-known snapshot of each versioned config struct (guarded by
    // api_mtx_), written into <event>.meta.json's "config" object. The
    // has_*_cfg_ flags distinguish "never set" (field omitted from the JSON)
    // from "set to its struct default".
    MotionEstimation_v1 cfg_motion_{};
    ObjectExtraction_v1 cfg_object_{};
    FallDetection_v3 cfg_fall_{};
    ImageRelated_v1 cfg_image_{};
    bool has_motion_cfg_ = false;
    bool has_object_cfg_ = false;
    bool has_fall_cfg_ = false;
    bool has_image_cfg_ = false;

    std::string base_dir_;           // /mnt/mmcblk1p1 or fallback
    std::string event_dir_;          // <base>/event_record

    // --- capture state machine (mutated under api_mtx_) ---
    std::atomic<bool> capturing_{false};
    uint64_t event_seq_ = 0;
    std::vector<Trigger> triggers_;
    std::vector<uint8_t> event_bg_;

    // Serializes the public API against Shutdown() from another thread.
    std::mutex api_mtx_;
    bool shutdown_ = false;

    // --- writer thread ---
    std::thread writer_;
    bool writer_running_ = false;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::deque<FinalizeJob> jobs_;
    bool stop_ = false;
    // Slots holding seq in [copy_cursor_, copy_last_] are still pending copy;
    // PushFrame drops instead of overwriting them. UINT64_MAX = no active copy.
    uint64_t copy_cursor_ = UINT64_MAX;
    uint64_t copy_last_ = 0;
    uint64_t dropped_frames_ = 0;

    // --- async spooler thread (only used when EVENT_RECORDER_ASYNC_SPOOL=1;
    // see event_recorder.cpp). Offloads the memcpy-into-the-mmap-ring plus
    // msync/madvise off the caller's PushFrame thread onto this thread, so the
    // kernel's dirty-page write throttling (balance_dirty_pages, which fires
    // on the thread that dirties file-backed pages when the SD card can't
    // absorb writes fast enough) stalls the spooler instead of the frame
    // feeder. PushFrame then only does a RAM->RAM copy into spool_q_ and
    // returns. Members are compiled unconditionally (unused when the flag is
    // off, which is harmless) to keep the header free of the switch. ---
    struct SpoolFrame {
        uint64_t seq;
        uint64_t timestamp;
        std::vector<uint8_t> data;
    };
    void SpoolerLoop();
    std::thread spooler_;
    bool spooler_running_ = false;
    std::deque<SpoolFrame> spool_q_;   // guarded by mtx_ (shared with writer)
    uint64_t spooled_seq_ = 0;         // every seq < spooled_seq_ is now in the ring
};

} // namespace VisionSDK

#endif // EVENT_RECORDER_H
