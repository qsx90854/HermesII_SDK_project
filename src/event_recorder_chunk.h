#ifndef EVENT_RECORDER_CHUNK_H
#define EVENT_RECORDER_CHUNK_H

// Chunk-storage event recorder (2026-07-31). A drop-in alternative to the
// mmap-ring EventRecorder (event_recorder.h/.cpp), designed to remove the
// finalize "read the window back from the ring + rewrite it as .raw" I/O spike
// that starves the frame spooler and drops pre-window frames on the board.
//
// Idea: every frame is appended, once, into rolling "chunk" files on the SD
// (each chunk = kChunkFrames frames). An event does NOT copy any frame data --
// it just records the window's seq range and which chunk files hold it into a
// tiny .meta.json, and marks those chunks "retained". A throttled background GC
// deletes only chunks that have aged out of the rolling buffer AND belong to no
// retained event. Real fall/bed-exit events are NEVER auto-deleted; when the SD
// is full new events are logged meta-only instead (see the design doc
// "事件錄影_chunk儲存架構設計.md").
//
// Public interface mirrors EventRecorder so the SDK glue can select either at
// build time. FrameAnalysis types are reused from event_recorder.h.

#include <atomic>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <condition_variable>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "HermesII_sdk.h"
#include "event_recorder.h"   // reuse EventRecorder::FrameAnalysis / FrameAnalysisObject

namespace VisionSDK {

class EventRecorderChunk {
public:
    using FrameAnalysis = EventRecorder::FrameAnalysis;
    using FrameAnalysisObject = EventRecorder::FrameAnalysisObject;

    EventRecorderChunk();
    ~EventRecorderChunk();

    void Configure(bool enable, int pre_frames, int post_frames, bool store_raw = true);
    void PushFrame(const Image& img);
    void OnEvent(const VisionSDKEvent& event, std::vector<uint8_t>&& bg_image);
    void TriggerSelfTest(std::vector<uint8_t>&& bg_image);
    void RecordProcessTime(uint64_t process_time_us);
    void RecordFrameAnalysis(FrameAnalysis&& fa);
    bool IsActive() const { return enabled_ && !failed_; }
    void SetBedRegion(const std::vector<std::pair<int, int>>& points);
    void SetMotionConfig(const MotionEstimation_v1& c);
    void SetObjectConfig(const ObjectExtraction_v1& c);
    void SetFallConfig(const FallDetection_v3& c);
    void SetImageConfig(const ImageRelated_v1& c);
    bool IsCapturing() const { return capturing_.load(std::memory_order_relaxed); }
    void Shutdown();
    void SetPcOutputBase(const char* base) { (void)base; }   // chunk backend: no PC-sidecar mode

private:
    struct Trigger {
        uint64_t seq;
        uint64_t timestamp_ms;
        int frame_index;
        bool is_fall;
        bool is_bed_exit;
        bool is_self_test;
        float confidence;
    };

    // A frame handed to the spooler thread to append into a chunk file.
    struct SpoolFrame {
        uint64_t seq;
        uint64_t timestamp_ms;
        std::vector<uint8_t> data;
    };

    // A completed window handed to the writer thread. Writes the .meta.json
    // (referencing chunk files -- NO frame data is copied), the background
    // snapshot, the analysis sidecar, retention.idx and the global log.
    struct MetaJob {
        uint64_t first_seq, event_seq, last_seq;
        bool complete;
        bool stored;                 // false => logged only (sd full or store_raw off)
        const char* not_stored_reason;   // "" when stored
        std::vector<Trigger> triggers;
        std::vector<uint8_t> bg_image;
        std::vector<FrameAnalysis> analysis;
        std::vector<std::pair<int, int>> bed_region;
        MotionEstimation_v1 cfg_motion;
        ObjectExtraction_v1 cfg_object;
        FallDetection_v3 cfg_fall;
        ImageRelated_v1 cfg_image;
        bool has_motion_cfg, has_object_cfg, has_fall_cfg, has_image_cfg;
        int cfg_pre_frames, cfg_post_frames;
    };

    // A retained event's kept seq range (its chunks must never be GC'd).
    struct Retained {
        uint64_t first_seq, last_seq;
        std::string meta_name;
    };

    bool InitStorage(int width, int height, int channels);   // caller holds api_mtx_
    void StartThreadsLocked();
    void SpoolerLoop();      // appends frames into chunk files
    void WriterLoop();       // writes meta jobs + runs periodic GC
    void QueueMeta(bool complete);   // caller holds api_mtx_
    void WriteMeta(const MetaJob& job);
    void RunGc();            // deletes rolling-buffer chunks not retained (throttled)
    bool SpaceForOneEvent() const;   // statvfs: room to retain another window?
    void Fail(const std::string& why);

    std::string ChunkPath(uint64_t chunk_id) const;
    uint64_t ChunkIdOf(uint64_t seq) const { return seq / (uint64_t)chunk_frames_; }
    bool OverlapsRetained(uint64_t chunk_id) const;   // caller holds mtx_

    // --- configuration (SDK thread) ---
    bool enabled_ = true;
    int pre_frames_ = 300;
    int post_frames_ = 150;
    bool store_raw_ = true;          // false => never retain chunks (json/analysis only)
    int chunk_frames_ = 128;

    // --- geometry ---
    int width_ = 0, height_ = 0, channels_ = 0;
    uint32_t frame_size_ = 0;
    bool ready_ = false;
    bool failed_ = false;

    std::string base_dir_;
    std::string event_dir_;          // <base>/event_record (parent of session folders)
    std::string session_dir_;        // <base>/event_record/<N> -- this run's folder
    long long session_id_ = 0;       // incrementing per SDK run (1,2,3,...)
    std::string frames_dir_;         // <session_dir_>/frames
    std::string retention_path_;     // <session_dir_>/retention.idx

    uint64_t write_seq_ = 0;         // monotonic frame counter (SDK thread)
    uint64_t last_frame_ts_ = 0;
    bool last_push_written_ = false;

    // --- capture state machine (api_mtx_) ---
    std::atomic<bool> capturing_{false};
    uint64_t event_seq_ = 0;
    std::vector<Trigger> triggers_;
    std::vector<uint8_t> event_bg_;

    // RAM ring of recent analysis snapshots, indexed by seq % analysis_cap_.
    std::vector<FrameAnalysis> analysis_ring_;
    uint32_t analysis_cap_ = 0;

    std::vector<std::pair<int, int>> bed_region_;
    MotionEstimation_v1 cfg_motion_{};
    ObjectExtraction_v1 cfg_object_{};
    FallDetection_v3 cfg_fall_{};
    ImageRelated_v1 cfg_image_{};
    bool has_motion_cfg_ = false, has_object_cfg_ = false;
    bool has_fall_cfg_ = false, has_image_cfg_ = false;

    std::mutex api_mtx_;
    bool shutdown_ = false;

    // --- shared worker state (mtx_) ---
    std::mutex mtx_;
    std::condition_variable cv_;
    bool stop_ = false;

    // spooler
    std::thread spooler_;
    bool spooler_running_ = false;
    std::deque<SpoolFrame> spool_q_;
    uint64_t spooled_seq_ = 0;
    uint64_t dropped_frames_ = 0;
    int cur_chunk_fd_ = -1;          // spooler thread only
    uint64_t cur_chunk_id_ = UINT64_MAX;  // spooler thread only

    // writer + GC
    std::thread writer_;
    bool writer_running_ = false;
    std::deque<MetaJob> jobs_;

    // retention / existing chunks (mtx_)
    std::deque<Retained> retained_;
    std::set<uint64_t> existing_chunks_;   // chunk_ids currently on disk
    uint64_t gc_last_seq_seen_ = 0;        // latest write_seq_ observed by GC's floor calc
};

} // namespace VisionSDK

#endif // EVENT_RECORDER_CHUNK_H
