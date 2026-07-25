// EventRecorder: event-triggered raw frame recording, self-contained.
//
// Data path (all sizes for gray8 800x450):
//   PushFrame (SDK thread) --memcpy 360KB--> mmap ring file on SD card
//   OnEvent   (SDK thread) --marks window [event-300, event+300]
//   WriterLoop (background thread) --copies window--> evt_*.raw + _bg.raw
//                                                    + .meta.json + event_record.jsonl
//
// The ring lives on the SD card so no frame history is held in RAM. The ring
// keeps margin slots beyond the event window so the writer can copy while the
// SDK keeps pushing; PushFrame drops (and counts) a frame rather than
// overwrite a slot the writer has not copied yet.

#include "event_recorder.h"

// Perf-isolation test switch (2026-07-22): set to 1 to disable ALL SD-card
// raw-frame storage -- no hermes_frame_ring.dat, no per-frame mmap
// memcpy/msync/madvise in PushFrame(), no evt_*.raw / evt_*_bg.raw on
// finalize. Only .meta.json / .analysis.jsonl / event_record.jsonl still get
// written, so the trigger/logging pipeline stays testable while isolating
// whether the raw-frame ring I/O is what's adding delay to frame feeding.
// Leave at 0 for normal operation.
#ifndef EVENT_RECORDER_SKIP_RAW_STORAGE
#define EVENT_RECORDER_SKIP_RAW_STORAGE 0
#endif

// Perf switch (2026-07-23): set to 1 to move the per-frame memcpy-into-the-
// mmap-ring (plus its msync/madvise) OFF the caller's PushFrame thread onto a
// dedicated spooler thread. On the edge board, raw storage periodically
// stalled the frame feeder ~250-800ms (confirmed genuine late arrivals, not
// dropped frames -- seq stayed contiguous), consistent with kernel
// balance_dirty_pages throttling the thread that dirties the file-backed
// pages when the SD card can't keep up. With this on, PushFrame only does a
// RAM->RAM copy into a small bounded queue and returns; the spooler thread
// takes the throttle hit. Costs ~kMaxSpoolFrames*frame_size RAM of buffering.
// Mutually exclusive with EVENT_RECORDER_SKIP_RAW_STORAGE (no ring to spool
// into there). Leave at 0 for the original synchronous behavior.
#ifndef EVENT_RECORDER_ASYNC_SPOOL
#define EVENT_RECORDER_ASYNC_SPOOL 0
#endif

#if EVENT_RECORDER_ASYNC_SPOOL && EVENT_RECORDER_SKIP_RAW_STORAGE
#error "EVENT_RECORDER_ASYNC_SPOOL and EVENT_RECORDER_SKIP_RAW_STORAGE are mutually exclusive"
#endif

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <map>

namespace VisionSDK {

namespace {

const char* kBasePath = "/mnt/mmcblk1p1";      // SD card mount point on the edge board
const char* kFallbackDir = "./event_record";   // used when SD path is not writable (PC test)
const char* kRingFileName = "hermes_frame_ring.dat";
const char* kLogFileName = "event_record.jsonl";
const uint32_t kRingMagic = 0x48524731;        // "HRG1"
const uint32_t kRingVersion = 2;               // v2: slot index gained process_time_us
// Extra ring slots beyond the pre+1+post window, so frames arriving DURING a
// finalize (which copies the ~230MB window out, ~19s on the board's async SD)
// have somewhere to land instead of wrapping into and overwriting the window
// still being copied -- overwrite protection drops them otherwise. Bumped
// 64 -> 256 (2026-07-25): at ~7fps a 19s finalize sees ~130 new frames; 64
// only buffered ~9s so ~40 got dropped (the ring_dropped_frames seen on the
// board), which would surface as MISSING frames in a back-to-back second
// event's recording. 256 (~37s buffer) covers a full finalize with headroom.
// Cost: larger ring FILE on SD (~330MB vs 256MB); no extra RAM (resident set
// is bounded by kResidentSlots).
const uint32_t kMarginSlots = 256;             // extra slots protecting the copy window
const size_t kPageSize = 4096;
const uint64_t kInvalidSeq = UINT64_MAX;
// madvise(DONTNEED) slots older than this.
//
// Tested bumping 32 -> 128 (2026-07-23) as a diagnostic: hypothesis was that
// madvise() on a slot whose async writeback hadn't finished yet blocks the
// calling thread, causing the periodic ~250-800ms frame-arrival stalls seen
// with raw storage enabled (~20-26 per 600 frames, irregular). Result: zero
// measurable change in stall count/frequency/magnitude at 128 vs 32 -- ruled
// out, reverted to 32. The stalls are most likely a lower-level SD
// card/kernel dirty-page characteristic, not tied to this eviction window.
const int kResidentSlots = 32;

// Max frames buffered in RAM waiting for the async spooler (EVENT_RECORDER_
// ASYNC_SPOOL). Sized to absorb a worst-case stall burst: observed stalls are
// <=~800ms and frames arrive ~every 130ms, so ~6 frames pile up per stall; 24
// gives ~4x headroom. At gray8 800x480 (~384KB/frame) this bounds the extra
// RAM to ~9MB. If the queue ever fills (spooler can't keep up on AVERAGE, not
// just in bursts) the incoming frame is dropped rather than growing RAM
// without bound -- same "drop, don't block" policy as the synchronous ring.
const size_t kMaxSpoolFrames = 24;

size_t PageAlign(size_t n) { return (n + kPageSize - 1) & ~(kPageSize - 1); }

bool WriteAll(int fd, const void* buf, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    while (len > 0) {
        ssize_t n = write(fd, p, len);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += n;
        len -= (size_t)n;
    }
    return true;
}

void FormatTime(time_t t, char* out, size_t out_len, const char* fmt) {
    struct tm tm_buf;
    localtime_r(&t, &tm_buf);
    strftime(out, out_len, fmt, &tm_buf);
}

// IMPORTANT: All number formatting in this file goes through snprintf/printf
// (plain C stdio), never through C++ iostreams. When an old executable that
// carries its own (static) libstdc++ loads this .so, the two libstdc++
// copies fight over locale/num_put facets and `stream << integer` silently
// puts the stream into a fail state — strings still print, but the first
// integer kills the stream and everything after it is dropped. That is how
// the truncated .meta.json / .jsonl files were produced.
#if defined(__GNUC__)
__attribute__((format(printf, 2, 3)))
#endif
void AppendF(std::string& out, const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n > 0) {
        out.append(buf, (n < (int)sizeof(buf)) ? (size_t)n : sizeof(buf) - 1);
    }
}

#if defined(__GNUC__)
__attribute__((format(printf, 1, 2)))
#endif
void LogF(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stdout, fmt, ap);
    va_end(ap);
    fputc('\n', stdout);
    fflush(stdout);
}

} // anonymous namespace

// On-disk layout: [header page][slot index pages][slot data]
struct RingFileHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t frame_size;
    uint32_t slot_size;
    uint32_t capacity;
    uint64_t write_seq;
};

struct RingSlotIndex {
    uint64_t seq;             // kInvalidSeq when the slot was never written
    uint64_t timestamp_ms;
    uint32_t process_time_us; // Detect() duration, back-filled after the frame
    uint32_t reserved0;
};

EventRecorder::EventRecorder() {}

EventRecorder::~EventRecorder() {
    Shutdown();
}

void EventRecorder::Configure(bool enable, int pre_frames, int post_frames) {
    std::lock_guard<std::mutex> api_lk(api_mtx_);
    if (enable) shutdown_ = false;   // allow re-enable after a Shutdown
    enabled_ = enable;
    if (pre_frames >= 0) pre_frames_ = pre_frames;
    if (post_frames >= 0) post_frames_ = post_frames;
    LogF("[EventRecorder] Configure: enable=%d pre_frames=%d post_frames=%d",
         (int)enable, pre_frames_, post_frames_);

    if (enabled_) StartWriterLocked();
}

void EventRecorder::StartWriterLocked() {
    if (writer_running_) return;
    stop_ = false;
    writer_ = std::thread(&EventRecorder::WriterLoop, this);
    writer_running_ = true;
#if EVENT_RECORDER_ASYNC_SPOOL
    if (!spooler_running_) {
        spooler_ = std::thread(&EventRecorder::SpoolerLoop, this);
        spooler_running_ = true;
    }
#endif
}

void EventRecorder::Fail(const std::string& why) {
    failed_ = true;
    LogF("[EventRecorder] DISABLED: %s (errno=%d %s)", why.c_str(), errno, strerror(errno));
}

bool EventRecorder::InitRing(int width, int height, int channels) {
    // Pick storage location: SD card, else local fallback for PC testing.
    if (access(kBasePath, W_OK) == 0) {
        base_dir_ = kBasePath;
        event_dir_ = std::string(kBasePath) + "/event_record";
    } else {
        LogF("[EventRecorder] Warning: %s not writable, falling back to %s",
             kBasePath, kFallbackDir);
        base_dir_ = kFallbackDir;
        event_dir_ = kFallbackDir;
        if (mkdir(base_dir_.c_str(), 0755) != 0 && errno != EEXIST) {
            Fail("mkdir fallback dir failed");
            return false;
        }
    }
    if (mkdir(event_dir_.c_str(), 0755) != 0 && errno != EEXIST) {
        Fail("mkdir event dir failed");
        return false;
    }

    width_ = width;
    height_ = height;
    channels_ = channels;
    frame_size_ = (uint32_t)(width * height * channels);
    slot_size_ = (uint32_t)PageAlign(frame_size_);
    capacity_ = (uint32_t)(pre_frames_ + 1 + post_frames_ + kMarginSlots);

#if EVENT_RECORDER_SKIP_RAW_STORAGE
    // Perf-isolation test mode: no ring file, no mmap, no per-frame SD I/O at
    // all. Only allocate the RAM-only analysis ring (needed for
    // .analysis.jsonl) and mark ourselves ready; fd_/map_/hdr_/index_/slots_
    // all stay null/-1, so every function that touches them must be (and is)
    // guarded to skip that work in this mode -- see PushFrame, RunFinalize.
    analysis_ring_.assign(capacity_, FrameAnalysis());
    write_seq_ = 0;
    ring_ready_ = true;
    LogF("[EventRecorder] PERF-TEST MODE: EVENT_RECORDER_SKIP_RAW_STORAGE=1 -- "
         "no SD ring file, no raw/bg frame storage. Only .meta.json / "
         ".analysis.jsonl / event_record.jsonl will be written, under %s",
         event_dir_.c_str());
    return true;
#else
    size_t index_bytes = PageAlign((size_t)capacity_ * sizeof(RingSlotIndex));
    size_t data_off = kPageSize + index_bytes;
    map_size_ = data_off + (size_t)capacity_ * slot_size_;

    std::string ring_path = base_dir_ + "/" + kRingFileName;
    fd_ = open(ring_path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_ < 0) {
        Fail("open ring file failed: " + ring_path);
        return false;
    }

    // Resume the previous ring if its geometry matches (crash recovery keeps
    // pre-event history), otherwise rebuild from scratch.
    RingFileHeader existing;
    bool resume = false;
    ssize_t got = pread(fd_, &existing, sizeof(existing), 0);
    if (got == (ssize_t)sizeof(existing) &&
        existing.magic == kRingMagic && existing.version == kRingVersion &&
        existing.width == (uint32_t)width && existing.height == (uint32_t)height &&
        existing.channels == (uint32_t)channels &&
        existing.frame_size == frame_size_ && existing.slot_size == slot_size_ &&
        existing.capacity == capacity_) {
        struct stat st;
        if (fstat(fd_, &st) == 0 && (size_t)st.st_size >= map_size_) {
            resume = true;
        }
    }

    if (ftruncate(fd_, (off_t)map_size_) != 0) {
        Fail("ftruncate ring file failed (SD full?)");
        close(fd_);
        fd_ = -1;
        return false;
    }

    void* m = mmap(nullptr, map_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (m == MAP_FAILED) {
        Fail("mmap ring file failed");
        close(fd_);
        fd_ = -1;
        return false;
    }
    map_ = static_cast<uint8_t*>(m);
    hdr_ = reinterpret_cast<RingFileHeader*>(map_);
    index_ = reinterpret_cast<RingSlotIndex*>(map_ + kPageSize);
    slots_ = map_ + data_off;

    if (resume) {
        write_seq_ = hdr_->write_seq;
        LogF("[EventRecorder] Resumed ring %s at seq=%llu",
             ring_path.c_str(), (unsigned long long)write_seq_);
    } else {
        write_seq_ = 0;
        for (uint32_t i = 0; i < capacity_; ++i) {
            index_[i].seq = kInvalidSeq;
            index_[i].timestamp_ms = 0;
            index_[i].process_time_us = 0;
            index_[i].reserved0 = 0;
        }
        hdr_->magic = kRingMagic;
        hdr_->version = kRingVersion;
        hdr_->width = (uint32_t)width;
        hdr_->height = (uint32_t)height;
        hdr_->channels = (uint32_t)channels;
        hdr_->frame_size = frame_size_;
        hdr_->slot_size = slot_size_;
        hdr_->capacity = capacity_;
        hdr_->write_seq = 0;
        msync(map_, kPageSize + index_bytes, MS_ASYNC);
        LogF("[EventRecorder] Created ring %s capacity=%u slots (%llu MB)",
             ring_path.c_str(), capacity_,
             (unsigned long long)(map_size_ / (1024 * 1024)));
    }

    analysis_ring_.assign(capacity_, FrameAnalysis());

    ring_ready_ = true;
    return true;
#endif // EVENT_RECORDER_SKIP_RAW_STORAGE
}

void EventRecorder::CloseRing() {
    if (map_) {
        msync(map_, map_size_, MS_ASYNC);
        munmap(map_, map_size_);
        map_ = nullptr;
        hdr_ = nullptr;
        index_ = nullptr;
        slots_ = nullptr;
    }
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
    ring_ready_ = false;
}

void EventRecorder::PushFrame(const Image& img) {
    std::lock_guard<std::mutex> api_lk(api_mtx_);
    if (shutdown_ || !enabled_ || failed_) return;
    if (!img.data || img.width <= 0 || img.height <= 0 || img.channels <= 0) return;

    if (!ring_ready_) {
        if (!InitRing(img.width, img.height, img.channels)) return;
        // Recorder is default-on: nobody may ever call Configure, so make
        // sure the finalize worker exists once frames start flowing.
        StartWriterLocked();
    }

    if ((uint32_t)(img.width * img.height * img.channels) != frame_size_) {
        Fail("frame geometry changed after ring init");
        return;
    }

    last_frame_ts_ = img.timestamp;

#if EVENT_RECORDER_SKIP_RAW_STORAGE
    // Perf-isolation test mode: no ring, no memcpy/msync/madvise -- just
    // advance the sequence counter so OnEvent/QueueFinalize window
    // bookkeeping (and thus .meta.json/.analysis.jsonl output) keeps working.
    last_push_written_ = true;
    write_seq_++;
#elif EVENT_RECORDER_ASYNC_SPOOL
    // Async spooler mode: do NOT touch the mmap ring on this (the caller's)
    // thread. Copy the frame into a small bounded RAM queue and hand it to
    // SpoolerLoop(), which does the ring memcpy + msync/madvise -- so the
    // kernel dirty-page write throttling stalls the spooler, not the feeder.
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (spool_q_.size() >= kMaxSpoolFrames) {
            // Spooler is behind even with the buffer: drop this frame rather
            // than grow RAM without bound. seq is NOT consumed (matches the
            // synchronous ring's "retry the same seq on the next frame").
            dropped_frames_++;
            last_push_written_ = false;
            return;
        }
        SpoolFrame sf;
        sf.seq = write_seq_;
        sf.timestamp = img.timestamp;
        sf.data.assign(reinterpret_cast<const uint8_t*>(img.data),
                       reinterpret_cast<const uint8_t*>(img.data) + frame_size_);
        spool_q_.push_back(std::move(sf));
    }
    cv_.notify_one();
    last_push_written_ = true;
    write_seq_++;
#else
    // Never overwrite a slot the writer has not copied out yet: drop instead.
    if (write_seq_ >= capacity_) {
        uint64_t overwritten_seq = write_seq_ - capacity_;
        std::lock_guard<std::mutex> lk(mtx_);
        if (copy_cursor_ != kInvalidSeq &&
            overwritten_seq >= copy_cursor_ && overwritten_seq <= copy_last_) {
            dropped_frames_++;
            last_push_written_ = false;
            return;
        }
    }

    uint32_t slot = (uint32_t)(write_seq_ % capacity_);
    uint8_t* dst = slots_ + (size_t)slot * slot_size_;
    memcpy(dst, img.data, frame_size_);
    index_[slot].seq = write_seq_;
    index_[slot].timestamp_ms = img.timestamp;
    index_[slot].process_time_us = 0;   // back-filled by RecordProcessTime()
    last_push_written_ = true;
    write_seq_++;
    hdr_->write_seq = write_seq_;

    // Queue async writeback of the slot; header/index synced periodically.
    msync(dst, slot_size_, MS_ASYNC);
    if ((write_seq_ & 63) == 0) {
        msync(map_, kPageSize, MS_ASYNC);
        msync(reinterpret_cast<uint8_t*>(index_),
              PageAlign((size_t)capacity_ * sizeof(RingSlotIndex)), MS_ASYNC);
    }
    // Keep resident memory bounded on the small-RAM board: pages written
    // kResidentSlots ago have long been queued for writeback, drop them.
    if (write_seq_ > (uint64_t)kResidentSlots) {
        uint32_t old_slot = (uint32_t)((write_seq_ - 1 - kResidentSlots) % capacity_);
        madvise(slots_ + (size_t)old_slot * slot_size_, slot_size_, MADV_DONTNEED);
    }
#endif // EVENT_RECORDER_SKIP_RAW_STORAGE

    // Post-window bookkeeping: latest written seq is write_seq_ - 1.
    if (capturing_ && (write_seq_ - 1) >= event_seq_ + (uint64_t)post_frames_) {
        QueueFinalize(true);
    }
}

void EventRecorder::OnEvent(const VisionSDKEvent& event, std::vector<uint8_t>&& bg_image) {
    std::lock_guard<std::mutex> api_lk(api_mtx_);
    if (shutdown_ || !enabled_ || failed_ || !ring_ready_ || write_seq_ == 0) return;

    Trigger t;
    t.seq = write_seq_ - 1;                // frame currently being processed
    t.timestamp_ms = last_frame_ts_;       // == index_[t.seq % capacity_].timestamp_ms when the ring exists
    t.frame_index = event.frame_index;
    t.is_fall = event.is_fall_detected;
    t.is_bed_exit = event.is_bed_exit;
    t.is_self_test = false;
    t.confidence = event.confidence;

    if (capturing_) {
        // Re-trigger during an active capture: merged into the trigger list,
        // window stays fixed at [event_seq_ - pre, event_seq_ + post].
        triggers_.push_back(t);
        return;
    }

    capturing_ = true;
    event_seq_ = t.seq;
    triggers_.clear();
    triggers_.push_back(t);
    event_bg_ = std::move(bg_image);
    LogF("[EventRecorder] Event start: type=%s frame_index=%d seq=%llu",
         t.is_fall ? (t.is_bed_exit ? "fall+bed_exit" : "fall") : "bed_exit",
         t.frame_index, (unsigned long long)t.seq);

    if (post_frames_ == 0) {
        QueueFinalize(true);
    }
}

void EventRecorder::TriggerSelfTest(std::vector<uint8_t>&& bg_image) {
    std::lock_guard<std::mutex> api_lk(api_mtx_);
    if (shutdown_ || !enabled_ || failed_ || !ring_ready_ || write_seq_ == 0) return;
    if (capturing_) return;

    Trigger t;
    t.seq = write_seq_ - 1;
    t.timestamp_ms = last_frame_ts_;
    t.frame_index = (int)t.seq;
    t.is_fall = false;
    t.is_bed_exit = false;
    t.is_self_test = true;
    t.confidence = 1.0f;

    capturing_ = true;
    event_seq_ = t.seq;
    triggers_.clear();
    triggers_.push_back(t);
    event_bg_ = std::move(bg_image);
    LogF("[EventRecorder][SELF-TEST] TEST recording started (NOT a real event) at seq=%llu"
         " (pre=%d post=%d); will save after %d more frames.",
         (unsigned long long)t.seq, pre_frames_, post_frames_, post_frames_);

    if (post_frames_ == 0) {
        QueueFinalize(true);
    }
}

void EventRecorder::RecordProcessTime(uint64_t process_time_us) {
    std::lock_guard<std::mutex> api_lk(api_mtx_);
    if (shutdown_ || !enabled_ || failed_ || !ring_ready_ || write_seq_ == 0) return;
    if (!last_push_written_) return;   // last frame was dropped, nothing to annotate
#if EVENT_RECORDER_ASYNC_SPOOL
    // The just-pushed frame may still be in the spool queue (not yet in the
    // ring), so we can't back-fill its ring slot here. Process time is carried
    // by the analysis snapshot (RecordFrameAnalysis stores it too) and sourced
    // from there in RunFinalize.
    (void)process_time_us;
    return;
#else
    if (!index_) return;   // no ring in EVENT_RECORDER_SKIP_RAW_STORAGE mode -- nothing to annotate
    uint32_t slot = (uint32_t)((write_seq_ - 1) % capacity_);
    index_[slot].process_time_us =
        (process_time_us > UINT32_MAX) ? UINT32_MAX : (uint32_t)process_time_us;
#endif
}

void EventRecorder::SetBedRegion(const std::vector<std::pair<int, int>>& points) {
    std::lock_guard<std::mutex> api_lk(api_mtx_);
    bed_region_ = points;
}

void EventRecorder::SetMotionConfig(const MotionEstimation_v1& c) {
    std::lock_guard<std::mutex> api_lk(api_mtx_);
    cfg_motion_ = c;
    has_motion_cfg_ = true;
}

void EventRecorder::SetObjectConfig(const ObjectExtraction_v1& c) {
    std::lock_guard<std::mutex> api_lk(api_mtx_);
    cfg_object_ = c;
    has_object_cfg_ = true;
}

void EventRecorder::SetFallConfig(const FallDetection_v3& c) {
    std::lock_guard<std::mutex> api_lk(api_mtx_);
    cfg_fall_ = c;
    has_fall_cfg_ = true;
}

void EventRecorder::SetImageConfig(const ImageRelated_v1& c) {
    std::lock_guard<std::mutex> api_lk(api_mtx_);
    cfg_image_ = c;
    has_image_cfg_ = true;
}

void EventRecorder::RecordFrameAnalysis(FrameAnalysis&& fa) {
    std::lock_guard<std::mutex> api_lk(api_mtx_);
    if (shutdown_ || !enabled_ || failed_ || !ring_ready_ || write_seq_ == 0) return;
    if (!last_push_written_) return;   // last frame was dropped
    uint32_t slot = (uint32_t)((write_seq_ - 1) % capacity_);
    fa.seq = write_seq_ - 1;
    analysis_ring_[slot] = std::move(fa);
}

void EventRecorder::QueueFinalize(bool complete) {
    FinalizeJob job;
    job.event_seq = event_seq_;
    job.first_seq = (event_seq_ >= (uint64_t)pre_frames_) ? event_seq_ - pre_frames_ : 0;
    uint64_t latest = write_seq_ - 1;
    uint64_t want_last = event_seq_ + (uint64_t)post_frames_;
    job.last_seq = (want_last <= latest) ? want_last : latest;
    job.complete = complete;
    job.triggers = triggers_;
    job.bg_image = std::move(event_bg_);
    job.bed_region = bed_region_;
    job.cfg_motion = cfg_motion_;
    job.cfg_object = cfg_object_;
    job.cfg_fall = cfg_fall_;
    job.cfg_image = cfg_image_;
    job.has_motion_cfg = has_motion_cfg_;
    job.has_object_cfg = has_object_cfg_;
    job.has_fall_cfg = has_fall_cfg_;
    job.has_image_cfg = has_image_cfg_;
    job.event_recording_enabled = enabled_;
    job.event_recording_pre_frames = pre_frames_;
    job.event_recording_post_frames = post_frames_;

    // Copy the window's analysis snapshots now (caller holds api_mtx_): the
    // RAM ring keeps mutating after we return, the writer must not touch it.
    if (!analysis_ring_.empty()) {
        job.analysis.reserve((size_t)(job.last_seq - job.first_seq + 1));
        for (uint64_t seq = job.first_seq; seq <= job.last_seq; ++seq) {
            const FrameAnalysis& fa = analysis_ring_[(uint32_t)(seq % capacity_)];
            if (fa.seq == seq) job.analysis.push_back(fa);
        }
    }

    capturing_ = false;
    triggers_.clear();
    event_bg_.clear();

    {
        std::lock_guard<std::mutex> lk(mtx_);
        jobs_.push_back(std::move(job));
    }
    cv_.notify_one();
}

void EventRecorder::WriterLoop() {
    for (;;) {
        FinalizeJob job;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [this] { return stop_ || !jobs_.empty(); });
            if (jobs_.empty()) {
                if (stop_) return;
                continue;
            }
            job = std::move(jobs_.front());
            jobs_.pop_front();
            copy_cursor_ = job.first_seq;   // protect window from overwrite
            copy_last_ = job.last_seq;
#if EVENT_RECORDER_ASYNC_SPOOL
            // Wait until the spooler has written every frame of this window
            // into the ring before we read it. The spooler advances
            // spooled_seq_ for each dequeued frame (even ones it drops for
            // overwrite protection), and the last_seq frame was enqueued
            // before this job, so spooled_seq_ is guaranteed to reach
            // last_seq+1 -- this cannot hang. Intentionally NOT gated on stop_:
            // on shutdown the spooler drains fully before it exits, so waiting
            // purely on spooled_seq_ still reads a complete window.
            cv_.wait(lk, [this, &job] { return spooled_seq_ > job.last_seq; });
#endif
        }

        RunFinalize(job);

        {
            std::lock_guard<std::mutex> lk(mtx_);
            copy_cursor_ = kInvalidSeq;
        }
    }
}

void EventRecorder::SpoolerLoop() {
#if EVENT_RECORDER_ASYNC_SPOOL
    for (;;) {
        SpoolFrame sf;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [this] { return stop_ || !spool_q_.empty(); });
            if (spool_q_.empty()) {
                if (stop_) return;   // fully drained and asked to stop
                continue;
            }
            sf = std::move(spool_q_.front());
            spool_q_.pop_front();
        }

        // Overwrite protection (same rule as the synchronous ring, just run on
        // this thread): writing sf.seq's slot overwrites the frame that lived
        // there capacity_ frames ago; skip (drop) if that older frame is inside
        // a window the writer is currently finalizing.
        uint32_t slot = (uint32_t)(sf.seq % capacity_);
        bool do_write = true;
        if (sf.seq >= capacity_) {
            uint64_t overwritten_seq = sf.seq - capacity_;
            std::lock_guard<std::mutex> lk(mtx_);
            if (copy_cursor_ != kInvalidSeq &&
                overwritten_seq >= copy_cursor_ && overwritten_seq <= copy_last_) {
                dropped_frames_++;
                do_write = false;
            }
        }

        if (do_write && slots_ && index_) {
            uint8_t* dst = slots_ + (size_t)slot * slot_size_;
            memcpy(dst, sf.data.data(), frame_size_);   // dirty-page throttle lands HERE now
            index_[slot].seq = sf.seq;
            index_[slot].timestamp_ms = sf.timestamp;
            index_[slot].process_time_us = 0;   // sourced from analysis in async mode
            if (hdr_) hdr_->write_seq = sf.seq + 1;
            msync(dst, slot_size_, MS_ASYNC);
            if (((sf.seq + 1) & 63) == 0) {
                msync(map_, kPageSize, MS_ASYNC);
                msync(reinterpret_cast<uint8_t*>(index_),
                      PageAlign((size_t)capacity_ * sizeof(RingSlotIndex)), MS_ASYNC);
            }
            if (sf.seq + 1 > (uint64_t)kResidentSlots) {
                uint32_t old_slot = (uint32_t)((sf.seq - kResidentSlots) % capacity_);
                madvise(slots_ + (size_t)old_slot * slot_size_, slot_size_, MADV_DONTNEED);
            }
        }

        {
            std::lock_guard<std::mutex> lk(mtx_);
            spooled_seq_ = sf.seq + 1;   // advance even on drop, so WriterLoop's
                                         // spooled_seq_ wait can never hang
        }
        cv_.notify_all();
    }
#endif
}

void EventRecorder::RunFinalize(const FinalizeJob& job) {
    time_t now = time(nullptr);
    char time_compact[32], time_human[32];
    FormatTime(now, time_compact, sizeof(time_compact), "%Y%m%d_%H%M%S");
    FormatTime(now, time_human, sizeof(time_human), "%Y-%m-%d %H:%M:%S");

    int event_frame_index = job.triggers.empty() ? -1 : job.triggers.front().frame_index;
    bool started_by_self_test = !job.triggers.empty() && job.triggers.front().is_self_test;
    char base_name[96];
    snprintf(base_name, sizeof(base_name), "evt_%s%s_f%d",
             started_by_self_test ? "selftest_" : "", time_compact, event_frame_index);

    std::string raw_name = std::string(base_name) + ".raw";
    std::string bg_name = std::string(base_name) + "_bg.raw";
    std::string raw_tmp = event_dir_ + "/.tmp_" + raw_name;
    std::string raw_final = event_dir_ + "/" + raw_name;

    uint64_t written = 0;
    uint64_t missing = (job.last_seq >= job.first_seq) ? (job.last_seq - job.first_seq + 1) : 0;
    uint64_t first_ts = 0, last_ts = 0;
    // Per-frame input timestamps (from SetInputMemory) and Detect() durations,
    // same order as the frames written into the .raw file.
    std::vector<uint64_t> frame_ts;
    std::vector<uint32_t> frame_proc_us;
    bool raw_saved = false;
    bool bg_saved = false;

#if EVENT_RECORDER_SKIP_RAW_STORAGE
    LogF("[EventRecorder] PERF-TEST MODE: skipping raw/bg file write for %s "
         "(EVENT_RECORDER_SKIP_RAW_STORAGE=1) -- meta/analysis/log JSON still written.",
         base_name);
#else
    // --- 1. Frame window: ring -> .raw (write to .tmp, fsync, rename) ---
    int out = open(raw_tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out < 0) {
        LogF("[EventRecorder] ERROR: finalize open %s failed: %s",
             raw_tmp.c_str(), strerror(errno));
        return;
    }

    missing = 0;
    frame_ts.reserve((size_t)(job.last_seq - job.first_seq + 1));
    frame_proc_us.reserve((size_t)(job.last_seq - job.first_seq + 1));
#if EVENT_RECORDER_ASYNC_SPOOL
    // In async mode the ring slots' process_time_us aren't back-filled (the
    // frame was already spooled by the time RecordProcessTime ran), so source
    // it from the analysis snapshots by seq instead.
    std::map<uint64_t, uint32_t> proc_by_seq;
    for (const FrameAnalysis& fa : job.analysis) proc_by_seq[fa.seq] = fa.process_time_us;
#endif
    bool ok = true;
    for (uint64_t seq = job.first_seq; seq <= job.last_seq; ++seq) {
        uint32_t slot = (uint32_t)(seq % capacity_);
        if (index_[slot].seq != seq) {  // dropped or never-written slot
            missing++;
        } else {
            if (!WriteAll(out, slots_ + (size_t)slot * slot_size_, frame_size_)) {
                LogF("[EventRecorder] ERROR: finalize write failed (SD full?): %s",
                     strerror(errno));
                ok = false;
                break;
            }
            if (written == 0) first_ts = index_[slot].timestamp_ms;
            last_ts = index_[slot].timestamp_ms;
            frame_ts.push_back(index_[slot].timestamp_ms);
#if EVENT_RECORDER_ASYNC_SPOOL
            auto pit = proc_by_seq.find(seq);
            frame_proc_us.push_back(pit != proc_by_seq.end() ? pit->second : 0);
#else
            frame_proc_us.push_back(index_[slot].process_time_us);
#endif
            written++;
        }
        std::lock_guard<std::mutex> lk(mtx_);
        copy_cursor_ = seq + 1;
    }
    if (ok) ok = (fsync(out) == 0);
    close(out);
    if (!ok || written == 0) {
        unlink(raw_tmp.c_str());
        return;
    }
    if (rename(raw_tmp.c_str(), raw_final.c_str()) != 0) {
        LogF("[EventRecorder] ERROR: finalize rename failed: %s", strerror(errno));
        return;
    }
    raw_saved = true;

    // --- 2. Background snapshot taken at the event start ---
    if (!job.bg_image.empty()) {
        std::string bg_tmp = event_dir_ + "/.tmp_" + bg_name;
        int bfd = open(bg_tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (bfd >= 0) {
            bg_saved = WriteAll(bfd, job.bg_image.data(), job.bg_image.size());
            if (bg_saved) bg_saved = (fsync(bfd) == 0);
            close(bfd);
            if (bg_saved) {
                bg_saved = (rename(bg_tmp.c_str(), (event_dir_ + "/" + bg_name).c_str()) == 0);
            }
            if (!bg_saved) unlink(bg_tmp.c_str());
        }
    }
#endif // EVENT_RECORDER_SKIP_RAW_STORAGE

    // Raw storage skipped (or otherwise yielded nothing): fall back to the
    // RAM-only analysis snapshots for per-frame timestamps/process-time, so
    // .meta.json's own frame_timestamps_ms/frame_process_time_us/
    // process_time_us_avg/max are populated directly instead of staying
    // empty/zero and forcing a reader to go open the sibling .analysis.jsonl.
    if (frame_ts.empty() && !job.analysis.empty()) {
        frame_ts.reserve(job.analysis.size());
        frame_proc_us.reserve(job.analysis.size());
        for (const FrameAnalysis& fa : job.analysis) {
            frame_ts.push_back(fa.timestamp_ms);
            frame_proc_us.push_back(fa.process_time_us);
        }
        first_ts = frame_ts.front();
        last_ts = frame_ts.back();
    }

    // --- 2.5 Per-frame analysis snapshots -> <base>.analysis.jsonl ---
    // Line 1 is a header (grid geometry + window info); every following line
    // is one frame: objects with their blocks / momentum / areas etc., i.e.
    // what the detector saw when it made the fall / bed-exit decision.
    std::string analysis_name = std::string(base_name) + ".analysis.jsonl";
    bool analysis_saved = false;
    if (!job.analysis.empty()) {
        std::string a_tmp = event_dir_ + "/.tmp_" + analysis_name;
        int afd = open(a_tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (afd >= 0) {
            bool a_ok = true;
            std::string buf;
            buf.reserve(8192);
            const FrameAnalysis& fr = job.analysis.front();
            AppendF(buf, "{\"type\":\"header\",\"grid_cols\":%d,\"grid_rows\":%d,"
                         "\"block_size\":%d,\"first_seq\":%llu,\"event_seq\":%llu,"
                         "\"last_seq\":%llu,\"frames\":%llu}\n",
                    fr.grid_cols, fr.grid_rows, fr.block_size,
                    (unsigned long long)job.first_seq, (unsigned long long)job.event_seq,
                    (unsigned long long)job.last_seq,
                    (unsigned long long)job.analysis.size());
            a_ok = WriteAll(afd, buf.data(), buf.size());

            for (size_t f = 0; a_ok && f < job.analysis.size(); ++f) {
                const FrameAnalysis& fa = job.analysis[f];
                buf.clear();
                AppendF(buf, "{\"seq\":%llu,\"frame_in_file\":%lld,\"ts\":%llu,"
                             "\"proc_us\":%u,\"total_fg\":%d,\"objects\":[",
                        (unsigned long long)fa.seq,
                        (long long)(fa.seq - job.first_seq),
                        (unsigned long long)fa.timestamp_ms, fa.process_time_us,
                        fa.total_fg_pixels);
                for (size_t o = 0; o < fa.objects.size(); ++o) {
                    const FrameAnalysisObject& ob = fa.objects[o];
                    AppendF(buf, "%s{\"id\":%d,\"cx\":%.2f,\"cy\":%.2f,"
                                 "\"dx\":%.2f,\"dy\":%.2f,\"strength\":%.2f,"
                                 "\"accel\":%.2f,\"pixels\":%d,\"safe_ratio\":%.3f,"
                                 "\"dir_var\":%.3f,\"obs\":%d,\"blocks\":[",
                            o ? "," : "", ob.id, (double)ob.cx, (double)ob.cy,
                            (double)ob.dx, (double)ob.dy, (double)ob.strength,
                            (double)ob.acceleration, ob.pixel_count,
                            (double)ob.safe_area_ratio, (double)ob.direction_variance,
                            ob.in_observation ? 1 : 0);
                    for (size_t b = 0; b < ob.blocks.size(); ++b) {
                        AppendF(buf, b ? ",%u" : "%u", (unsigned)ob.blocks[b]);
                    }
                    buf += "]}";
                }
                buf += "]}\n";
                a_ok = WriteAll(afd, buf.data(), buf.size());
            }
            if (a_ok) a_ok = (fsync(afd) == 0);
            close(afd);
            if (a_ok) {
                a_ok = (rename(a_tmp.c_str(), (event_dir_ + "/" + analysis_name).c_str()) == 0);
            }
            if (!a_ok) unlink(a_tmp.c_str());
            analysis_saved = a_ok;
        }
        if (!analysis_saved) {
            LogF("[EventRecorder] ERROR: write analysis failed (SD full?): %s : %s",
                 analysis_name.c_str(), strerror(errno));
        }
    }

    // --- 3. Event type / stats from the merged trigger list ---
    bool any_fall = false, any_bed_exit = false;
    float max_conf = 0.0f;
    for (size_t i = 0; i < job.triggers.size(); ++i) {
        if (job.triggers[i].is_fall) any_fall = true;
        if (job.triggers[i].is_bed_exit) any_bed_exit = true;
        if (job.triggers[i].confidence > max_conf) max_conf = job.triggers[i].confidence;
    }
    // Real events win over the self-test marker if both land in one window.
    const char* event_type = (any_fall && any_bed_exit) ? "fall+bed_exit"
                             : (any_fall ? "fall"
                             : (any_bed_exit ? "bed_exit" : "self_test"));

    uint64_t dropped;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        dropped = dropped_frames_;
    }

    // --- 4. Per-event .meta.json sidecar (hand-rolled JSON, flat schema).
    // Built exclusively with snprintf (see AppendF comment above): iostream
    // integer formatting is unreliable when an old exe brings its own
    // libstdc++. ---
    {
        std::string js;
        js.reserve(4096 + frame_ts.size() * 14 + job.triggers.size() * 110);
        js += "{\n";
        if (raw_saved) AppendF(js, "  \"file\": \"%s\",\n", raw_name.c_str());
        else           js += "  \"file\": null,\n";
        if (bg_saved) AppendF(js, "  \"bg_file\": \"%s\",\n", bg_name.c_str());
        else          js += "  \"bg_file\": null,\n";
        if (analysis_saved) {
            AppendF(js, "  \"analysis_file\": \"%s\",\n", analysis_name.c_str());
            AppendF(js, "  \"analysis_frames\": %llu,\n",
                    (unsigned long long)job.analysis.size());
        } else {
            js += "  \"analysis_file\": null,\n";
        }
        AppendF(js, "  \"datetime\": \"%s\",\n", time_human);
        AppendF(js, "  \"event_type\": \"%s\",\n", event_type);
        AppendF(js, "  \"event_frame_index\": %d,\n", event_frame_index);
        AppendF(js, "  \"event_timestamp_ms\": %llu,\n",
                (unsigned long long)(job.triggers.empty() ? 0 : job.triggers.front().timestamp_ms));
        AppendF(js, "  \"width\": %d,\n  \"height\": %d,\n  \"channels\": %d,\n",
                width_, height_, channels_);
        AppendF(js, "  \"frame_size\": %u,\n", frame_size_);
        AppendF(js, "  \"pre_frames\": %llu,\n  \"post_frames\": %llu,\n",
                (unsigned long long)(job.event_seq - job.first_seq),
                (unsigned long long)(job.last_seq - job.event_seq));
        AppendF(js, "  \"total_frames\": %llu,\n  \"missing_frames\": %llu,\n",
                (unsigned long long)written, (unsigned long long)missing);
        AppendF(js, "  \"complete\": %s,\n", job.complete ? "true" : "false");
        AppendF(js, "  \"ring_dropped_frames\": %llu,\n", (unsigned long long)dropped);
        AppendF(js, "  \"first_timestamp_ms\": %llu,\n  \"last_timestamp_ms\": %llu,\n",
                (unsigned long long)first_ts, (unsigned long long)last_ts);
        js += "  \"bed_region\": [";
        for (size_t i = 0; i < job.bed_region.size(); ++i) {
            AppendF(js, "%s[%d,%d]", i ? "," : "",
                    job.bed_region[i].first, job.bed_region[i].second);
        }
        js += "],\n";

        // --- Config snapshot: exactly what the caller last passed to
        // SetConfig() for each of these types, in effect when this event
        // fired. Purely informational -- EventRecorder never reads these
        // itself. A type the caller never set is written as null rather
        // than a struct of zero-valued defaults, so it is not mistaken for
        // "explicitly configured to 0/false". ---
        js += "  \"config\": {\n";
        if (job.has_motion_cfg) {
            const MotionEstimation_v1& c = job.cfg_motion;
            AppendF(js,
                "    \"motion_estimation_v1\": {\"grid_cols\": %d, \"grid_rows\": %d, "
                "\"block_size\": %d, \"search_range\": %d, \"history_size\": %d, "
                "\"search_mode\": %d, \"block_change_threshold\": %.6f, "
                "\"enable_block_decay\": %s, \"block_decay_frames\": %d, "
                "\"enable_block_dilation\": %s, \"block_dilation_threshold\": %d},\n",
                c.grid_cols, c.grid_rows, c.block_size, c.search_range, c.history_size,
                c.search_mode, c.block_change_threshold,
                c.enable_block_decay ? "true" : "false", c.block_decay_frames,
                c.enable_block_dilation ? "true" : "false", c.block_dilation_threshold);
        } else {
            js += "    \"motion_estimation_v1\": null,\n";
        }
        if (job.has_object_cfg) {
            const ObjectExtraction_v1& c = job.cfg_object;
            AppendF(js,
                "    \"object_extraction_v1\": {\"object_extraction_threshold\": %.4f, "
                "\"object_merge_radius\": %d, \"foreground_merge_radius\": %d, "
                "\"tracking_overlap_threshold\": %.4f, \"tracking_mode\": %d, "
                "\"tracking_ttl\": %d},\n",
                (double)c.object_extraction_threshold, c.object_merge_radius,
                c.foreground_merge_radius, (double)c.tracking_overlap_threshold,
                c.tracking_mode, c.tracking_ttl);
        } else {
            js += "    \"object_extraction_v1\": null,\n";
        }
        if (job.has_fall_cfg) {
            // Split across several AppendF calls: AppendF's internal buffer
            // (see its definition above) is 512 bytes, and this struct alone
            // has 31 fields -- one single call silently truncates mid-string,
            // corrupting the rest of the JSON document after it.
            const FallDetection_v3& c = job.cfg_fall;
            AppendF(js,
                "    \"fall_detection_v3\": {\"fall_movement_threshold\": %.4f, "
                "\"fall_strong_threshold\": %.4f, \"safe_area_ratio_threshold\": %.4f, "
                "\"fall_acceleration_threshold\": %.4f, \"fall_window_size\": %d, "
                "\"fall_duration\": %d, \"enable_face_detection\": %s, ",
                (double)c.fall_movement_threshold, (double)c.fall_strong_threshold,
                (double)c.safe_area_ratio_threshold, (double)c.fall_acceleration_threshold,
                c.fall_window_size, c.fall_duration,
                c.enable_face_detection ? "true" : "false");
            AppendF(js,
                "\"face_detect_interval_frames\": %d, \"enable_save_bg_mask\": %s, "
                "\"bg_init_start_frame\": %d, \"bg_init_end_frame\": %d, "
                "\"bg_diff_threshold\": %d, \"bg_update_interval_frames\": %d, "
                "\"bg_update_alpha\": %.4f, ",
                c.face_detect_interval_frames, c.enable_save_bg_mask ? "true" : "false",
                c.bg_init_start_frame, c.bg_init_end_frame, c.bg_diff_threshold,
                c.bg_update_interval_frames, (double)c.bg_update_alpha);
            AppendF(js,
                "\"fall_acceleration_upper_threshold\": %.4f, "
                "\"fall_acceleration_lower_threshold\": %.4f, "
                "\"post_fall_distance_threshold\": %.4f, \"post_fall_check_frames\": %d, "
                "\"enable_bed_exit_verification\": %s, "
                "\"enable_block_shrink_verification\": %s, ",
                (double)c.fall_acceleration_upper_threshold,
                (double)c.fall_acceleration_lower_threshold,
                (double)c.post_fall_distance_threshold, c.post_fall_check_frames,
                c.enable_bed_exit_verification ? "true" : "false",
                c.enable_block_shrink_verification ? "true" : "false");
            AppendF(js,
                "\"bed_update_alpha_multiplier\": %.4f, \"opt_flow_frame_distance\": %d, "
                "\"perspective_point_x\": %d, \"perspective_point_y\": %d, "
                "\"min_trigger_area\": %d, \"bed_pixel_ratio_threshold\": %.4f, "
                "\"momentum_calc_type\": %d, ",
                (double)c.bed_update_alpha_multiplier, c.opt_flow_frame_distance,
                c.perspective_point_x, c.perspective_point_y, c.min_trigger_area,
                (double)c.bed_pixel_ratio_threshold, c.momentum_calc_type);
            AppendF(js,
                "\"enable_post_bed_exit_threshold\": %s, "
                "\"post_bed_exit_threshold_multiplier\": %.4f, "
                "\"projection_use_foreground\": %s, \"enable_edge_drop_filter\": %s, "
                "\"enable_fall_and_bed_exit\": %s},\n",
                c.enable_post_bed_exit_threshold ? "true" : "false",
                (double)c.post_bed_exit_threshold_multiplier,
                c.projection_use_foreground ? "true" : "false",
                c.enable_edge_drop_filter ? "true" : "false",
                c.enable_fall_and_bed_exit ? "true" : "false");
        } else {
            js += "    \"fall_detection_v3\": null,\n";
        }
        if (job.has_image_cfg) {
            const ImageRelated_v1& c = job.cfg_image;
            AppendF(js,
                "    \"image_related_v1\": {\"expected_frame_interval_ms\": %d, "
                "\"frame_interval_tolerance_ms\": %d, \"enable_draw_bg_noise\": %s, "
                "\"enable_save_images\": %s, ",
                c.expected_frame_interval_ms, c.frame_interval_tolerance_ms,
                c.enable_draw_bg_noise ? "true" : "false",
                c.enable_save_images ? "true" : "false");
            // save_image_path is caller-controlled and unbounded in length;
            // give it its own call (with a defensive %.400s cap) rather than
            // risk overflowing AppendF's 512-byte buffer together with the
            // other fields above.
            AppendF(js, "\"save_image_path\": \"%.400s\"},\n", c.save_image_path.c_str());
        } else {
            js += "    \"image_related_v1\": null,\n";
        }
        AppendF(js,
            "    \"event_recording_v1\": {\"enable\": %s, \"pre_frames\": %d, "
            "\"post_frames\": %d}\n",
            job.event_recording_enabled ? "true" : "false",
            job.event_recording_pre_frames, job.event_recording_post_frames);
        js += "  },\n";
        uint64_t proc_sum = 0;
        uint32_t proc_max = 0;
        for (size_t i = 0; i < frame_proc_us.size(); ++i) {
            proc_sum += frame_proc_us[i];
            if (frame_proc_us[i] > proc_max) proc_max = frame_proc_us[i];
        }
        AppendF(js, "  \"process_time_us_avg\": %llu,\n  \"process_time_us_max\": %u,\n",
                (unsigned long long)(frame_proc_us.empty() ? 0 : proc_sum / frame_proc_us.size()),
                proc_max);
        js += "  \"frame_timestamps_ms\": [";
        for (size_t i = 0; i < frame_ts.size(); ++i) {
            AppendF(js, i ? ",%llu" : "%llu", (unsigned long long)frame_ts[i]);
        }
        js += "],\n  \"frame_process_time_us\": [";
        for (size_t i = 0; i < frame_proc_us.size(); ++i) {
            AppendF(js, i ? ",%u" : "%u", frame_proc_us[i]);
        }
        js += "],\n  \"triggers\": [";
        for (size_t i = 0; i < job.triggers.size(); ++i) {
            const Trigger& t = job.triggers[i];
            const char* ttype = t.is_self_test ? "self_test"
                                : (t.is_fall ? (t.is_bed_exit ? "fall+bed_exit" : "fall") : "bed_exit");
            AppendF(js, "%s\n    {\"frame_index\": %d, \"seq\": %llu, \"timestamp_ms\": %llu,"
                        " \"type\": \"%s\", \"confidence\": %.4f}",
                    i ? "," : "", t.frame_index, (unsigned long long)t.seq,
                    (unsigned long long)t.timestamp_ms, ttype, (double)t.confidence);
        }
        js += "\n  ]\n}\n";

        // tmp + rename so a power cut / full SD never leaves a half-written
        // (unparseable) .meta.json behind.
        std::string meta_path = event_dir_ + "/" + base_name + ".meta.json";
        std::string meta_tmp = event_dir_ + "/.tmp_" + base_name + ".meta.json";
        int mfd = open(meta_tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        bool meta_ok = false;
        if (mfd >= 0) {
            meta_ok = WriteAll(mfd, js.data(), js.size());
            if (meta_ok) meta_ok = (fsync(mfd) == 0);
            close(mfd);
            if (meta_ok) meta_ok = (rename(meta_tmp.c_str(), meta_path.c_str()) == 0);
            if (!meta_ok) unlink(meta_tmp.c_str());
        }
        if (!meta_ok) {
            LogF("[EventRecorder] ERROR: write meta failed (SD full?): %s : %s",
                 meta_path.c_str(), strerror(errno));
        }
    }

    // --- 5. Append one line to the global JSONL event log (snprintf only) ---
    {
        std::string line;
        line.reserve(512);
        AppendF(line, "{\"datetime\":\"%s\",", time_human);
        if (raw_saved) AppendF(line, "\"file\":\"%s\",", raw_name.c_str());
        else           line += "\"file\":null,";
        if (bg_saved) AppendF(line, "\"bg_file\":\"%s\",", bg_name.c_str());
        else          line += "\"bg_file\":null,";
        uint64_t proc_sum = 0;
        uint32_t proc_max = 0;
        for (size_t i = 0; i < frame_proc_us.size(); ++i) {
            proc_sum += frame_proc_us[i];
            if (frame_proc_us[i] > proc_max) proc_max = frame_proc_us[i];
        }
        AppendF(line, "\"event_type\":\"%s\",\"first_trigger_frame\":%d,"
                      "\"event_timestamp_ms\":%llu,\"trigger_count\":%llu,"
                      "\"max_confidence\":%.4f,\"total_frames\":%llu,"
                      "\"process_time_us_avg\":%llu,\"process_time_us_max\":%u,"
                      "\"complete\":%s}\n",
                event_type, event_frame_index,
                (unsigned long long)(job.triggers.empty() ? 0 : job.triggers.front().timestamp_ms),
                (unsigned long long)job.triggers.size(),
                (double)max_conf, (unsigned long long)written,
                (unsigned long long)(frame_proc_us.empty() ? 0 : proc_sum / frame_proc_us.size()),
                proc_max,
                job.complete ? "true" : "false");

        std::string log_path = event_dir_ + "/" + kLogFileName;
        int lfd = open(log_path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
        bool log_ok = false;
        if (lfd >= 0) {
            log_ok = WriteAll(lfd, line.data(), line.size());
            if (log_ok) log_ok = (fsync(lfd) == 0);
            close(lfd);
        }
        if (!log_ok) {
            LogF("[EventRecorder] ERROR: append %s failed (SD full?): %s",
                 log_path.c_str(), strerror(errno));
        }
    }

    LogF("[EventRecorder]%s%s (%llu frames, type=%s, triggers=%llu)",
         started_by_self_test ? "[SELF-TEST] TEST recording saved: " : " Saved ",
         raw_saved ? raw_final.c_str() : "(json-only, no raw file)",
         (unsigned long long)written, event_type,
         (unsigned long long)job.triggers.size());
}

void EventRecorder::Shutdown() {
    // Safe to call from any thread (Release() / destructor), even while the
    // frame-feeding thread is inside PushFrame: api_mtx_ serializes us behind
    // it, and once shutdown_ is set every later entry point is a no-op, so the
    // ring can never be unmapped under an in-flight memcpy.
    std::lock_guard<std::mutex> api_lk(api_mtx_);
    if (shutdown_ && !writer_running_ && !map_) return;   // idempotent
    shutdown_ = true;

    // Flush an in-flight capture as a partial recording before stopping.
    if (capturing_ && ring_ready_ && !failed_) {
        LogF("[EventRecorder] Shutdown with capture in progress, saving partial event.");
        QueueFinalize(false);
    }
    if (writer_running_) {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stop_ = true;
        }
        cv_.notify_all();
#if EVENT_RECORDER_ASYNC_SPOOL
        // Join the spooler FIRST: it drains all remaining queued frames into
        // the ring (advancing spooled_seq_) and then exits on stop_. Only once
        // it is done can the writer's spooled_seq_ wait complete and read a
        // fully-populated window. Both threads only touch mtx_, never api_mtx_,
        // so joining while holding api_mtx_ cannot deadlock.
        if (spooler_running_) {
            if (spooler_.joinable()) spooler_.join();
            spooler_running_ = false;
            cv_.notify_all();   // wake the writer's spooled_seq_ wait if pending
        }
#endif
        // Drains queued finalize jobs; the writer only touches mtx_, never
        // api_mtx_, so joining while holding api_mtx_ cannot deadlock.
        if (writer_.joinable()) writer_.join();
        writer_running_ = false;
    }
    CloseRing();
    enabled_ = false;
    capturing_ = false;
}

} // namespace VisionSDK
