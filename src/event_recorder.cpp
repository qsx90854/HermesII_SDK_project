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

namespace VisionSDK {

namespace {

const char* kBasePath = "/mnt/mmcblk1p1";      // SD card mount point on the edge board
const char* kFallbackDir = "./event_record";   // used when SD path is not writable (PC test)
const char* kRingFileName = "hermes_frame_ring.dat";
const char* kLogFileName = "event_record.jsonl";
const uint32_t kRingMagic = 0x48524731;        // "HRG1"
const uint32_t kRingVersion = 2;               // v2: slot index gained process_time_us
const uint32_t kMarginSlots = 64;              // extra slots protecting the copy window
const size_t kPageSize = 4096;
const uint64_t kInvalidSeq = UINT64_MAX;
const int kResidentSlots = 32;                 // madvise(DONTNEED) slots older than this

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
    t.timestamp_ms = index_[(uint32_t)(t.seq % capacity_)].timestamp_ms;
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
    t.timestamp_ms = index_[(uint32_t)(t.seq % capacity_)].timestamp_ms;
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
    uint32_t slot = (uint32_t)((write_seq_ - 1) % capacity_);
    index_[slot].process_time_us =
        (process_time_us > UINT32_MAX) ? UINT32_MAX : (uint32_t)process_time_us;
}

void EventRecorder::SetBedRegion(const std::vector<std::pair<int, int>>& points) {
    std::lock_guard<std::mutex> api_lk(api_mtx_);
    bed_region_ = points;
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
        }

        RunFinalize(job);

        {
            std::lock_guard<std::mutex> lk(mtx_);
            copy_cursor_ = kInvalidSeq;
        }
    }
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

    // --- 1. Frame window: ring -> .raw (write to .tmp, fsync, rename) ---
    int out = open(raw_tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out < 0) {
        LogF("[EventRecorder] ERROR: finalize open %s failed: %s",
             raw_tmp.c_str(), strerror(errno));
        return;
    }

    uint64_t written = 0, missing = 0;
    uint64_t first_ts = 0, last_ts = 0;
    // Per-frame input timestamps (from SetInputMemory) and Detect() durations,
    // same order as the frames written into the .raw file.
    std::vector<uint64_t> frame_ts;
    std::vector<uint32_t> frame_proc_us;
    frame_ts.reserve((size_t)(job.last_seq - job.first_seq + 1));
    frame_proc_us.reserve((size_t)(job.last_seq - job.first_seq + 1));
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
            frame_proc_us.push_back(index_[slot].process_time_us);
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

    // --- 2. Background snapshot taken at the event start ---
    bool bg_saved = false;
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
                             "\"total_fg\":%d,\"objects\":[",
                        (unsigned long long)fa.seq,
                        (long long)(fa.seq - job.first_seq),
                        (unsigned long long)fa.timestamp_ms, fa.total_fg_pixels);
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
        AppendF(js, "{\n  \"file\": \"%s\",\n", raw_name.c_str());
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
        AppendF(line, "{\"datetime\":\"%s\",\"file\":\"%s\",", time_human, raw_name.c_str());
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
         raw_final.c_str(), (unsigned long long)written, event_type,
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
        cv_.notify_one();
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
