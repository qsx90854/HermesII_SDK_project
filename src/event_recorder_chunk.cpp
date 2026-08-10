// EventRecorderChunk: chunk-storage backend for event recording.
// See event_recorder_chunk.h and 事件錄影_chunk儲存架構設計.md.
//
// Every frame is appended (once) into rolling chunk files; an event only writes
// a small .meta.json referencing the chunk range (no frame copy, no ring
// read-back), so there is no finalize I/O spike to starve the spooler. A
// throttled GC deletes only rolling-buffer chunks that belong to no retained
// event; real events are never auto-deleted (SD full => new events logged only).

#include "event_recorder_chunk.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/statvfs.h>
#include <dirent.h>

#include <cerrno>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>

namespace VisionSDK {
namespace {

const char* kBasePath = "/mnt/mmcblk1p1";
const char* kFallbackBase = ".";
const char* kEventDirName = "event_record";
const char* kFramesDirName = "frames";
const char* kLogFileName = "event_record.jsonl";
const char* kRetentionName = "retention.idx";

const uint32_t kGcIntervalMs = 5000;       // GC runs at most this often
const int kGcMaxDeletePerPass = 4;         // throttle: unlink at most N chunks per pass
const size_t kMaxSpoolFrames = 32;         // RAM queue bound (drop, don't block)

bool WriteAll(int fd, const void* buf, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    while (len > 0) {
        ssize_t n = write(fd, p, len);
        if (n < 0) { if (errno == EINTR) continue; return false; }
        p += n; len -= (size_t)n;
    }
    return true;
}

bool PWriteAll(int fd, const void* buf, size_t len, off_t off) {
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    while (len > 0) {
        ssize_t n = pwrite(fd, p, len, off);
        if (n < 0) { if (errno == EINTR) continue; return false; }
        p += n; len -= (size_t)n; off += n;
    }
    return true;
}

__attribute__((format(printf, 2, 3)))
void AppendF(std::string& out, const char* fmt, ...) {
    char buf[512];
    va_list ap; va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n > 0) out.append(buf, (size_t)((n < (int)sizeof(buf)) ? n : (int)sizeof(buf) - 1));
}

__attribute__((format(printf, 1, 2)))
void LogF(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    vfprintf(stdout, fmt, ap);
    va_end(ap);
    fputc('\n', stdout);
}

void FormatTime(time_t t, char* out, size_t out_len, const char* fmt) {
    struct tm tmv;
    localtime_r(&t, &tmv);
    strftime(out, out_len, fmt, &tmv);
}

bool MakeDir(const std::string& d) {
    return (mkdir(d.c_str(), 0755) == 0) || errno == EEXIST;
}

}  // namespace

EventRecorderChunk::EventRecorderChunk() {}
EventRecorderChunk::~EventRecorderChunk() { Shutdown(); }

void EventRecorderChunk::Configure(bool enable, int pre_frames, int post_frames, bool store_raw) {
    std::lock_guard<std::mutex> api_lk(api_mtx_);
    if (enable) shutdown_ = false;
    enabled_ = enable;
    if (pre_frames >= 0) pre_frames_ = pre_frames;
    if (post_frames >= 0) post_frames_ = post_frames;
    store_raw_ = store_raw;
    LogF("[EventRecorderChunk] Configure: enable=%d pre=%d post=%d store_raw=%d chunk_frames=%d",
         (int)enable, pre_frames_, post_frames_, (int)store_raw_, chunk_frames_);
    if (enabled_) StartThreadsLocked();
}

void EventRecorderChunk::StartThreadsLocked() {
    if (writer_running_) return;
    stop_ = false;
    writer_ = std::thread(&EventRecorderChunk::WriterLoop, this);
    writer_running_ = true;
    spooler_ = std::thread(&EventRecorderChunk::SpoolerLoop, this);
    spooler_running_ = true;
}

void EventRecorderChunk::Fail(const std::string& why) {
    failed_ = true;
    LogF("[EventRecorderChunk] DISABLED: %s (errno=%d %s)", why.c_str(), errno, strerror(errno));
}

std::string EventRecorderChunk::ChunkPath(uint64_t chunk_id) const {
    char sub[32], name[64];
    snprintf(sub, sizeof(sub), "%llu", (unsigned long long)(chunk_id / 1000));
    snprintf(name, sizeof(name), "chunk_%08llu.raw", (unsigned long long)chunk_id);
    return frames_dir_ + "/" + sub + "/" + name;
}

// Each SDK run gets its own incrementing session folder under event_record/ so
// nothing mixes across restarts. On startup we scan for the highest numeric
// folder N and create N+1; everything this run produces (chunks/, meta.json,
// retention.idx, event_record.jsonl) goes inside it. A fresh folder means we
// start seq at 0 with no existing chunks / retained events -- no scan/resume.
bool EventRecorderChunk::InitStorage(int width, int height, int channels) {
    struct stat st;
    if (stat(kBasePath, &st) == 0 && S_ISDIR(st.st_mode)) base_dir_ = kBasePath;
    else base_dir_ = kFallbackBase;
    event_dir_ = base_dir_ + "/" + kEventDirName;
    if (!MakeDir(event_dir_)) { Fail("mkdir event_record failed"); return false; }

    // Find the highest existing numeric session folder -> new session = max + 1.
    long long max_session = 0;
    DIR* ed = opendir(event_dir_.c_str());
    if (ed) {
        struct dirent* e;
        while ((e = readdir(ed)) != nullptr) {
            const char* n = e->d_name;
            if (n[0] < '0' || n[0] > '9') continue;   // skip ".", "..", non-numeric
            bool all_digit = true;
            for (const char* p = n; *p; ++p) if (*p < '0' || *p > '9') { all_digit = false; break; }
            if (!all_digit) continue;
            long long v = atoll(n);
            if (v > max_session) max_session = v;
        }
        closedir(ed);
    }
    session_id_ = max_session + 1;   // 1 if none yet

    char sid[24];
    snprintf(sid, sizeof(sid), "%lld", session_id_);
    session_dir_ = event_dir_ + "/" + sid;
    frames_dir_ = session_dir_ + "/" + kFramesDirName;
    retention_path_ = session_dir_ + "/" + kRetentionName;
    if (!MakeDir(session_dir_) || !MakeDir(frames_dir_)) { Fail("mkdir session dir failed"); return false; }

    width_ = width; height_ = height; channels_ = channels;
    frame_size_ = (uint32_t)(width * height * channels);
    if (frame_size_ == 0) { Fail("zero frame size"); return false; }

    // RAM ring of analysis snapshots covering the rolling window (pre+post+2 chunks).
    analysis_cap_ = (uint32_t)(pre_frames_ + 1 + post_frames_ + 2 * chunk_frames_);
    analysis_ring_.assign(analysis_cap_, FrameAnalysis());

    // Fresh session folder: start clean.
    write_seq_ = 0;
    spooled_seq_ = 0;

    ready_ = true;
    LogF("[EventRecorderChunk] storage ready: session #%lld  dir=%s  frame_size=%u",
         session_id_, session_dir_.c_str(), frame_size_);
    return true;
}

void EventRecorderChunk::PushFrame(const Image& img) {
    std::lock_guard<std::mutex> api_lk(api_mtx_);
    if (shutdown_ || !enabled_ || failed_) return;
    if (!ready_) {
        if (!InitStorage(img.width, img.height, img.channels)) return;
        // Lazy-start the spooler/writer even when nobody called Configure()
        // (recorder is default-on). Without this the frames queue but no thread
        // drains them -> nothing written. (Idempotent; no-op if already running.)
        StartThreadsLocked();
    }
    if ((uint32_t)(img.width * img.height * img.channels) != frame_size_) return;  // geometry changed

    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (spool_q_.size() >= kMaxSpoolFrames) {
            // Spooler behind: drop this frame (leaves a hole in its chunk) rather
            // than grow RAM. seq is NOT consumed -- retried on the next frame.
            dropped_frames_++;
            last_push_written_ = false;
            return;
        }
        SpoolFrame sf;
        sf.seq = write_seq_;
        sf.timestamp_ms = img.timestamp;
        sf.data.assign(reinterpret_cast<const uint8_t*>(img.data),
                       reinterpret_cast<const uint8_t*>(img.data) + frame_size_);
        spool_q_.push_back(std::move(sf));
    }
    cv_.notify_all();
    last_push_written_ = true;
    last_frame_ts_ = img.timestamp;
    write_seq_++;

    // Finalize the window ONE frame after last_seq (event_seq+post): the last
    // window frame's analysis snapshot is recorded (RecordFrameAnalysis) only
    // AFTER this PushFrame returns, so triggering exactly at last_seq would miss
    // it and under-count by 1. Waiting one more frame guarantees it is present.
    if (capturing_ && (write_seq_ - 1) > event_seq_ + (uint64_t)post_frames_) {
        QueueMeta(true);
    }
}

void EventRecorderChunk::SpoolerLoop() {
    for (;;) {
        SpoolFrame sf;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [this] { return stop_ || !spool_q_.empty(); });
            if (spool_q_.empty()) { if (stop_) break; else continue; }
            sf = std::move(spool_q_.front());
            spool_q_.pop_front();
        }

        uint64_t chunk_id = sf.seq / (uint64_t)chunk_frames_;
        if (chunk_id != cur_chunk_id_) {
            if (cur_chunk_fd_ >= 0) {
#ifdef POSIX_FADV_DONTNEED
                posix_fadvise(cur_chunk_fd_, 0, 0, POSIX_FADV_DONTNEED);  // drop old chunk's page cache
#endif
                close(cur_chunk_fd_);
                cur_chunk_fd_ = -1;
            }
            // ensure subdir <chunk_id/1000>
            char sub[32];
            snprintf(sub, sizeof(sub), "%llu", (unsigned long long)(chunk_id / 1000));
            MakeDir(frames_dir_ + "/" + sub);
            std::string path = ChunkPath(chunk_id);
            cur_chunk_fd_ = open(path.c_str(), O_WRONLY | O_CREAT, 0644);
            cur_chunk_id_ = chunk_id;
            if (cur_chunk_fd_ >= 0) {
                std::lock_guard<std::mutex> lk(mtx_);
                existing_chunks_.insert(chunk_id);
            }
        }
        if (cur_chunk_fd_ >= 0) {
            off_t off = (off_t)(sf.seq % (uint64_t)chunk_frames_) * (off_t)frame_size_;
            if (!PWriteAll(cur_chunk_fd_, sf.data.data(), frame_size_, off)) {
                LogF("[EventRecorderChunk] chunk write failed (SD full?): %s", strerror(errno));
            }
        }

        {
            std::lock_guard<std::mutex> lk(mtx_);
            spooled_seq_ = sf.seq + 1;
        }
        cv_.notify_all();
    }
    if (cur_chunk_fd_ >= 0) { close(cur_chunk_fd_); cur_chunk_fd_ = -1; }
}

void EventRecorderChunk::OnEvent(const VisionSDKEvent& event, std::vector<uint8_t>&& bg_image) {
    std::lock_guard<std::mutex> api_lk(api_mtx_);
    if (shutdown_ || !enabled_ || failed_ || !ready_ || write_seq_ == 0) return;

    Trigger t;
    t.seq = write_seq_ - 1;
    t.timestamp_ms = last_frame_ts_;
    t.frame_index = event.frame_index;
    t.is_fall = event.is_fall_detected;
    t.is_bed_exit = event.is_bed_exit;
    t.is_self_test = false;
    t.confidence = event.confidence;

    if (capturing_) { triggers_.push_back(t); return; }

    capturing_ = true;
    event_seq_ = t.seq;
    triggers_.clear();
    triggers_.push_back(t);
    event_bg_ = std::move(bg_image);
    LogF("[EventRecorderChunk] Event start: type=%s frame_index=%d seq=%llu",
         t.is_fall ? (t.is_bed_exit ? "fall+bed_exit" : "fall") : "bed_exit",
         t.frame_index, (unsigned long long)t.seq);
    if (post_frames_ == 0) QueueMeta(true);
}

void EventRecorderChunk::TriggerSelfTest(std::vector<uint8_t>&& bg_image) {
    std::lock_guard<std::mutex> api_lk(api_mtx_);
    if (shutdown_ || !enabled_ || failed_ || !ready_ || write_seq_ == 0) return;
    if (capturing_) return;
    Trigger t;
    t.seq = write_seq_ - 1;
    t.timestamp_ms = last_frame_ts_;
    t.frame_index = (int)t.seq;
    t.is_fall = false; t.is_bed_exit = false; t.is_self_test = true; t.confidence = 1.0f;
    capturing_ = true;
    event_seq_ = t.seq;
    triggers_.clear();
    triggers_.push_back(t);
    event_bg_ = std::move(bg_image);
    LogF("[EventRecorderChunk][SELF-TEST] recording started at seq=%llu (pre=%d post=%d)",
         (unsigned long long)t.seq, pre_frames_, post_frames_);
    if (post_frames_ == 0) QueueMeta(true);
}

void EventRecorderChunk::RecordProcessTime(uint64_t process_time_us) {
    std::lock_guard<std::mutex> api_lk(api_mtx_);
    if (shutdown_ || !enabled_ || failed_ || !ready_ || write_seq_ == 0) return;
    if (!last_push_written_ || analysis_cap_ == 0) return;
    uint32_t slot = (uint32_t)((write_seq_ - 1) % analysis_cap_);
    if (analysis_ring_[slot].seq == write_seq_ - 1)
        analysis_ring_[slot].process_time_us =
            (process_time_us > UINT32_MAX) ? UINT32_MAX : (uint32_t)process_time_us;
}

void EventRecorderChunk::RecordFrameAnalysis(FrameAnalysis&& fa) {
    std::lock_guard<std::mutex> api_lk(api_mtx_);
    if (shutdown_ || !enabled_ || failed_ || !ready_ || write_seq_ == 0) return;
    if (!last_push_written_ || analysis_cap_ == 0) return;
    fa.seq = write_seq_ - 1;
    analysis_ring_[(uint32_t)((write_seq_ - 1) % analysis_cap_)] = std::move(fa);
}

void EventRecorderChunk::SetBedRegion(const std::vector<std::pair<int, int>>& points) {
    std::lock_guard<std::mutex> api_lk(api_mtx_);
    bed_region_ = points;
}
void EventRecorderChunk::SetMotionConfig(const MotionEstimation_v1& c) {
    std::lock_guard<std::mutex> api_lk(api_mtx_); cfg_motion_ = c; has_motion_cfg_ = true;
}
void EventRecorderChunk::SetObjectConfig(const ObjectExtraction_v1& c) {
    std::lock_guard<std::mutex> api_lk(api_mtx_); cfg_object_ = c; has_object_cfg_ = true;
}
void EventRecorderChunk::SetFallConfig(const FallDetection_v3& c) {
    std::lock_guard<std::mutex> api_lk(api_mtx_); cfg_fall_ = c; has_fall_cfg_ = true;
}
void EventRecorderChunk::SetImageConfig(const ImageRelated_v1& c) {
    std::lock_guard<std::mutex> api_lk(api_mtx_); cfg_image_ = c; has_image_cfg_ = true;
}

bool EventRecorderChunk::SpaceForOneEvent() const {
    struct statvfs vfs;
    if (statvfs(base_dir_.c_str(), &vfs) != 0) return true;   // unknown -> allow
    uint64_t free_bytes = (uint64_t)vfs.f_bavail * (uint64_t)vfs.f_frsize;
    uint64_t window = (uint64_t)(pre_frames_ + 1 + post_frames_) * frame_size_;
    uint64_t rolling_reserve = (uint64_t)(pre_frames_ + post_frames_ + 2 * chunk_frames_) * frame_size_;
    return free_bytes >= (window + rolling_reserve + window /*safety*/);
}

void EventRecorderChunk::QueueMeta(bool complete) {
    MetaJob job;
    job.event_seq = event_seq_;
    job.first_seq = (event_seq_ >= (uint64_t)pre_frames_) ? event_seq_ - pre_frames_ : 0;
    uint64_t latest = write_seq_ - 1;
    uint64_t want_last = event_seq_ + (uint64_t)post_frames_;
    job.last_seq = (want_last <= latest) ? want_last : latest;
    job.complete = complete;
    // Retain (keep chunks) only if store_raw AND there is room. Otherwise the
    // event is logged meta-only and its frames age out with the rolling buffer.
    job.stored = store_raw_ && SpaceForOneEvent();
    job.not_stored_reason = job.stored ? "" : (!store_raw_ ? "store_raw_off" : "sd_full");
    job.triggers = triggers_;
    job.bg_image = std::move(event_bg_);
    job.bed_region = bed_region_;
    job.cfg_motion = cfg_motion_; job.cfg_object = cfg_object_;
    job.cfg_fall = cfg_fall_;     job.cfg_image = cfg_image_;
    job.has_motion_cfg = has_motion_cfg_; job.has_object_cfg = has_object_cfg_;
    job.has_fall_cfg = has_fall_cfg_;     job.has_image_cfg = has_image_cfg_;
    job.cfg_pre_frames = pre_frames_; job.cfg_post_frames = post_frames_;

    if (analysis_cap_ > 0) {
        job.analysis.reserve((size_t)(job.last_seq - job.first_seq + 1));
        for (uint64_t seq = job.first_seq; seq <= job.last_seq; ++seq) {
            const FrameAnalysis& fa = analysis_ring_[(uint32_t)(seq % analysis_cap_)];
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
    cv_.notify_all();
}

void EventRecorderChunk::WriteMeta(const MetaJob& job) {
    time_t now = time(nullptr);
    char tc[32], th[32];
    FormatTime(now, tc, sizeof(tc), "%Y%m%d_%H%M%S");
    FormatTime(now, th, sizeof(th), "%Y-%m-%d %H:%M:%S");
    int event_frame_index = job.triggers.empty() ? -1 : job.triggers.front().frame_index;
    bool self_test = !job.triggers.empty() && job.triggers.front().is_self_test;
    char base_name[96];
    snprintf(base_name, sizeof(base_name), "evt_%s%s_f%d",
             self_test ? "selftest_" : "", tc, event_frame_index);

    uint64_t first_chunk = job.first_seq / (uint64_t)chunk_frames_;
    uint64_t last_chunk = job.last_seq / (uint64_t)chunk_frames_;
    uint64_t window = (job.last_seq >= job.first_seq) ? (job.last_seq - job.first_seq + 1) : 0;

    // --- background snapshot (small, one write) -- only for stored events ---
    bool bg_saved = false;
    std::string bg_name = std::string(base_name) + "_bg.raw";
    if (job.stored && store_raw_ && !job.bg_image.empty()) {
        std::string bg_tmp = session_dir_ + "/.tmp_" + bg_name;
        int bfd = open(bg_tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (bfd >= 0) {
            bg_saved = WriteAll(bfd, job.bg_image.data(), job.bg_image.size()) && (fsync(bfd) == 0);
            close(bfd);
            if (bg_saved) bg_saved = (rename(bg_tmp.c_str(), (session_dir_ + "/" + bg_name).c_str()) == 0);
            if (!bg_saved) unlink(bg_tmp.c_str());
        }
    }

    // --- analysis sidecar (RAM-sourced) ---
    std::string analysis_name = std::string(base_name) + ".analysis.jsonl";
    bool analysis_saved = false;
    if (!job.analysis.empty()) {
        std::string a_tmp = session_dir_ + "/.tmp_" + analysis_name;
        int afd = open(a_tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (afd >= 0) {
            bool ok = true;
            std::string buf; buf.reserve(8192);
            const FrameAnalysis& fr = job.analysis.front();
            AppendF(buf, "{\"type\":\"header\",\"grid_cols\":%d,\"grid_rows\":%d,"
                         "\"block_size\":%d,\"first_seq\":%llu,\"event_seq\":%llu,"
                         "\"last_seq\":%llu,\"frames\":%llu}\n",
                    fr.grid_cols, fr.grid_rows, fr.block_size,
                    (unsigned long long)job.first_seq, (unsigned long long)job.event_seq,
                    (unsigned long long)job.last_seq, (unsigned long long)job.analysis.size());
            ok = WriteAll(afd, buf.data(), buf.size());
            for (size_t f = 0; ok && f < job.analysis.size(); ++f) {
                const FrameAnalysis& fa = job.analysis[f];
                buf.clear();
                AppendF(buf, "{\"seq\":%llu,\"frame_in_file\":%lld,\"ts\":%llu,\"proc_us\":%u,"
                             "\"total_fg\":%d,\"objects\":[",
                        (unsigned long long)fa.seq, (long long)(fa.seq - job.first_seq),
                        (unsigned long long)fa.timestamp_ms, fa.process_time_us, fa.total_fg_pixels);
                for (size_t o = 0; o < fa.objects.size(); ++o) {
                    const FrameAnalysisObject& ob = fa.objects[o];
                    AppendF(buf, "%s{\"id\":%d,\"cx\":%.2f,\"cy\":%.2f,\"dx\":%.2f,\"dy\":%.2f,"
                                 "\"strength\":%.2f,\"accel\":%.2f,\"pixels\":%d,\"safe_ratio\":%.3f,"
                                 "\"dir_var\":%.3f,\"obs\":%d,\"is_fall\":%d,\"is_bed_exit\":%d,\"fg_area\":%d,\"coasting\":%d,\"blocks\":[",
                            o ? "," : "", ob.id, (double)ob.cx, (double)ob.cy, (double)ob.dx, (double)ob.dy,
                            (double)ob.strength, (double)ob.acceleration, ob.pixel_count,
                            (double)ob.safe_area_ratio, (double)ob.direction_variance,
                            ob.in_observation ? 1 : 0, ob.is_fall ? 1 : 0, ob.is_bed_exit ? 1 : 0, ob.fg_area, ob.is_coasting ? 1 : 0);
                    for (size_t b = 0; b < ob.blocks.size(); ++b)
                        AppendF(buf, b ? ",%u" : "%u", (unsigned)ob.blocks[b]);
                    buf += "]}";
                }
                buf += "]}\n";
                if (!WriteAll(afd, buf.data(), buf.size())) { ok = false; break; }
            }
            if (ok) ok = (fsync(afd) == 0);
            close(afd);
            if (ok) ok = (rename(a_tmp.c_str(), (session_dir_ + "/" + analysis_name).c_str()) == 0);
            analysis_saved = ok;
            if (!ok) unlink(a_tmp.c_str());
        }
    }

    // per-frame timestamps/proc from analysis (no chunk read at write time)
    std::vector<uint64_t> frame_ts; std::vector<uint32_t> proc;
    uint64_t first_ts = 0, last_ts = 0;
    for (const FrameAnalysis& fa : job.analysis) { frame_ts.push_back(fa.timestamp_ms); proc.push_back(fa.process_time_us); }
    if (!frame_ts.empty()) { first_ts = frame_ts.front(); last_ts = frame_ts.back(); }
    uint64_t present = frame_ts.size();
    uint64_t missing = (window >= present) ? (window - present) : 0;

    bool any_fall = false, any_bed = false;
    float max_conf = 0.f;
    for (const Trigger& t : job.triggers) {
        any_fall |= t.is_fall; any_bed |= t.is_bed_exit;
        if (t.confidence > max_conf) max_conf = t.confidence;
    }
    const char* event_type = (any_fall && any_bed) ? "fall+bed_exit"
                             : (any_fall ? "fall" : (any_bed ? "bed_exit" : "self_test"));

    // --- meta.json ---
    std::string meta_name = std::string(base_name) + ".meta.json";
    {
        std::string js; js.reserve(4096 + frame_ts.size() * 14 + job.triggers.size() * 110);
        js += "{\n";
        AppendF(js, "  \"storage\": \"chunked\",\n");
        AppendF(js, "  \"chunk_frames\": %d,\n  \"frame_size\": %u,\n", chunk_frames_, frame_size_);
        if (job.stored) {
            AppendF(js, "  \"first_chunk\": %llu,\n  \"last_chunk\": %llu,\n  \"frames_dir\": \"%s\",\n",
                    (unsigned long long)first_chunk, (unsigned long long)last_chunk, kFramesDirName);
        } else {
            js += "  \"first_chunk\": null,\n  \"last_chunk\": null,\n  \"frames_dir\": null,\n";
        }
        if (bg_saved) AppendF(js, "  \"bg_file\": \"%s\",\n", bg_name.c_str()); else js += "  \"bg_file\": null,\n";
        if (analysis_saved) {
            AppendF(js, "  \"analysis_file\": \"%s\",\n  \"analysis_frames\": %llu,\n",
                    analysis_name.c_str(), (unsigned long long)job.analysis.size());
        } else js += "  \"analysis_file\": null,\n";
        AppendF(js, "  \"datetime\": \"%s\",\n  \"event_type\": \"%s\",\n", th, event_type);
        AppendF(js, "  \"event_frame_index\": %d,\n  \"event_timestamp_ms\": %llu,\n",
                event_frame_index, (unsigned long long)(job.triggers.empty() ? 0 : job.triggers.front().timestamp_ms));
        AppendF(js, "  \"width\": %d,\n  \"height\": %d,\n  \"channels\": %d,\n", width_, height_, channels_);
        AppendF(js, "  \"first_seq\": %llu,\n  \"last_seq\": %llu,\n  \"event_seq\": %llu,\n",
                (unsigned long long)job.first_seq, (unsigned long long)job.last_seq, (unsigned long long)job.event_seq);
        AppendF(js, "  \"pre_frames\": %llu,\n  \"post_frames\": %llu,\n",
                (unsigned long long)(job.event_seq - job.first_seq),
                (unsigned long long)(job.last_seq - job.event_seq));
        AppendF(js, "  \"total_frames\": %llu,\n  \"missing_frames\": %llu,\n",
                (unsigned long long)present, (unsigned long long)missing);
        AppendF(js, "  \"complete\": %s,\n", job.complete ? "true" : "false");
        AppendF(js, "  \"stored\": %s,\n", job.stored ? "true" : "false");
        if (!job.stored) AppendF(js, "  \"not_stored_reason\": \"%s\",\n", job.not_stored_reason);
        AppendF(js, "  \"ring_dropped_frames\": %llu,\n", (unsigned long long)dropped_frames_);
        AppendF(js, "  \"first_timestamp_ms\": %llu,\n  \"last_timestamp_ms\": %llu,\n",
                (unsigned long long)first_ts, (unsigned long long)last_ts);
        js += "  \"bed_region\": [";
        for (size_t i = 0; i < job.bed_region.size(); ++i)
            AppendF(js, "%s[%d,%d]", i ? "," : "", job.bed_region[i].first, job.bed_region[i].second);
        js += "],\n";
        // config snapshot (same field layout as EventRecorder; split to fit AppendF's 512B buffer)
        js += "  \"config\": {\n";
        if (job.has_motion_cfg) {
            const MotionEstimation_v1& c = job.cfg_motion;
            AppendF(js, "    \"motion_estimation_v1\": {\"grid_cols\": %d, \"grid_rows\": %d, "
                        "\"block_size\": %d, \"search_range\": %d, \"history_size\": %d, "
                        "\"search_mode\": %d, \"block_change_threshold\": %.6f, "
                        "\"enable_block_decay\": %s, \"block_decay_frames\": %d, "
                        "\"enable_block_dilation\": %s, \"block_dilation_threshold\": %d},\n",
                    c.grid_cols, c.grid_rows, c.block_size, c.search_range, c.history_size,
                    c.search_mode, c.block_change_threshold, c.enable_block_decay ? "true" : "false",
                    c.block_decay_frames, c.enable_block_dilation ? "true" : "false", c.block_dilation_threshold);
        } else js += "    \"motion_estimation_v1\": null,\n";
        if (job.has_object_cfg) {
            const ObjectExtraction_v1& c = job.cfg_object;
            AppendF(js, "    \"object_extraction_v1\": {\"object_extraction_threshold\": %.4f, "
                        "\"object_merge_radius\": %d, \"foreground_merge_radius\": %d, "
                        "\"tracking_overlap_threshold\": %.4f, \"tracking_mode\": %d, \"tracking_ttl\": %d, ",
                    (double)c.object_extraction_threshold, c.object_merge_radius, c.foreground_merge_radius,
                    (double)c.tracking_overlap_threshold, c.tracking_mode, c.tracking_ttl);
            AppendF(js, "\"merge_overlapping_enable\": %s, \"merge_overlapping_iou\": %.4f, "
                        "\"merge_tracked_enable\": %s, \"merge_tracked_overlap\": %.4f, "
                        "\"merge_tracked_max_dist\": %.4f, ",
                    c.merge_overlapping_enable ? "true" : "false", (double)c.merge_overlapping_iou,
                    c.merge_tracked_enable ? "true" : "false", (double)c.merge_tracked_overlap,
                    (double)c.merge_tracked_max_dist);
            AppendF(js, "\"use_fg_area\": %s, \"min_trigger_fg_area\": %d, \"still_lying_fg_area\": %d, "
                        "\"use_fg_area_trigger\": %s, \"enable_kalman_predict\": %s},\n",
                    c.use_fg_area ? "true" : "false", c.min_trigger_fg_area, c.still_lying_fg_area,
                    c.use_fg_area_trigger ? "true" : "false", c.enable_kalman_predict ? "true" : "false");
        } else js += "    \"object_extraction_v1\": null,\n";
        if (job.has_fall_cfg) {
            const FallDetection_v3& c = job.cfg_fall;
            AppendF(js, "    \"fall_detection_v3\": {\"fall_movement_threshold\": %.4f, "
                        "\"fall_strong_threshold\": %.4f, \"safe_area_ratio_threshold\": %.4f, "
                        "\"fall_acceleration_threshold\": %.4f, \"fall_window_size\": %d, "
                        "\"fall_duration\": %d, \"enable_face_detection\": %s, ",
                    (double)c.fall_movement_threshold, (double)c.fall_strong_threshold,
                    (double)c.safe_area_ratio_threshold, (double)c.fall_acceleration_threshold,
                    c.fall_window_size, c.fall_duration, c.enable_face_detection ? "true" : "false");
            AppendF(js, "\"face_detect_interval_frames\": %d, \"enable_save_bg_mask\": %s, "
                        "\"bg_init_start_frame\": %d, \"bg_init_end_frame\": %d, \"bg_diff_threshold\": %d, "
                        "\"bg_update_interval_frames\": %d, \"bg_update_alpha\": %.4f, ",
                    c.face_detect_interval_frames, c.enable_save_bg_mask ? "true" : "false",
                    c.bg_init_start_frame, c.bg_init_end_frame, c.bg_diff_threshold,
                    c.bg_update_interval_frames, (double)c.bg_update_alpha);
            AppendF(js, "\"bg_protect_max_frames\": %d, \"bg_protect_min_fg\": %d, ",
                    c.bg_protect_max_frames, c.bg_protect_min_fg);
            AppendF(js, "\"fall_acceleration_upper_threshold\": %.4f, \"fall_acceleration_lower_threshold\": %.4f, "
                        "\"post_fall_distance_threshold\": %.4f, \"post_fall_check_frames\": %d, "
                        "\"enable_bed_exit_verification\": %s, \"enable_block_shrink_verification\": %s, ",
                    (double)c.fall_acceleration_upper_threshold, (double)c.fall_acceleration_lower_threshold,
                    (double)c.post_fall_distance_threshold, c.post_fall_check_frames,
                    c.enable_bed_exit_verification ? "true" : "false",
                    c.enable_block_shrink_verification ? "true" : "false");
            AppendF(js, "\"bed_update_alpha_multiplier\": %.4f, \"opt_flow_frame_distance\": %d, "
                        "\"perspective_point_x\": %d, \"perspective_point_y\": %d, \"min_trigger_area\": %d, "
                        "\"bed_pixel_ratio_threshold\": %.4f, \"momentum_calc_type\": %d, ",
                    (double)c.bed_update_alpha_multiplier, c.opt_flow_frame_distance, c.perspective_point_x,
                    c.perspective_point_y, c.min_trigger_area, (double)c.bed_pixel_ratio_threshold, c.momentum_calc_type);
            AppendF(js, "\"enable_post_bed_exit_threshold\": %s, \"post_bed_exit_threshold_multiplier\": %.4f, "
                        "\"projection_use_foreground\": %s, \"enable_edge_drop_filter\": %s, "
                        "\"enable_fall_and_bed_exit\": %s},\n",
                    c.enable_post_bed_exit_threshold ? "true" : "false", (double)c.post_bed_exit_threshold_multiplier,
                    c.projection_use_foreground ? "true" : "false", c.enable_edge_drop_filter ? "true" : "false",
                    c.enable_fall_and_bed_exit ? "true" : "false");
        } else js += "    \"fall_detection_v3\": null,\n";
        if (job.has_image_cfg) {
            const ImageRelated_v1& c = job.cfg_image;
            AppendF(js, "    \"image_related_v1\": {\"expected_frame_interval_ms\": %d, "
                        "\"frame_interval_tolerance_ms\": %d, \"enable_draw_bg_noise\": %s, \"enable_save_images\": %s, ",
                    c.expected_frame_interval_ms, c.frame_interval_tolerance_ms,
                    c.enable_draw_bg_noise ? "true" : "false", c.enable_save_images ? "true" : "false");
            AppendF(js, "\"save_image_path\": \"%.400s\"},\n", c.save_image_path.c_str());
        } else js += "    \"image_related_v1\": null,\n";
        AppendF(js, "    \"event_recording_v1\": {\"enable\": true, \"pre_frames\": %d, "
                    "\"post_frames\": %d, \"store_raw\": %s}\n", job.cfg_pre_frames, job.cfg_post_frames,
                store_raw_ ? "true" : "false");
        js += "  },\n";
        uint64_t psum = 0; uint32_t pmax = 0;
        for (uint32_t v : proc) { psum += v; if (v > pmax) pmax = v; }
        AppendF(js, "  \"process_time_us_avg\": %llu,\n  \"process_time_us_max\": %u,\n",
                (unsigned long long)(proc.empty() ? 0 : psum / proc.size()), pmax);
        js += "  \"frame_timestamps_ms\": [";
        for (size_t i = 0; i < frame_ts.size(); ++i) AppendF(js, i ? ",%llu" : "%llu", (unsigned long long)frame_ts[i]);
        js += "],\n  \"frame_process_time_us\": [";
        for (size_t i = 0; i < proc.size(); ++i) AppendF(js, i ? ",%u" : "%u", proc[i]);
        js += "],\n  \"triggers\": [";
        for (size_t i = 0; i < job.triggers.size(); ++i) {
            const Trigger& t = job.triggers[i];
            const char* tt = t.is_fall ? (t.is_bed_exit ? "fall+bed_exit" : "fall") : (t.is_bed_exit ? "bed_exit" : "self_test");
            AppendF(js, "%s\n    {\"frame_index\": %d, \"seq\": %llu, \"timestamp_ms\": %llu, "
                        "\"type\": \"%s\", \"confidence\": %.4f}",
                    i ? "," : "", t.frame_index, (unsigned long long)t.seq,
                    (unsigned long long)t.timestamp_ms, tt, (double)t.confidence);
        }
        js += "\n  ]\n}\n";

        std::string meta_tmp = session_dir_ + "/.tmp_" + meta_name;
        int mfd = open(meta_tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        bool ok = false;
        if (mfd >= 0) {
            ok = WriteAll(mfd, js.data(), js.size()) && (fsync(mfd) == 0);
            close(mfd);
            if (ok) ok = (rename(meta_tmp.c_str(), (session_dir_ + "/" + meta_name).c_str()) == 0);
            if (!ok) unlink(meta_tmp.c_str());
        }
        if (!ok) LogF("[EventRecorderChunk] ERROR: write meta failed: %s", meta_name.c_str());
    }

    // --- register retention (protect chunks from GC) + persist to retention.idx ---
    if (job.stored) {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            retained_.push_back({job.first_seq, job.last_seq, meta_name});
        }
        int rfd = open(retention_path_.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (rfd >= 0) {
            std::string line;
            AppendF(line, "{\"meta\":\"%s\",\"first\":%llu,\"last\":%llu}\n",
                    meta_name.c_str(), (unsigned long long)job.first_seq, (unsigned long long)job.last_seq);
            WriteAll(rfd, line.data(), line.size()); fsync(rfd); close(rfd);
        }
    }

    // --- global jsonl log ---
    {
        std::string line;
        AppendF(line, "{\"datetime\":\"%s\",\"meta\":\"%s\",\"event_type\":\"%s\","
                      "\"first_trigger_frame\":%d,\"event_timestamp_ms\":%llu,\"trigger_count\":%llu,"
                      "\"max_confidence\":%.4f,\"total_frames\":%llu,\"stored\":%s,\"not_stored_reason\":\"%s\","
                      "\"complete\":%s}\n",
                th, meta_name.c_str(), event_type, event_frame_index,
                (unsigned long long)(job.triggers.empty() ? 0 : job.triggers.front().timestamp_ms),
                (unsigned long long)job.triggers.size(), (double)max_conf, (unsigned long long)present,
                job.stored ? "true" : "false", job.not_stored_reason, job.complete ? "true" : "false");
        int lfd = open((session_dir_ + "/" + kLogFileName).c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (lfd >= 0) { WriteAll(lfd, line.data(), line.size()); fsync(lfd); close(lfd); }
    }
}

bool EventRecorderChunk::OverlapsRetained(uint64_t chunk_id) const {
    uint64_t lo = chunk_id * (uint64_t)chunk_frames_;
    uint64_t hi = lo + (uint64_t)chunk_frames_ - 1;
    for (const Retained& r : retained_)
        if (!(hi < r.first_seq || lo > r.last_seq)) return true;
    return false;
}

void EventRecorderChunk::RunGc() {
    static uint64_t last_gc_ms = 0;
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t now_ms = (uint64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
    if (now_ms - last_gc_ms < kGcIntervalMs) return;
    last_gc_ms = now_ms;

    uint64_t margin = (uint64_t)(pre_frames_ + post_frames_ + 2 * chunk_frames_);
    std::vector<uint64_t> to_delete;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (spooled_seq_ <= margin) return;
        uint64_t floor_seq = spooled_seq_ - margin;
        for (uint64_t cid : existing_chunks_) {
            if ((cid + 1) * (uint64_t)chunk_frames_ > floor_seq) break;  // within rolling buffer -> keep
            if (OverlapsRetained(cid)) continue;                          // belongs to a retained event
            to_delete.push_back(cid);
            if ((int)to_delete.size() >= kGcMaxDeletePerPass) break;
        }
    }
    if (to_delete.empty()) return;
    for (uint64_t cid : to_delete) unlink(ChunkPath(cid).c_str());   // unlink OUTSIDE the lock
    {
        std::lock_guard<std::mutex> lk(mtx_);
        for (uint64_t cid : to_delete) existing_chunks_.erase(cid);
    }
}

void EventRecorderChunk::WriterLoop() {
    for (;;) {
        MetaJob job; bool have = false; bool quit = false;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait_for(lk, std::chrono::milliseconds(kGcIntervalMs),
                         [this] { return stop_ || !jobs_.empty(); });
            if (!jobs_.empty()) { job = std::move(jobs_.front()); jobs_.pop_front(); have = true; }
            else if (stop_) quit = true;
        }
        if (have) WriteMeta(job);
        RunGc();
        if (quit) return;
    }
}

void EventRecorderChunk::Shutdown() {
    {
        std::lock_guard<std::mutex> api_lk(api_mtx_);
        if (shutdown_) return;
        shutdown_ = true;
        // Flush an in-flight capture as a (partial) recording.
        if (capturing_ && ready_ && write_seq_ > 0) QueueMeta(false);
    }
    {
        std::lock_guard<std::mutex> lk(mtx_);
        stop_ = true;
    }
    cv_.notify_all();
    if (spooler_.joinable()) spooler_.join();
    if (writer_.joinable()) writer_.join();
    spooler_running_ = false;
    writer_running_ = false;
    if (cur_chunk_fd_ >= 0) { close(cur_chunk_fd_); cur_chunk_fd_ = -1; }
}

}  // namespace VisionSDK
