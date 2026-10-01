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

#include <algorithm>
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

// ===========================================================================
// 執行中的異常防護 (2026-09-09)
// ===========================================================================
// InitStorage 的檢查只涵蓋「開機當下」。SD 跑到一半被拔掉、因 EIO 被 remount
// read-only、或被別的東西寫滿時,原本的程式碼沒有任何反應:每幀照樣複製、照樣
// pwrite 失敗、照樣印一行 log,永遠不會自我關閉。以下三個上限把那些路徑箝死。

// writer thread 待處理的 meta job 上限。每個 job 帶一張 bg_image + 整個 window
// 的 analysis(MB 級),而 SD 進入 mmc I/O timeout 時 writer 會卡在 fsync 數十
// 秒。這是整個 recorder 唯一真正無界的 RAM 成長路徑 -> 必須有硬上限。
const size_t kMaxPendingJobs = 3;

// 單一 session 內最多保留幾個事件的 chunk。retained_ 只增不減(每筆含一個
// std::string),而 OverlapsRetained() 是線性掃 -> RAM 與 GC 的 CPU 都會隨執行
// 時間單調成長。超過上限後新事件仍會寫 meta,只是不再保留 raw chunk。
const int kMaxRetainedEvents = 256;

// 連續多少幀寫入失敗才判定 SD 沒救了。ENOSPC 有機會被 GC 救回來,所以給它一
// 整個 chunk 的寬容度;致命的 errno(SD 不見了)則不等待,立刻停用。
const int kSpoolErrStreakFatal = 128;

const uint32_t kErrLogIntervalMs = 5000;   // 錯誤 log 節流間隔

uint64_t NowMs() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
}

// 這個 errno 代表儲存裝置本身沒了/不可寫,重試沒有意義。
bool IsFatalIoErrno(int e) {
    return e == EIO || e == ENODEV || e == ENXIO || e == EROFS ||
           e == EBADF || e == ENOENT || e == EACCES;
}

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

// ===========================================================================
// 儲存位置安全性 + 跨 session 容量控管 (2026-08-28)
// ===========================================================================
// 原本 InitStorage 只用 stat()+S_ISDIR 判斷 kBasePath,那只驗證「目錄存在」。
// SD 沒掛載時 /mnt/mmcblk1p1 仍然是一個空的掛載點目錄,判斷會通過,於是整條
// 滾動緩衝就寫進 rootfs 掛載點底下 -> 吃光 RAM/flash -> 板子被 OOM 打死、必須
// 重新上電。而且之後 RetainBlockedReason() 的 statvfs 量到的也會變成 rootfs,
// 容量保護等於整個失效。
//
// 新規則(失敗一律是「停用錄影」,絕不「換個地方寫」):
//   * kBasePath 不存在        -> PC 開發環境,沿用 "." fallback(維持原本行為)
//   * kBasePath 存在但沒掛載  -> Fail
//   * kBasePath 掛載但不可寫  -> Fail
//   * 剩餘空間不足            -> Fail
const uint64_t kMinFreeBytesToStart = 256ull * 1024 * 1024;   // 啟動門檻絕對下限

// 保留幾個「先前執行」的 session 目錄,更舊的在啟動時刪除。
// 這是跨 session 的容量控管:RunGc() 只管得到「當前 session」的 chunk
// (existing_chunks_ 每次都是全新的空集合),所以每次重開機,上一輪的滾動緩衝與
// 被事件保留的 chunk 就永遠留在 SD 上沒人清。板子當掉->重上電的循環會讓它單調
// 累積,而當掉正是我們在追的症狀 -> 形成正回饋。
const int kMaxOldSessions = 3;

// path 是否為真正的掛載點(與其父目錄不同 st_dev)。
bool IsMountPoint(const std::string& path) {
    struct stat me, par;
    if (stat(path.c_str(), &me) != 0) return false;
    std::string p = path;
    while (p.size() > 1 && p[p.size() - 1] == '/') p.erase(p.size() - 1);
    size_t slash = p.find_last_of('/');
    std::string parent = (slash == std::string::npos) ? std::string(".")
                       : (slash == 0 ? std::string("/") : p.substr(0, slash));
    if (stat(parent.c_str(), &par) != 0) return false;
    return me.st_dev != par.st_dev;
}

uint64_t FreeBytes(const std::string& dir) {
    struct statvfs vfs;
    if (statvfs(dir.c_str(), &vfs) != 0) return 0;
    return (uint64_t)vfs.f_bavail * (uint64_t)vfs.f_frsize;
}

bool RemoveDirRecursive(const std::string& dir) {
    DIR* d = opendir(dir.c_str());
    if (!d) return false;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0) continue;
        std::string child = dir + "/" + e->d_name;
        struct stat st;
        if (lstat(child.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) RemoveDirRecursive(child);
        else unlink(child.c_str());
    }
    closedir(d);
    return rmdir(dir.c_str()) == 0;
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

bool EventRecorderChunk::ErrThrottle::Allow(uint64_t& suppressed) {
    uint64_t now = NowMs();
    if (last_ms != 0 && now - last_ms < kErrLogIntervalMs) { ++hidden; return false; }
    last_ms = now;
    suppressed = hidden;
    hidden = 0;
    return true;
}

// 可以從任何 thread 呼叫(spooler 在偵測到執行中的 I/O 錯誤時也會呼叫),
// 並且只會真正生效/印出一次。
void EventRecorderChunk::Fail(const std::string& why) {
    if (failed_.exchange(true, std::memory_order_relaxed)) return;
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
    width_ = width; height_ = height; channels_ = channels;
    frame_size_ = (uint32_t)((uint64_t)width * (uint64_t)height * (uint64_t)channels);
    if (width <= 0 || height <= 0 || channels <= 0 || frame_size_ == 0) {
        Fail("zero/invalid frame size");
        return false;
    }

    // --- 1) 決定儲存位置。詳見上方 kMinFreeBytesToStart 附近的說明。 ---
    struct stat st;
    if (stat(kBasePath, &st) != 0) {
        // 掛載點目錄本身不存在。這在 PC 上是正常的(開發環境),在板子上則
        // 代表開機腳本沒把掛載點建出來 —— 此時退回 CWD 是危險的:CWD 可能
        // 落在 tmpfs/rootfs 上,statvfs 量到的是 tmpfs 上限(約 RAM 的一半),
        // 空間門檻會通過,然後幾百 MB 的滾動緩衝就直接吃光 RAM -> OOM。
        // 所以 fallback 只在 PC build 開啟,板子一律停用錄影。
#if defined(__PC_BUILD__) || defined(HERMES_ALLOW_CWD_FALLBACK)
        base_dir_ = kFallbackBase;
        LogF("[EventRecorderChunk] %s not present, using dev fallback '%s'",
             kBasePath, kFallbackBase);
#else
        Fail(std::string(kBasePath) + " does not exist (SD mount point missing) -- "
             "refusing to fall back to the working directory");
        return false;
#endif
    } else if (!S_ISDIR(st.st_mode)) {
        Fail(std::string(kBasePath) + " exists but is not a directory");
        return false;
    } else if (!IsMountPoint(kBasePath)) {
        // 目錄在,但沒有任何檔案系統掛在上面 => SD 沒插/沒掛載/掉了。
        // 絕對不能往這裡寫:那是 rootfs,寫下去會吃光 RAM 或 flash。
        Fail(std::string(kBasePath) + " is not a mount point (SD not mounted?) -- "
             "refusing to record onto the rootfs");
        return false;
    } else if (access(kBasePath, W_OK) != 0) {
        // 有掛載但不可寫(常見於 I/O error 後被 remount read-only)。
        Fail(std::string(kBasePath) + " is mounted but not writable (remounted read-only?)");
        return false;
    } else {
        base_dir_ = kBasePath;
    }

    event_dir_ = base_dir_ + "/" + kEventDirName;
    if (!MakeDir(event_dir_)) { Fail("mkdir event_record failed"); return false; }

    // --- 2) 掃描既有 session 目錄,新 session = max + 1 ---
    std::vector<long long> sessions;
    DIR* ed = opendir(event_dir_.c_str());
    if (ed) {
        struct dirent* e;
        while ((e = readdir(ed)) != nullptr) {
            const char* n = e->d_name;
            if (n[0] < '0' || n[0] > '9') continue;   // skip ".", "..", non-numeric
            bool all_digit = true;
            for (const char* q = n; *q; ++q) if (*q < '0' || *q > '9') { all_digit = false; break; }
            if (!all_digit) continue;
            sessions.push_back(atoll(n));
        }
        closedir(ed);
    }
    std::sort(sessions.begin(), sessions.end());
    session_id_ = sessions.empty() ? 1 : sessions.back() + 1;

    // --- 3) 跨 session 容量控管:刪掉過舊的 session 目錄 ---
    // RunGc() 只回收「當前 session」的 chunk,所以先前每一輪執行留下的滾動緩衝
    // 與被事件保留的 chunk 永遠不會有人清。這裡在啟動時補上:
    //   (a) 只保留最近 kMaxOldSessions 個舊 session;
    //   (b) 若剩餘空間仍低於本次滾動緩衝所需 + 門檻,繼續由舊往新刪。
    // 注意:這會永久刪除舊的事件錄影資料。要保留更多,調大 kMaxOldSessions。
    const uint64_t rolling_reserve =
        (uint64_t)(pre_frames_ + post_frames_ + 2 * chunk_frames_) * (uint64_t)frame_size_;
    const uint64_t need = kMinFreeBytesToStart + rolling_reserve;

    size_t keep_from = (sessions.size() > (size_t)kMaxOldSessions)
                     ? sessions.size() - (size_t)kMaxOldSessions : 0;
    for (size_t i = 0; i < sessions.size(); ++i) {
        bool too_old  = (i < keep_from);
        bool too_full = (!too_old && i + 1 < sessions.size() && FreeBytes(event_dir_) < need);
        if (!too_old && !too_full) break;   // 由舊往新,一旦不需要刪就停
        char sid_old[24];
        snprintf(sid_old, sizeof(sid_old), "%lld", sessions[i]);
        std::string victim = event_dir_ + "/" + sid_old;
        LogF("[EventRecorderChunk] GC: removing old session dir %s (%s)",
             victim.c_str(), too_old ? "beyond kMaxOldSessions" : "low free space");
        RemoveDirRecursive(victim);
    }

    // --- 4) 空間門檻:寧可不錄,也不要把儲存裝置寫爆 ---
    uint64_t freeb = FreeBytes(event_dir_);
    if (freeb < need) {
        char why[192];
        snprintf(why, sizeof(why),
                 "insufficient free space at %s: %llu MB free, need %llu MB "
                 "(rolling reserve %llu MB + floor %llu MB)",
                 event_dir_.c_str(),
                 (unsigned long long)(freeb / (1024 * 1024)),
                 (unsigned long long)(need / (1024 * 1024)),
                 (unsigned long long)(rolling_reserve / (1024 * 1024)),
                 (unsigned long long)(kMinFreeBytesToStart / (1024 * 1024)));
        Fail(why);
        return false;
    }

    char sid[24];
    snprintf(sid, sizeof(sid), "%lld", session_id_);
    session_dir_ = event_dir_ + "/" + sid;
    frames_dir_ = session_dir_ + "/" + kFramesDirName;
    retention_path_ = session_dir_ + "/" + kRetentionName;
    if (!MakeDir(session_dir_) || !MakeDir(frames_dir_)) { Fail("mkdir session dir failed"); return false; }

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
            dropped_frames_.fetch_add(1, std::memory_order_relaxed);
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

        // 已經因為執行中的 I/O 錯誤停用了:把佇列裡殘留的幀丟掉就好,
        // 不要繼續對一張壞掉/不存在的卡重試。
        if (failed_.load(std::memory_order_relaxed)) continue;

        bool io_err = false;
        int io_errno = 0;
        const char* io_what = "";

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
            } else {
                // 原本這裡完全靜默:整個 chunk(chunk_frames_ 幀)的資料無聲消失,
                // 事後只能靠對照 meta 才看得出來。open 失敗通常比 write 失敗更早、
                // 更明確地代表 SD 不見了,所以一併納入錯誤升級。
                io_err = true; io_errno = errno; io_what = "open chunk";
            }
        }
        if (cur_chunk_fd_ >= 0) {
            off_t off = (off_t)(sf.seq % (uint64_t)chunk_frames_) * (off_t)frame_size_;
            if (!PWriteAll(cur_chunk_fd_, sf.data.data(), frame_size_, off)) {
                io_err = true; io_errno = errno; io_what = "write chunk";
            }
        }

        if (io_err) {
            ++spool_err_streak_;
            uint64_t hidden = 0;
            if (spool_err_log_.Allow(hidden)) {
                LogF("[EventRecorderChunk] %s failed: %s (errno=%d, consecutive=%d, "
                     "suppressed=%llu since last line)",
                     io_what, strerror(io_errno), io_errno, spool_err_streak_,
                     (unsigned long long)hidden);
            }
            // 致命 errno 立刻停用;ENOSPC 之類的先撐一個 chunk,給 GC 一點時間
            // 把空間清回來,真的清不回來才放棄。停用後 PushFrame 會直接 return,
            // 不再每幀複製整張影像,IsActive() 也會讓上層跳過 analysis 的組裝。
            if (IsFatalIoErrno(io_errno) || spool_err_streak_ >= kSpoolErrStreakFatal) {
                errno = io_errno;
                Fail(IsFatalIoErrno(io_errno)
                         ? "chunk I/O error at runtime (SD removed / remounted read-only?)"
                         : "chunk writes keep failing (storage full?)");
                if (cur_chunk_fd_ >= 0) { close(cur_chunk_fd_); cur_chunk_fd_ = -1; }
            }
        } else {
            spool_err_streak_ = 0;
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

// nullptr => 可以保留。回傳的字串都是靜態字面量,會被存進 MetaJob 的
// not_stored_reason 並寫進 meta.json / event_record.jsonl。
const char* EventRecorderChunk::RetainBlockedReason() const {
    // 保留數量硬上限:retained_ 在單一 session 內只增不減(每筆含一個
    // std::string),OverlapsRetained() 又是線性掃 -> RAM 與 GC 的 CPU 都會隨
    // 執行時間單調成長。到頂之後事件照樣寫 meta,只是不再保留 raw chunk。
    if (retained_count_.load(std::memory_order_relaxed) >= (size_t)kMaxRetainedEvents)
        return "retention_full";

    struct statvfs vfs;
    if (statvfs(base_dir_.c_str(), &vfs) != 0) {
        // 原本這裡是 return true(允許保留)。但 SD 掉了 / EIO 時 statvfs 正好
        // 就是會失敗 —— fail-open 等於在最該保守的時候最寬鬆。改成 fail-closed,
        // 與 InitStorage「寧可不錄,也不要把儲存裝置寫爆」的原則一致。
        return "statvfs_failed";
    }
    uint64_t free_bytes = (uint64_t)vfs.f_bavail * (uint64_t)vfs.f_frsize;
    uint64_t window = (uint64_t)(pre_frames_ + 1 + post_frames_) * frame_size_;
    uint64_t rolling_reserve = (uint64_t)(pre_frames_ + post_frames_ + 2 * chunk_frames_) * frame_size_;
    if (free_bytes < (window + rolling_reserve + window /*safety*/)) return "sd_full";
    return nullptr;
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
    const char* blocked = store_raw_ ? RetainBlockedReason() : "store_raw_off";
    job.stored = (blocked == nullptr);
    job.not_stored_reason = blocked ? blocked : "";
    if (blocked && store_raw_) {
        uint64_t hidden = 0;
        if (retain_block_log_.Allow(hidden))
            LogF("[EventRecorderChunk] event NOT retaining raw chunks: %s "
                 "(meta still written; suppressed=%llu since last line)",
                 blocked, (unsigned long long)hidden);
    }
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
    size_t dropped_now = 0;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        // 硬上限。SD 進入 mmc I/O timeout 時 writer 會卡在 fsync 數十秒到數分鐘,
        // 這期間事件仍會持續進來(post_frames_==0 時甚至是每幀一個),而每個 job
        // 帶一張 bg_image + 整個 window 的 analysis,是 MB 級的。沒有上限的話
        // 這是唯一會無限成長的結構。滿了就丟最舊的 —— 留最新的事件比留最舊的
        // 有意義,而且丟掉的只是 meta/analysis,chunk 本身還在滾動緩衝裡。
        while (jobs_.size() >= kMaxPendingJobs) {
            jobs_.pop_front();
            ++dropped_now;
        }
        jobs_.push_back(std::move(job));
    }
    if (dropped_now > 0) {
        uint64_t total = dropped_jobs_.fetch_add(dropped_now, std::memory_order_relaxed) + dropped_now;
        uint64_t hidden = 0;
        if (job_drop_log_.Allow(hidden))
            LogF("[EventRecorderChunk] writer backlog at limit (%d): dropped %d oldest "
                 "meta job(s), total dropped=%llu (suppressed=%llu since last line)",
                 (int)kMaxPendingJobs, (int)dropped_now,
                 (unsigned long long)total, (unsigned long long)hidden);
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
        AppendF(js, "  \"ring_dropped_frames\": %llu,\n",
                (unsigned long long)dropped_frames_.load(std::memory_order_relaxed));
        // 診斷欄位:writer 塞車而被丟棄的 meta job 數。持續非 0 代表 SD 寫入
        // 已經慢到跟不上事件產生的速度。
        AppendF(js, "  \"dropped_meta_jobs\": %llu,\n",
                (unsigned long long)dropped_jobs_.load(std::memory_order_relaxed));
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
            // 給 RetainBlockedReason() 用的無鎖鏡像(見 kMaxRetainedEvents)。
            retained_count_.store(retained_.size(), std::memory_order_relaxed);
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
