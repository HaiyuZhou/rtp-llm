#include "rtp_llm/cpp/observability/ExecutionRecorder.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <unistd.h>

namespace rtp_llm {
namespace {
std::string env(const char* name, const std::string& fallback = "") {
    const auto* value = std::getenv(name);
    return value ? value : fallback;
}
size_t positiveEnv(const char* name, size_t fallback) {
    const auto text = env(name);
    if (text.empty())
        return fallback;
    try {
        size_t     end   = 0;
        const auto value = std::stoull(text, &end);
        return text[0] != '-' && end == text.size() && value > 0 ? value : fallback;
    } catch (...) {
        return fallback;
    }
}
}  // namespace

int64_t ExecutionRecorder::monotonicNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
int64_t ExecutionRecorder::unixNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}
std::string ExecutionRecorder::quote(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\')
            out << '\\' << c;
        else if (c < 32)
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
        else
            out << c;
    }
    out << '"';
    return out.str();
}

ExecutionRecorder& ExecutionRecorder::instance() {
    // A shared explicit session ID is required for cross-rank correlation.
    static ExecutionRecorder recorder(env("RTP_LLM_RECORD_DIR"),
                                      env("RTP_LLM_RECORD_SESSION"),
                                      0,
                                      0,
                                      positiveEnv("RTP_LLM_RECORD_QUEUE_SIZE", 4096),
                                      positiveEnv("RTP_LLM_RECORD_FILE_BYTES", 100 * 1024 * 1024),
                                      positiveEnv("RTP_LLM_RECORD_TOTAL_BYTES", 1024ULL * 1024 * 1024));
    return recorder;
}

ExecutionRecorder::ExecutionRecorder(std::string directory,
                                     std::string session,
                                     int         rank,
                                     int         dp_rank,
                                     size_t      capacity,
                                     size_t      max_file_bytes,
                                     size_t      max_total_bytes):
    directory_(std::move(directory)),
    session_(std::move(session)),
    rank_(rank),
    dp_rank_(dp_rank),
    capacity_(capacity),
    max_file_bytes_(max_file_bytes),
    max_total_bytes_(max_total_bytes) {
    if (directory_.empty() || session_.empty() || !capacity_ || !max_file_bytes_)
        return;
    owner_   = std::to_string(getpid()) + "-" + std::to_string(unixNs());
    replica_ = env("RTP_LLM_RECORD_REPLICA");
    next_id_ = unixNs();
    directory_ += "/owner-" + owner_;
    try {
        std::filesystem::create_directories(directory_);
        manifest(false);
        enabled_ = true;
        worker_  = std::thread([this] { run(); });
    } catch (...) {
        ++errors_;
        enabled_ = false;
    }
}
ExecutionRecorder::~ExecutionRecorder() {
    close();
}
void ExecutionRecorder::configure(int rank, int dp_rank) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (generated_ != 0)
        return;
    rank_    = rank;
    dp_rank_ = dp_rank;
    if (enabled()) {
        try {
            manifest(false);
        } catch (...) {
            ++errors_;
            enabled_ = false;
        }
    }
}
void ExecutionRecorder::setMetadata(std::string metadata) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (generated_ != 0)
        return;
    metadata_ = std::move(metadata);
    if (enabled()) {
        try {
            manifest(false);
        } catch (...) {
            ++errors_;
            enabled_ = false;
        }
    }
}
std::string ExecutionRecorder::identity() const {
    return "\"schema_version\":1,\"session_id\":" + quote(session_)
           + ",\"replica_id\":" + quote(replica_.empty() ? "dp" + std::to_string(dp_rank_) : replica_)
           + ",\"dp_rank\":" + std::to_string(dp_rank_) + ",\"world_rank\":" + std::to_string(rank_)
           + ",\"owner_instance_id\":" + quote(owner_);
}
bool ExecutionRecorder::submit(const std::string& file, std::string line) noexcept {
    return submit(file, [line = std::move(line)] { return line; });
}
bool ExecutionRecorder::submit(const std::string& file, std::function<std::string()> make_line) noexcept {
    if (!enabled())
        return false;
    ++generated_;
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_ || queue_.size() >= capacity_) {
            ++dropped_;
            return false;
        }
        queue_.push_back({file, std::move(make_line)});
        ready_.notify_one();
        return true;
    } catch (...) {
        ++dropped_;
        return false;
    }
}
void ExecutionRecorder::manifest(bool closed) {
    std::ofstream out(directory_ + "/manifest.json.tmp");
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << '{' << identity() << ",\"closed\":" << (closed ? "true" : "false")
        << ",\"complete\":" << (closed && dropped_ == 0 && errors_ == 0 ? "true" : "false")
        << ",\"generated\":" << generated_ << ",\"written\":" << written_ << ",\"dropped\":" << dropped_
        << ",\"errors\":" << errors_ << ",\"scope\":\"engine\",\"request_sampling\":false,\"engine\":" << metadata_
        << ",\"max_total_bytes\":" << max_total_bytes_ << ",\"bytes_written\":" << total_bytes_ << ",\"files\":[";
    bool first = true;
    for (const auto& entry : files_) {
        for (size_t part = 0; part <= entry.second.part; ++part) {
            if (!first)
                out << ',';
            first = false;
            out << quote(entry.first + (part ? "." + std::to_string(part) : ""));
        }
    }
    out << "]}\n";
    out.close();
    std::filesystem::rename(directory_ + "/manifest.json.tmp", directory_ + "/manifest.json");
}
void ExecutionRecorder::run() {
    for (;;) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [this] { return stop_ || !queue_.empty(); });
            if (queue_.empty())
                break;
            task = std::move(queue_.front());
            queue_.pop_front();
        }
        try {
            const auto line = task.make_line();
            if (line.size() >= max_total_bytes_ || total_bytes_ > max_total_bytes_ - line.size() - 1) {
                ++dropped_;
                ++errors_;
                enabled_ = false;
                manifest(false);
                continue;
            }
            auto& file = files_[task.file];
            if (file.stream.is_open() && file.bytes && file.bytes + line.size() + 1 > max_file_bytes_) {
                file.stream.close();
                file.bytes = 0;
                ++file.part;
            }
            if (!file.stream.is_open()) {
                file.stream.open(directory_ + "/" + task.file + (file.part ? "." + std::to_string(file.part) : ""));
                file.stream.exceptions(std::ios::failbit | std::ios::badbit);
            }
            file.stream << line << '\n';
            file.stream.flush();
            file.bytes += line.size() + 1;
            total_bytes_ += line.size() + 1;
            ++written_;
            if (written_ % 64 == 0)
                manifest(false);
        } catch (...) {
            ++errors_;
            enabled_ = false;
            try {
                manifest(false);
            } catch (...) {}
        }
    }
}
void ExecutionRecorder::close() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        enabled_ = false;
        stop_    = true;
    }
    ready_.notify_one();
    if (worker_.joinable()) {
        worker_.join();
        try {
            manifest(true);
        } catch (...) {
            ++errors_;
        }
    }
}

void RecordedRequest::emit(const char* event, int64_t execution, int sequence, int error_code) noexcept {
    try {
        const auto mono = ExecutionRecorder::monotonicNs();
        const auto wall = ExecutionRecorder::unixNs();
        recorder_.submit(
            "request_events.jsonl",
            "{" + recorder_.identity() + ",\"request_id\":" + ExecutionRecorder::quote("r" + std::to_string(id_))
                + ",\"event\":" + ExecutionRecorder::quote(event) + ",\"event_seq\":" + std::to_string(seq_++)
                + ",\"timestamp_monotonic_ns\":" + std::to_string(mono) + ",\"timestamp_unix_ns\":"
                + std::to_string(wall) + ",\"clock_id\":" + ExecutionRecorder::quote(recorder_.owner() + "-monotonic")
                + ",\"execution_id\":" + (execution ? std::to_string(execution) : "null") + ",\"scheduler_step_id\":"
                + (execution && first_scheduler_step_ ? std::to_string(first_scheduler_step_) : "null")
                + ",\"sequence_id\":" + (sequence >= 0 ? std::to_string(sequence) : "null")
                + ",\"output_mode\":" + ExecutionRecorder::quote(streaming_ ? "streaming" : "non_streaming")
                + ",\"generated_tokens\":" + std::to_string(published_) + ",\"reason\":"
                + ExecutionRecorder::quote(error_code ? "engine_error" : (terminal_ ? "completed" : ""))
                + ",\"error_code\":" + (error_code ? std::to_string(error_code) : "null") + "}");
    } catch (...) {
        recorder_.markError();
    }
}
void RecordedRequest::enqueue(bool streaming) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (enqueued_)
        return;
    streaming_ = streaming;
    enqueued_  = true;
    emit("enqueue");
}
void RecordedRequest::scheduled(int64_t execution, int64_t scheduler_step) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enqueued_ || terminal_ || scheduled_)
        return;
    scheduled_            = true;
    first_scheduler_step_ = scheduler_step;
    emit("first_scheduled", execution);
}
void RecordedRequest::published(int64_t tokens, int sequence) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enqueued_ || terminal_ || tokens <= 0)
        return;
    published_ += tokens;
    if (!first_) {
        first_ = true;
        emit("first_token", 0, sequence);
    }
}
void RecordedRequest::terminal(int error_code, bool cancelled) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enqueued_ || terminal_)
        return;
    terminal_ = true;
    emit(cancelled ? "cancel" : error_code ? "error" : "finish", 0, -1, error_code);
}
}  // namespace rtp_llm
