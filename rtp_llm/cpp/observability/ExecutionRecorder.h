#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace rtp_llm {

// CPU-only writer. No CUDA or inference dependency; never serializes tensors.
class ExecutionRecorder {
public:
    static ExecutionRecorder& instance();
    static std::string        quote(const std::string& value);
    static int64_t            monotonicNs();
    static int64_t            unixNs();
    ExecutionRecorder(std::string directory,
                      std::string session,
                      int         rank,
                      int         dp_rank,
                      size_t      capacity        = 4096,
                      size_t      max_file_bytes  = 100 * 1024 * 1024,
                      size_t      max_total_bytes = 1024ULL * 1024 * 1024);
    ~ExecutionRecorder();
    bool enabled() const {
        return enabled_.load(std::memory_order_relaxed);
    }
    // Configure once during engine construction, before requests enter.
    void    configure(int rank, int dp_rank);
    int64_t nextId() {
        return next_id_.fetch_add(1);
    }
    std::string        identity() const;
    const std::string& owner() const {
        return owner_;
    }
    void setMetadata(std::string metadata);
    void markError() noexcept {
        ++errors_;
    }
    bool     submit(const std::string& file, std::function<std::string()> make_line) noexcept;
    bool     submit(const std::string& file, std::string line) noexcept;
    uint64_t dropped() const {
        return dropped_.load();
    }
    void close();

private:
    void run();
    void manifest(bool closed);
    struct Task {
        std::string                  file;
        std::function<std::string()> make_line;
    };
    struct File {
        std::ofstream stream;
        size_t        bytes = 0;
        size_t        part  = 0;
    };
    std::string                 directory_, session_, owner_, replica_;
    std::string                 metadata_ = "{}";
    int                         rank_, dp_rank_;
    size_t                      capacity_, max_file_bytes_, max_total_bytes_, total_bytes_ = 0;
    std::atomic<bool>           enabled_{false};
    std::atomic<int64_t>        next_id_{1};
    std::atomic<uint64_t>       generated_{0}, written_{0}, dropped_{0}, errors_{0};
    std::mutex                  mutex_;
    std::condition_variable     ready_;
    std::deque<Task>            queue_;
    std::map<std::string, File> files_;
    bool                        stop_ = false;
    std::thread                 worker_;
};

// Shared by copies of a stream. Serializes lifecycle transitions independently
// of the engine's stream lock. IDs never expose the external request ID.
class RecordedRequest {
public:
    explicit RecordedRequest(ExecutionRecorder& recorder): recorder_(recorder), id_(recorder.nextId()) {}
    int64_t id() const {
        return id_;
    }
    void enqueue(bool streaming);
    void scheduled(int64_t execution, int64_t scheduler_step = 0);
    void published(int64_t tokens, int sequence = 0);
    void terminal(int error_code, bool cancelled);

private:
    void               emit(const char* event, int64_t execution = 0, int sequence = -1, int error_code = 0) noexcept;
    ExecutionRecorder& recorder_;
    int64_t            id_, seq_ = 0, published_ = 0;
    int64_t            first_scheduler_step_ = 0;
    bool               enqueued_ = false, scheduled_ = false, first_ = false, terminal_ = false, streaming_ = true;
    std::mutex         mutex_;
};
}  // namespace rtp_llm
