#include "rtp_llm/cpp/observability/ExecutionRecorder.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <future>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include "alog/Logger.h"
#include "alog/Appender.h"

// CPU-only test; link against alog (no CUDA dependency).
static std::string readFile(const std::string& path) {
    std::ifstream input(path);
    assert(input.good());
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

int main(int argc, char** argv) {
    const char* directory = argc == 2 ? argv[1] : std::getenv("TEST_TMPDIR");
    assert(directory);
    const auto root_path = std::string(directory) + "/root.log";
    alog::Logger::getRootLogger()->setAppender(alog::FileAppender::getAppender(root_path.c_str()));
    using namespace rtp_llm;
    assert(ExecutionRecorder::quote("a\n\"\\") == "\"a\\u000a\\\"\\\\\"");
    {
        ExecutionRecorder recorder(std::string(directory) + "/normal", "test", 2, 1, 1024, 1024);
        assert(recorder.enabled());
        RecordedRequest request(recorder);
        request.enqueue(true);
        request.enqueue(true);
        std::vector<std::thread> threads;
        for (int i = 0; i < 16; ++i) {
            threads.emplace_back([&] {
                request.scheduled(42);
                request.published(1);
            });
        }
        for (auto& thread : threads)
            thread.join();
        threads.clear();
        for (int i = 0; i < 16; ++i)
            threads.emplace_back([&] { request.terminal(0, false); });
        for (auto& thread : threads)
            thread.join();
        RecordedRequest cancelled(recorder);
        cancelled.enqueue(false);
        cancelled.terminal(8100, true);
        RecordedRequest failed(recorder);
        failed.enqueue(true);
        failed.terminal(606, false);
        recorder.close();
        assert(recorder.dropped() == 0);
        const auto owner_dir = std::string(directory) + "/normal/owner-" + recorder.owner();
        size_t     lines     = 0;
        for (const auto& entry : std::filesystem::directory_iterator(owner_dir)) {
            if (entry.path().filename().string().find("request_events.jsonl") != 0)
                continue;
            std::ifstream input(entry.path());
            std::string   line;
            while (std::getline(input, line)) {
                assert(line.front() == '{' && line.back() == '}');
                ++lines;
            }
        }
        assert(lines == 8);
        const auto manifest = readFile(owner_dir + "/manifest.json");
        assert(manifest.find("\"submitted_to_alog\":8") != std::string::npos);
        assert(manifest.find("\"written\":null") != std::string::npos);
        assert(manifest.find("\"complete\":false") != std::string::npos);
        assert(manifest.find("\"closed\":true") != std::string::npos);
    }
    {
        ExecutionRecorder  recorder(std::string(directory) + "/overflow", "test", 0, 0, 1);
        std::promise<void> entered, release;
        auto               gate = release.get_future().share();
        assert(recorder.submit("events.jsonl", [&] {
            entered.set_value();
            gate.wait();
            return "{}";
        }));
        entered.get_future().wait();
        assert(recorder.submit("events.jsonl", "{}"));
        assert(!recorder.submit("events.jsonl", "{}"));
        release.set_value();
        recorder.close();
        assert(recorder.dropped() == 1);
    }
    {
        ExecutionRecorder disabled("", "test", 0, 0);
        assert(!disabled.enabled());
        assert(!disabled.submit("events.jsonl", "{}"));
    }
    {
        ExecutionRecorder recorder(std::string(directory) + "/live", "test", 0, 0);
        assert(recorder.submit("batches.jsonl", "{\"live\":true}"));
        const auto path    = std::string(directory) + "/live/owner-" + recorder.owner() + "/batches.jsonl";
        bool       visible = false;
        for (int i = 0; i < 200; ++i) {
            if (std::filesystem::exists(path) && readFile(path) == "{\"live\":true}\n") {
                visible = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        assert(visible);  // Periodic alog flush works before recorder.close().
        recorder.close();
    }
    {
        // Exercise messages larger than even the production alog.max_msg_len.
        ExecutionRecorder recorder(std::string(directory) + "/large", "test", 0, 0);
        const auto        line = "{\"payload\":\"" + std::string(3 * 1024 * 1024, 'x') + "\"}";
        assert(recorder.submit("batches.jsonl", line));
        recorder.close();
        assert(readFile(std::string(directory) + "/large/owner-" + recorder.owner() + "/batches.jsonl") == line + "\n");
    }
    {
        ExecutionRecorder recorder(std::string(directory) + "/producer_error", "test", 0, 0);
        assert(recorder.submit("events.jsonl", []() -> std::string { throw std::runtime_error("snapshot failed"); }));
        recorder.close();
        const auto manifest =
            readFile(std::string(directory) + "/producer_error/owner-" + recorder.owner() + "/manifest.json");
        assert(manifest.find("\"errors\":1") != std::string::npos);
    }
    {
        ExecutionRecorder limited(std::string(directory) + "/budget", "test", 0, 0, 8, 100, 4);
        assert(limited.submit("events.jsonl", "12345"));
        limited.close();
        assert(limited.dropped() == 1);
    }
    alog::Logger::getRootLogger()->flush();
    assert(readFile(root_path).empty());  // Recorder JSON never inherits the engine/root appender.
}
