#include "rtp_llm/cpp/observability/ExecutionRecorder.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdlib>
#include <future>
#include <vector>

// Standalone CPU test: g++ -std=c++17 -pthread -I. ExecutionRecorder{,Test}.cc
int main(int argc, char** argv) {
    const char* directory = argc == 2 ? argv[1] : std::getenv("TEST_TMPDIR");
    assert(directory);
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
        ExecutionRecorder limited(std::string(directory) + "/budget", "test", 0, 0, 8, 100, 4);
        assert(limited.submit("events.jsonl", "12345"));
        limited.close();
        assert(limited.dropped() == 1);
    }
}
