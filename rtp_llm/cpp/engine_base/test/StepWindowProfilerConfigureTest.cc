#include "gtest/gtest.h"

#define private public
#include "rtp_llm/cpp/engine_base/TorchProfiler.h"

#include <climits>
#include <string>

namespace rtp_llm {

// Trust-boundary tests for StepWindowProfiler::configure().
// These simulate callers that bypass the Python HTTP entry's sanitize step
// (direct gRPC/ARPC clients calling into EmbeddingRpcServer or NormalEngine's
// QueryConverter). configure() must independently defend the output file path
// and the profiler capture window regardless of how it was reached.

class StepWindowProfilerConfigureTest: public ::testing::Test {
protected:
    StepWindowProfiler profiler_{".", /*world_rank=*/0};
};

TEST_F(StepWindowProfilerConfigureTest, SanitizesPathTraversal) {
    profiler_.configure(true, "../../etc/passwd", 0, 1);
    // Dots and slashes must be stripped; only [A-Za-z0-9_-] survive.
    EXPECT_EQ(profiler_.trace_name_, "etcpasswd");
}

TEST_F(StepWindowProfilerConfigureTest, SanitizesAbsolutePath) {
    profiler_.configure(true, "/tmp/evil", 0, 1);
    EXPECT_EQ(profiler_.trace_name_, "tmpevil");
}

TEST_F(StepWindowProfilerConfigureTest, KeepsSafeCharacters) {
    profiler_.configure(true, "my_trace-01", 0, 1);
    EXPECT_EQ(profiler_.trace_name_, "my_trace-01");
}

TEST_F(StepWindowProfilerConfigureTest, TruncatesOverlongTraceName) {
    std::string long_name(500, 'a');
    profiler_.configure(true, long_name, 0, 1);
    EXPECT_LE(profiler_.trace_name_.size(), 64u);
    EXPECT_EQ(profiler_.trace_name_, std::string(64, 'a'));
}

TEST_F(StepWindowProfilerConfigureTest, EmptyTraceNameAllowed) {
    profiler_.configure(true, "", 0, 1);
    EXPECT_TRUE(profiler_.trace_name_.empty());
}

TEST_F(StepWindowProfilerConfigureTest, ClampsExcessiveNumSteps) {
    profiler_.configure(true, "trace", 0, INT_MAX);
    EXPECT_EQ(profiler_.num_steps_.load(), 1000);
}

TEST_F(StepWindowProfilerConfigureTest, ClampsNegativeToDefault) {
    profiler_.configure(true, "trace", 0, -5);
    EXPECT_EQ(profiler_.num_steps_.load(), 3);
}

TEST_F(StepWindowProfilerConfigureTest, ClampsZeroToDefault) {
    profiler_.configure(true, "trace", 0, 0);
    EXPECT_EQ(profiler_.num_steps_.load(), 3);
}

TEST_F(StepWindowProfilerConfigureTest, KeepsNumStepsInRange) {
    profiler_.configure(true, "trace", 0, 42);
    EXPECT_EQ(profiler_.num_steps_.load(), 42);
}

TEST_F(StepWindowProfilerConfigureTest, ClampsNegativeStartStep) {
    profiler_.configure(true, "trace", -7, 1);
    EXPECT_EQ(profiler_.start_step_.load(), 0);
}

TEST_F(StepWindowProfilerConfigureTest, StopIsScopedAndCanDisarmBeforeStart) {
    profiler_.configure(true, "sessionA_window0", 0, 20);
    EXPECT_FALSE(profiler_.requestStop("sessionB_window0"));
    EXPECT_TRUE(profiler_.enabled());
    EXPECT_EQ(profiler_.status().state, "armed");
    EXPECT_TRUE(profiler_.requestStop("sessionA_window0"));
    EXPECT_FALSE(profiler_.enabled());
    profiler_.serviceControl();
    EXPECT_FALSE(profiler_.has_profiler_.load());
    EXPECT_TRUE(profiler_.requestStop("sessionA_window0"));
    EXPECT_EQ(profiler_.status().state, "finished");
    EXPECT_TRUE(profiler_.status().collection_complete);
    profiler_.configure(true, "sessionA_window0", 0, 20);
    EXPECT_FALSE(profiler_.enabled());
    profiler_.configure(true, "sessionB_window0", 0, 20);
    EXPECT_TRUE(profiler_.requestStop("sessionA_window0"));
    EXPECT_TRUE(profiler_.enabled());
    EXPECT_EQ(profiler_.status().trace_name, "sessionB_window0");
}

TEST_F(StepWindowProfilerConfigureTest, CancelledWindowRemainsBlockedAfterFinishedCleanup) {
    profiler_.configure(true, "cancelled_window", 0, 20);
    ASSERT_TRUE(profiler_.requestStop("cancelled_window"));
    profiler_.serviceControl();
    EXPECT_TRUE(profiler_.requestStop("cancelled_window"));
    profiler_.configure(true, "cancelled_window", 3, 1);
    EXPECT_FALSE(profiler_.enabled());
    EXPECT_EQ(profiler_.status().state, "finished");
    profiler_.configure(true, "next_window", 0, 20);
    EXPECT_TRUE(profiler_.enabled());
    EXPECT_TRUE(profiler_.requestStop("cancelled_window"));
    EXPECT_FALSE(profiler_.status().stop_requested);
    EXPECT_EQ(profiler_.status().trace_name, "next_window");
}

}  // namespace rtp_llm
