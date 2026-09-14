#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "rtp_llm/cpp/observability/ExecutionRecorder.h"
#include <filesystem>
#include <fstream>
#include <iterator>
#include <unistd.h>

namespace rtp_llm {
class RecordingEngineTest: public DeviceTestBase {};

TEST_F(RecordingEngineTest, LifecycleAndSnapshot) {
    char directory[] = "/tmp/rtp-recording-engine-XXXXXX";
    ASSERT_NE(mkdtemp(directory), nullptr);
    setenv("RTP_LLM_RECORD_DIR", directory, 1);
    setenv("RTP_LLM_RECORD_SESSION", "engine-test", 1);
    auto engine                            = createMockEngine(CustomConfig{});
    auto input                             = std::make_shared<GenerateInput>();
    input->input_ids                       = torch::tensor({1, 2, 3}, torch::kInt32);
    input->generate_config                 = std::make_shared<GenerateConfig>();
    input->generate_config->max_new_tokens = 4;
    input->generate_config->is_streaming   = true;
    auto stream                            = engine->enqueue(input);
    ASSERT_TRUE(stream->nextOutput().ok());
    ASSERT_TRUE(stream->nextOutput().ok());
    ASSERT_TRUE(stream->nextOutput().ok());
    ASSERT_TRUE(stream->nextOutput().ok());
    ASSERT_FALSE(stream->nextOutput().ok());
    ASSERT_TRUE(engine->stop().ok());
    ExecutionRecorder::instance().close();
    std::string events, batches;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
        const auto name = entry.path().filename().string();
        if (name != "request_events.jsonl" && name != "batches.jsonl")
            continue;
        std::ifstream file(entry.path());
        std::string   text((std::istreambuf_iterator<char>(file)), {});
        (name == "request_events.jsonl" ? events : batches) += text;
    }
    for (const auto* name : {"enqueue", "first_scheduled", "first_token", "finish"}) {
        const auto needle = std::string("\"event\":\"") + name + "\"";
        const auto first  = events.find(needle);
        ASSERT_NE(first, std::string::npos) << events;
        EXPECT_EQ(events.find(needle, first + 1), std::string::npos) << events;
    }
    EXPECT_NE(batches.find("\"q_tokens\":3"), std::string::npos) << batches;
    const auto* decode_env    = std::getenv("RTP_LLM_RECORD_DECODE");
    const bool  record_decode = decode_env && std::string(decode_env) == "1";
    if (record_decode) {
        EXPECT_NE(batches.find("\"phase\":\"decode\""), std::string::npos) << batches;
        EXPECT_NE(batches.find("\"q_tokens\":1"), std::string::npos) << batches;
    } else {
        EXPECT_EQ(batches.find("\"phase\":\"decode\""), std::string::npos) << batches;
        EXPECT_EQ(batches.find("\"q_tokens\":1"), std::string::npos) << batches;
    }
    unsetenv("RTP_LLM_RECORD_DIR");
    unsetenv("RTP_LLM_RECORD_SESSION");
}

TEST_F(RecordingEngineTest, NativeReplay) {
    char directory[] = "/tmp/rtp-replay-engine-XXXXXX";
    ASSERT_NE(mkdtemp(directory), nullptr);
    const auto path = std::string(directory) + "/replay.plan";
    {
        std::ofstream plan(path);
        // Mixed decode/prefill with prefix KV, then an independent prefill.
        plan << "RTP_BATCH_REPLAY_V1 1 2 2\n11 2\n0 1 4 3\n1 2 3 5\n12 1\n1 3 0 3\n";
    }
    setenv("RTP_LLM_REPLAY_PLAN", path.c_str(), 1);
    auto engine = createMockEngine(CustomConfig{});
    // stop joins the dedicated loop; it does not interrupt fixed replay iterations.
    ASSERT_TRUE(engine->stop().ok());
    unsetenv("RTP_LLM_REPLAY_PLAN");
    std::ifstream     file(path + ".rank0.result.jsonl");
    const std::string result((std::istreambuf_iterator<char>(file)), {});
    EXPECT_NE(result.find("\"status\":\"complete\""), std::string::npos) << result;
    EXPECT_EQ(result.find("\"status\":\"error\""), std::string::npos) << result;
}
}  // namespace rtp_llm
