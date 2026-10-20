#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServiceImpl.h"
#include "rtp_llm/cpp/distribute/CpuTpBroadcaster.h"
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace rtp_llm {
namespace {

bool waitFile(const std::string& path) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (std::chrono::steady_clock::now() < deadline) {
        if (std::filesystem::exists(path))
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

void writeFile(const std::string& path, const std::string& value) {
    std::ofstream output(path + ".tmp");
    output << value;
    output.close();
    std::filesystem::rename(path + ".tmp", path);
}

class PressureTpServer: public LocalRpcServer {
public:
    PressureTpServer(const std::shared_ptr<NormalEngine>& engine, int rank, const std::string& peer = "") {
        engine_                                            = engine;
        maga_init_params_.parallelism_config.world_rank    = rank;
        maga_init_params_.parallelism_config.tp_rank       = rank;
        maga_init_params_.parallelism_config.tp_size       = 2;
        maga_init_params_.runtime_config.worker_grpc_addrs = {"", peer};
    }
};

class DelayedPressureService: public LocalRpcServiceImpl {
public:
    explicit DelayedPressureService(std::shared_ptr<LocalRpcServer> server) {
        local_server_ = std::move(server);
    }
    grpc::Status ControlPressureDiagnosticsInternal(grpc::ServerContext*                   context,
                                                    const PressureSessionControlRequestPB* request,
                                                    PressureSessionControlResponsePB*      response) override {
        // The root's initial wake/ACK happens before this peer even queues its control.
        std::this_thread::sleep_for(std::chrono::milliseconds(request->phase() == "expired" ? 1500 : 200));
        return LocalRpcServiceImpl::ControlPressureDiagnosticsInternal(context, request, response);
    }
};

class PressureControlTpChild: public DeviceTestBase {};

TEST_F(PressureControlTpChild, DISABLED_RunRank) {
    ASSERT_NE(std::getenv("RTP_PRESSURE_TP_RANK"), nullptr);
    ASSERT_NE(std::getenv("RTP_PRESSURE_TP_BASE"), nullptr);
    ::alarm(60);
    const int         rank = std::atoi(std::getenv("RTP_PRESSURE_TP_RANK"));
    const std::string base = std::getenv("RTP_PRESSURE_TP_BASE");
    CpuTpBroadcaster::instance().initialize(rank, 2, base);
    ModelConfig   model;
    RuntimeConfig runtime;
    KVCacheConfig cache;
    auto          params               = createEngineInitParams(CustomConfig{}, model, runtime, cache);
    params.runtime_config.warm_up      = false;
    NormalExecutor::test_model_factory = [&](const GptModelInitParams&) {
        return std::make_unique<MockModel>(model.vocab_size, [](const GptModelInputs&) {
            ADD_FAILURE() << "idle control test must never run a model forward";
        });
    };
    // Initialize cache/model locally, then exercise real idle TP execution using
    // UDS. This avoids requiring a CUDA process group for cache-capacity setup.
    auto engine                           = std::make_shared<NormalEngine>(params, nullptr, false);
    engine->parallelism_config.tp_size    = 2;
    engine->parallelism_config.tp_rank    = rank;
    auto* executor                        = static_cast<NormalExecutor*>(engine->executor_.get());
    executor->parallelism_config_.tp_size = 2;
    executor->parallelism_config_.tp_rank = rank;
    executor->tp_rank_                    = rank;
    ASSERT_TRUE(engine->startLoop().ok());
    NormalExecutor::test_model_factory = nullptr;

    if (rank == 1) {
        auto                   server = std::make_shared<PressureTpServer>(engine, rank);
        DelayedPressureService service(server);
        grpc::ServerBuilder    builder;
        int                    port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(&service);
        auto rpc_server = builder.BuildAndStart();
        ASSERT_TRUE(rpc_server);
        writeFile(base + ".port", std::to_string(port));
        ASSERT_TRUE(waitFile(base + ".stop"));
        {
            std::lock_guard<std::mutex> lock(engine->pressure_recorder_->mutex_);
            EXPECT_EQ(engine->pressure_recorder_->session_id_, "peer-only");
            EXPECT_TRUE(engine->pressure_recorder_->phase_.empty()) << "expired phase must never apply later";
        }
        // Stop both loops with one final root idle broadcast, then join normally.
        engine->running_ = false;
        writeFile(base + ".stopping", "1");
        ASSERT_TRUE(waitFile(base + ".root-stopped"));
        EXPECT_TRUE(engine->stop().ok());
        rpc_server->Shutdown();
    } else {
        ASSERT_TRUE(waitFile(base + ".port"));
        int port = 0;
        std::ifstream(base + ".port") >> port;
        PressureTpServer server(engine, rank, "127.0.0.1:" + std::to_string(port));
        auto             control = [&](const std::string&                      session,
                           PressureSessionControlRequestPB::Action action,
                           bool                                    peer_only,
                           const std::string&                      phase = "") {
            PressureSessionControlRequestPB request;
            request.set_session_id(session);
            request.set_action(action);
            request.set_phase(phase);
            if (peer_only)
                request.add_target_ranks(1);
            PressureSessionControlResponsePB response;
            grpc::ServerContext              context;
            const auto                       begin = std::chrono::steady_clock::now();
            EXPECT_TRUE(server.ControlPressureDiagnostics(&context, &request, &response).ok());
            EXPECT_LT(std::chrono::steady_clock::now() - begin, std::chrono::seconds(2));
            EXPECT_EQ(response.ranks_size(), peer_only ? 1 : 2);
            for (const auto& ack : response.ranks()) {
                if (phase == "expired") {
                    EXPECT_FALSE(ack.error().empty());
                    EXPECT_EQ(ack.applied_monotonic_ns(), 0);
                } else {
                    EXPECT_TRUE(ack.error().empty()) << ack.error();
                    EXPECT_GT(ack.applied_monotonic_ns(), 0);
                }
            }
        };
        control("all", PressureSessionControlRequestPB::ENABLE, false);
        control("all", PressureSessionControlRequestPB::MARK_PHASE, true, "peer-phase");
        control("all", PressureSessionControlRequestPB::DISABLE, true);
        control("peer-only", PressureSessionControlRequestPB::ENABLE, true);
        {
            std::lock_guard<std::mutex> lock(engine->pressure_recorder_->mutex_);
            EXPECT_EQ(engine->pressure_recorder_->session_id_, "all");
            EXPECT_TRUE(engine->pressure_recorder_->enabled());
            EXPECT_TRUE(engine->pressure_recorder_->phase_.empty());
        }
        EXPECT_TRUE(engine->getPressureSteps("all", 0, 10).records.empty());
        control("peer-only", PressureSessionControlRequestPB::MARK_PHASE, true, "expired");
        // Let the delayed handler attempt enqueue after its RPC deadline has passed.
        std::this_thread::sleep_for(std::chrono::milliseconds(700));
        writeFile(base + ".stop", "1");
        ASSERT_TRUE(waitFile(base + ".stopping"));
        EXPECT_TRUE(engine->stop().ok());
        writeFile(base + ".root-stopped", "1");
    }
    engine.reset();
    CpuTpBroadcaster::instance().reset();
    ::alarm(0);
}

TEST(PressureControlTpTest, IdleDelayedPeerAndPeerOnlyControlsApplyBeforeDeadline) {
    const char*       tmpdir  = std::getenv("TMPDIR");
    std::string       pattern = std::string(tmpdir ? tmpdir : "/tmp") + "/pressure_tp.XXXXXX";
    std::vector<char> path(pattern.begin(), pattern.end());
    path.push_back('\0');
    ASSERT_NE(::mkdtemp(path.data()), nullptr);
    const std::string  base = std::string(path.data()) + "/b";
    std::vector<pid_t> children;
    for (int rank = 0; rank < 2; ++rank) {
        std::vector<std::string> environment;
        for (char** entry = environ; *entry; ++entry) {
            const std::string value(*entry);
            if (value.rfind("GTEST_", 0) && value.rfind("RTP_PRESSURE_TP_", 0)
                && value.rfind("TEST_PREMATURE_EXIT_FILE=", 0)) {
                environment.push_back(value);
            }
        }
        environment.push_back("RTP_PRESSURE_TP_RANK=" + std::to_string(rank));
        environment.push_back("RTP_PRESSURE_TP_BASE=" + base);
        std::vector<char*> env;
        for (auto& value : environment)
            env.push_back(value.data());
        env.push_back(nullptr);
        std::vector<std::string> arguments = {"/proc/self/exe",
                                              "--gtest_filter=PressureControlTpChild.DISABLED_RunRank",
                                              "--gtest_also_run_disabled_tests",
                                              "--gtest_output="};
        std::vector<char*>       args;
        for (auto& value : arguments)
            args.push_back(value.data());
        args.push_back(nullptr);
        pid_t      pid    = -1;
        const auto result = ::posix_spawn(&pid, "/proc/self/exe", nullptr, nullptr, args.data(), env.data());
        EXPECT_EQ(result, 0);
        if (!result)
            children.push_back(pid);
    }
    for (auto pid : children) {
        int   status = 0;
        pid_t waited;
        do {
            waited = ::waitpid(pid, &status, 0);
        } while (waited < 0 && errno == EINTR);
        EXPECT_EQ(waited, pid);
        EXPECT_TRUE(WIFEXITED(status)) << status;
        if (WIFEXITED(status)) {
            EXPECT_EQ(WEXITSTATUS(status), 0);
        }
    }
    std::filesystem::remove_all(path.data());
}
}  // namespace
}  // namespace rtp_llm
