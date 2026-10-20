#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/TorchProfiler.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServiceImpl.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"

using namespace ::testing;

namespace rtp_llm {

class MockGenerateStream: public GenerateStream {
public:
    MockGenerateStream(const std::shared_ptr<GenerateInput>& input,
                       const ModelConfig&                    model_config,
                       const RuntimeConfig&                  runtime_config):
        GenerateStream(input, model_config, runtime_config, ResourceContext{}, nullptr) {}

    MOCK_METHOD((ErrorResult<GenerateOutputs>), nextOutput, (int64_t), (override));
    MOCK_METHOD(void, updateOutput, (const StreamUpdateInfo&), (override));
};

class TestLocalRpcServer: public LocalRpcServer {
public:
    void configurePressure(int world_rank, int tp_rank = 0, int tp_size = 1, std::vector<std::string> addresses = {}) {
        maga_init_params_.parallelism_config.world_rank    = world_rank;
        maga_init_params_.parallelism_config.tp_rank       = tp_rank;
        maga_init_params_.parallelism_config.tp_size       = tp_size;
        maga_init_params_.runtime_config.worker_grpc_addrs = std::move(addresses);
        meta_                                              = std::make_shared<RpcServerRuntimeMeta>();
    }
    grpc::Status poll(std::shared_ptr<GenerateStream>& stream) {
        return pollStreamOutput(nullptr, "request", nullptr, stream);
    }

    grpc::Status poll(WriterInterface* writer, std::shared_ptr<GenerateStream>& stream) {
        return pollStreamOutput(nullptr, "request", writer, stream);
    }

    ErrorInfo prepare(const GenerateInputPB& input, std::shared_ptr<GenerateInput>& output) {
        return prepareInput(input, output);
    }

    ErrorInfo collect(std::shared_ptr<GenerateStream>& stream) {
        GenerateOutputs last_outputs;
        return collectStreamOutput(nullptr, stream, nullptr, last_outputs);
    }

    void configureAllocatorDump(bool enabled, std::string auth_token, double cooldown_seconds = 0.0) {
        torch_allocator_dump_enabled_                = enabled;
        torch_allocator_dump_auth_token_             = std::move(auth_token);
        torch_allocator_dump_cooldown_seconds_       = cooldown_seconds;
        maga_init_params_.parallelism_config.tp_rank = 0;
        maga_init_params_.parallelism_config.tp_size = 1;
    }

    void configureInternalAllocatorDumpPeer(bool allowed) {
        internal_allocator_dump_peer_allowed_        = allowed;
        maga_init_params_.parallelism_config.tp_size = 2;
    }

    grpc::Status authorizeAllocatorDump(const TorchAllocatorDumpRequestPB& request) const {
        return authorizeTorchAllocatorDump(request);
    }

    grpc::Status beginAllocatorDump(const std::string& dump_id) {
        return beginTorchAllocatorDump(dump_id);
    }

    void finishAllocatorDump() {
        finishTorchAllocatorDump();
    }

    size_t allocatorDumpReplayHistorySize() const {
        return torch_allocator_dump_ids_.size();
    }

    grpc::Status aggregateAllocatorDumpResults(const std::string&                             dump_id,
                                               const std::vector<TorchAllocatorDumpResultPB>& results,
                                               TorchAllocatorDumpResponsePB*                  response) const {
        return aggregateTorchAllocatorDumpResults(dump_id, results, response);
    }

    void setAllocatorDumpCallback(std::function<TorchAllocatorDumpResultPB(const std::string&)> callback) {
        allocator_dump_callback_ = std::move(callback);
    }

    std::future<void> cancellationChecked() {
        return cancellation_checked_.get_future();
    }

    void setEngineForTest(const std::shared_ptr<EngineBase>& engine) {
        engine_ = engine;
    }

    std::atomic<bool> cancelled{false};

protected:
    bool isCancelled(grpc::ServerContext*) const override {
        std::call_once(cancellation_check_once_, [this] { cancellation_checked_.set_value(); });
        return cancelled.load();
    }

    bool isTorchAllocatorDumpInternalPeer(grpc::ServerContext*) const override {
        return internal_allocator_dump_peer_allowed_;
    }

    TorchAllocatorDumpResultPB dumpTorchAllocatorOnCurrentProcess(const std::string& dump_id) override {
        if (allocator_dump_callback_) {
            return allocator_dump_callback_(dump_id);
        }
        TorchAllocatorDumpResultPB result;
        result.set_dump_id(dump_id);
        result.set_success(true);
        return result;
    }

private:
    std::function<TorchAllocatorDumpResultPB(const std::string&)> allocator_dump_callback_;
    bool                                                          internal_allocator_dump_peer_allowed_{false};
    mutable std::once_flag                                        cancellation_check_once_;
    mutable std::promise<void>                                    cancellation_checked_;
};

enum class EngineFailureMode {
    THROW_STD_EXCEPTION,
    THROW_UNKNOWN_EXCEPTION,
    BATCH_SIZE_MISMATCH,
    REJECT_ALL,
    REJECT_ALL_WITH_STREAM_ERROR,
};

class FailingEngine: public EngineBase {
public:
    explicit FailingEngine(EngineFailureMode mode): EngineBase(EngineInitParams()), mode_(mode) {}

    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>&) override {
        if (mode_ == EngineFailureMode::THROW_STD_EXCEPTION) {
            throw std::runtime_error("injected engine enqueue failure");
        }
        if (mode_ == EngineFailureMode::THROW_UNKNOWN_EXCEPTION) {
            throw 42;
        }
        return streams.empty() ? nullptr : streams.front();
    }

    void enqueue(std::shared_ptr<GenerateStream>&) override {}

    std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
    enqueueMultiple(const std::vector<std::shared_ptr<GenerateInput>>& inputs) override {
        if (mode_ == EngineFailureMode::THROW_STD_EXCEPTION) {
            throw std::runtime_error("injected engine enqueue failure");
        }
        if (mode_ == EngineFailureMode::THROW_UNKNOWN_EXCEPTION) {
            throw 42;
        }
        if (mode_ == EngineFailureMode::REJECT_ALL || mode_ == EngineFailureMode::REJECT_ALL_WITH_STREAM_ERROR) {
            if (streams.size() != inputs.size()) {
                throw std::runtime_error("reject-all engine requires one stream per input");
            }
            if (mode_ == EngineFailureMode::REJECT_ALL_WITH_STREAM_ERROR) {
                for (const auto& stream : streams) {
                    stream->reportError(ErrorCode::INVALID_PARAMS, "injected invalid batch parameters");
                }
            }
            return {std::vector<bool>(inputs.size(), false), streams};
        }
        if (streams.size() != inputs.size() + 1) {
            throw std::runtime_error("batch-size mismatch engine requires one extra prebuilt stream");
        }
        return {std::vector<bool>(streams.size(), true), streams};
    }

    absl::Status stop() override {
        return absl::OkStatus();
    }

    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("unused in LocalRpcServerTest");
    }

    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return {};
    }

    std::vector<GenerateStreamPtr> streams;

private:
    EngineFailureMode mode_;
};

class PressureEngine: public FailingEngine {
public:
    PressureEngine(): FailingEngine(EngineFailureMode::REJECT_ALL) {}
    StepWindowProfiler profiler;
    bool               stopTimelineProfiling(const std::string& trace) override {
        return profiler.requestStop(trace);
    }
    ProfileStatus getProfileStatus() const override {
        return profiler.status();
    }
    ErrorResult<CachePressureSnapshot> getCachePressure() const override {
        ++snapshot_calls;
        return CachePressureSnapshot(snapshot);
    }
    mutable int             snapshot_calls = 0;
    CachePressureSnapshot   snapshot;
    PressureStepRecorder    recorder{2};
    std::optional<uint64_t> waiting_requests;
    std::optional<uint64_t> getPressureWaitingRequests() const override {
        return waiting_requests;
    }
    PressureStepPage getPressureSteps(const std::string& session, uint64_t after, size_t limit) const override {
        return recorder.read(session, after, limit);
    }
    ErrorResult<int64_t> controlPressureDiagnostics(const PressureSessionControl& control,
                                                    std::chrono::system_clock::time_point) override {
        try {
            switch (control.action) {
                case PressureControlAction::ENABLE:
                    recorder.enable(control.session_id, control.recorder_capacity);
                    break;
                case PressureControlAction::MARK_PHASE:
                    recorder.markPhase(control.session_id, control.phase);
                    break;
                case PressureControlAction::DISABLE:
                    if (!recorder.disable(control.session_id))
                        return {ErrorCode::INVALID_PARAMS, "session mismatch"};
                    break;
            }
            return pressureClockNs(CLOCK_MONOTONIC);
        } catch (const std::exception& error) {
            return {ErrorCode::INVALID_PARAMS, error.what()};
        }
    }
};
class PressureService: public LocalRpcServiceImpl {
public:
    explicit PressureService(const std::shared_ptr<LocalRpcServer>& server) {
        local_server_ = server;
    }
};

TEST(LocalRpcServerProfileTest, ForwardedStopIsScopedIdempotentAndVisibleInDiagnosticStatus) {
    auto engine = std::make_shared<PressureEngine>();
    engine->profiler.configure(true, "session_window", 0, 20);
    auto server = std::make_shared<TestLocalRpcServer>();
    server->configurePressure(0);
    server->setEngineForTest(engine);
    PressureService       service(server);
    StopProfileRequestPB  request;
    StopProfileResponsePB response;
    grpc::ServerContext   context;
    request.set_trace_name("wrong_window");
    EXPECT_EQ(service.StopProfile(&context, &request, &response).error_code(), grpc::StatusCode::FAILED_PRECONDITION);
    EXPECT_TRUE(engine->profiler.enabled());
    request.set_trace_name("session_window");
    ASSERT_TRUE(service.StopProfile(&context, &request, &response).ok());
    ASSERT_EQ(response.ranks_size(), 1);
    EXPECT_TRUE(response.ranks(0).accepted());
    EXPECT_TRUE(response.ranks(0).profile().collection_complete());
    EXPECT_EQ(response.ranks(0).profile().capture_begin_ns(), 0);
    EXPECT_EQ(response.ranks(0).profile().capture_end_ns(), 0);
    StopProfileInternalRequestPB internal;
    internal.set_trace_name(request.trace_name());
    EXPECT_TRUE(service.StopProfileInternal(&context, &internal, &response).ok());
    EXPECT_TRUE(response.ranks(0).accepted());
    CachePressureRequestPB  poll;
    CachePressureResponsePB snapshot;
    ASSERT_TRUE(service.GetCachePressure(&context, &poll, &snapshot).ok());
    ASSERT_EQ(snapshot.ranks_size(), 1);
    EXPECT_EQ(snapshot.ranks(0).profile().trace_name(), "session_window");
    EXPECT_EQ(snapshot.ranks(0).profile().state(), "finished");
    EXPECT_TRUE(snapshot.ranks(0).profile().stop_requested());
    EXPECT_EQ(snapshot.ranks(0).service_instance(), response.ranks(0).service_instance());
}

TEST(LocalRpcServerProfileTest, UnsupportedAndInvalidTargetsDoNotAcknowledgeStop) {
    auto server = std::make_shared<TestLocalRpcServer>();
    server->configurePressure(0, 0, 2);
    server->setEngineForTest(std::make_shared<FailingEngine>(EngineFailureMode::REJECT_ALL));
    PressureService      service(server);
    StopProfileRequestPB request;
    request.set_trace_name("session_window");
    StopProfileResponsePB response;
    grpc::ServerContext   context;
    EXPECT_EQ(service.StopProfile(&context, &request, &response).error_code(), grpc::StatusCode::UNIMPLEMENTED);
    EXPECT_FALSE(response.ranks(0).accepted());
    request.set_enable_all_rank(true);
    request.add_target_ranks(1);
    request.add_target_ranks(1);
    EXPECT_EQ(service.StopProfile(&context, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(LocalRpcServerProfileTest, AcceptedStopTimeoutDoesNotClaimCollectionOrExport) {
    TestLocalRpcServer server;
    server.configurePressure(0);
    auto engine = std::make_shared<PressureEngine>();
    server.setEngineForTest(engine);
    engine->profiler.configure(true, "session_window", 0, 20);
    std::promise<void> started, release;
    auto               ready = started.get_future();
    auto               gate  = release.get_future();
    std::thread        worker([&] {
        engine->profiler.startStep();
        engine->profiler.finishStep();
        started.set_value();
        gate.wait();
        engine->profiler.serviceControl();
    });
    ready.wait();
    StopProfileRequestPB request;
    request.set_trace_name("session_window");
    StopProfileResponsePB response;
    grpc::ServerContext   context;
    const auto            begin = std::chrono::steady_clock::now();
    EXPECT_TRUE(server.StopProfile(&context, &request, &response).ok());
    EXPECT_LT(std::chrono::steady_clock::now() - begin, std::chrono::seconds(2));
    EXPECT_EQ(response.ranks_size(), 1);
    if (response.ranks_size() == 1) {
        EXPECT_TRUE(response.ranks(0).accepted());
        EXPECT_FALSE(response.ranks(0).error().empty());
        EXPECT_EQ(response.ranks(0).profile().state(), "stopping");
        EXPECT_FALSE(response.ranks(0).profile().collection_complete());
        EXPECT_FALSE(response.ranks(0).profile().export_complete());
        EXPECT_GT(response.ranks(0).profile().capture_begin_ns(), 0);
        EXPECT_EQ(response.ranks(0).profile().capture_end_ns(), 0);
    }
    release.set_value();
    worker.join();
    EXPECT_TRUE(engine->getProfileStatus().collection_complete);
    CachePressureRequestPB  poll;
    CachePressureResponsePB snapshot;
    ASSERT_TRUE(server.GetCachePressure(&context, &poll, &snapshot).ok());
    EXPECT_EQ(snapshot.ranks(0).profile().capture_begin_ns(), engine->getProfileStatus().capture_begin_ns);
    EXPECT_EQ(snapshot.ranks(0).profile().capture_end_ns(), engine->getProfileStatus().capture_end_ns);
    EXPECT_GT(snapshot.ranks(0).profile().capture_end_ns(), snapshot.ranks(0).profile().capture_begin_ns());
}

class RecordingWriter: public LocalRpcServer::WriterInterface {
public:
    bool Write(const GenerateOutputsPB& outputs, grpc::WriteOptions) override {
        outputs_.push_back(outputs);
        return true;
    }

    std::vector<GenerateOutputsPB> outputs_;
};

class FailingWriter: public LocalRpcServer::WriterInterface {
public:
    bool Write(const GenerateOutputsPB&, grpc::WriteOptions) override {
        ++write_count_;
        return false;
    }

    size_t write_count_{0};
};

enum class WakeReason {
    OUTPUT,
    FINISHED,
    STREAM_ERROR,
    TIMEOUT
};

std::shared_ptr<MockGenerateStream> createMockStream() {
    auto input             = std::make_shared<GenerateInput>();
    input->generate_config = std::make_shared<GenerateConfig>();
    input->input_ids       = torch::tensor({1, 2, 3}, torch::kInt32);

    ModelConfig model_config;
    model_config.max_seq_len = 3;
    return std::make_shared<MockGenerateStream>(input, model_config, RuntimeConfig{});
}

std::shared_ptr<NormalGenerateStream> createNormalStream() {
    auto input             = std::make_shared<GenerateInput>();
    input->generate_config = std::make_shared<GenerateConfig>();
    input->begin_time_us   = autil::TimeUtility::currentTimeInMicroSeconds();
    input->input_ids       = torch::tensor({1, 2, 3}, torch::kInt32);

    ModelConfig model_config;
    model_config.max_seq_len = 3;
    return std::make_shared<NormalGenerateStream>(input, model_config, RuntimeConfig{}, ResourceContext{}, nullptr);
}

ErrorResult<GenerateOutputs> wakeResult(WakeReason reason) {
    switch (reason) {
        case WakeReason::OUTPUT: {
            GenerateOutputs outputs;
            return ErrorResult<GenerateOutputs>(std::move(outputs));
        }
        case WakeReason::FINISHED:
            return ErrorResult<GenerateOutputs>(ErrorCode::FINISHED, "finished");
        case WakeReason::STREAM_ERROR:
            return ErrorResult<GenerateOutputs>(ErrorCode::EXECUTION_EXCEPTION, "failed");
        case WakeReason::TIMEOUT:
            return ErrorResult<GenerateOutputs>(ErrorCode::GENERATE_TIMEOUT, "timeout");
    }
    return ErrorResult<GenerateOutputs>(ErrorCode::UNKNOWN_ERROR, "unknown wake reason");
}

void publishWakeError(MockGenerateStream* stream, WakeReason reason) {
    if (reason == WakeReason::STREAM_ERROR) {
        stream->reportError(ErrorCode::EXECUTION_EXCEPTION, "failed");
    } else if (reason == WakeReason::TIMEOUT) {
        stream->reportError(ErrorCode::GENERATE_TIMEOUT, "timeout");
    }
}

ErrorCode expectedStreamError(WakeReason reason) {
    if (reason == WakeReason::STREAM_ERROR) {
        return ErrorCode::EXECUTION_EXCEPTION;
    }
    if (reason == WakeReason::TIMEOUT) {
        return ErrorCode::GENERATE_TIMEOUT;
    }
    return ErrorCode::CANCELLED;
}

TEST(LocalRpcServerTest, AllocatorDumpAuthorizationRequiresEnablementAndSecret) {
    TestLocalRpcServer          server;
    TorchAllocatorDumpRequestPB request;
    request.set_auth_token("secret");
    request.set_dump_id("dump-123");

    EXPECT_EQ(server.authorizeAllocatorDump(request).error_code(), grpc::StatusCode::PERMISSION_DENIED);

    server.configureAllocatorDump(true, "expected-secret");
    EXPECT_EQ(server.authorizeAllocatorDump(request).error_code(), grpc::StatusCode::UNAUTHENTICATED);

    request.set_auth_token("expected-secret");
    EXPECT_TRUE(server.authorizeAllocatorDump(request).ok());
}

TEST(LocalRpcServerTest, AllocatorDumpAuthorizationRejectsUnsafeCorrelationId) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "secret");
    TorchAllocatorDumpRequestPB request;
    request.set_auth_token("secret");
    request.set_dump_id("../leak-path");

    EXPECT_EQ(server.authorizeAllocatorDump(request).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(LocalRpcServerTest, RejectedAllocatorDumpRpcReturnsNoBackendDetails) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "expected-secret");
    grpc::ServerContext          context;
    TorchAllocatorDumpRequestPB  request;
    TorchAllocatorDumpResponsePB response;
    TorchAllocatorDumpResultPB   internal_response;
    request.set_auth_token("wrong-secret");
    request.set_dump_id("dump-123");
    response.add_results()->set_file_path("/stale/private/path");
    internal_response.set_file_path("/stale/private/path");
    internal_response.set_pid(1234);

    const auto status          = server.DumpTorchAllocator(&context, &request, &response);
    const auto internal_status = server.DumpTorchAllocatorInternal(&context, &request, &internal_response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAUTHENTICATED);
    EXPECT_EQ(internal_status.error_code(), grpc::StatusCode::UNAUTHENTICATED);
    EXPECT_EQ(response.results_size(), 0);
    EXPECT_TRUE(internal_response.file_path().empty());
    EXPECT_EQ(internal_response.pid(), 0);
}

TEST(LocalRpcServerTest, DirectAllocatorDumpRpcRejectsReplayAndCooldown) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "secret", 60.0);
    grpc::ServerContext          context;
    TorchAllocatorDumpRequestPB  request;
    TorchAllocatorDumpResponsePB response;
    request.set_auth_token("secret");
    request.set_dump_id("dump-first");

    EXPECT_TRUE(server.DumpTorchAllocator(&context, &request, &response).ok());
    ASSERT_EQ(response.results_size(), 1);
    EXPECT_EQ(response.results(0).dump_id(), "dump-first");

    EXPECT_EQ(server.DumpTorchAllocator(&context, &request, &response).error_code(), grpc::StatusCode::ALREADY_EXISTS);
    EXPECT_EQ(response.results_size(), 0);

    request.set_dump_id("dump-second");
    EXPECT_EQ(server.DumpTorchAllocator(&context, &request, &response).error_code(),
              grpc::StatusCode::RESOURCE_EXHAUSTED);
    EXPECT_EQ(response.results_size(), 0);
}

TEST(LocalRpcServerTest, DirectAllocatorDumpRpcIsSingleFlight) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "secret");
    std::promise<void> dump_started;
    std::promise<void> release_dump;
    auto               release_future = release_dump.get_future().share();
    std::atomic<int>   dump_calls{0};
    server.setAllocatorDumpCallback([&](const std::string& dump_id) {
        if (dump_calls.fetch_add(1) == 0) {
            dump_started.set_value();
            release_future.wait();
        }
        TorchAllocatorDumpResultPB result;
        result.set_dump_id(dump_id);
        result.set_success(true);
        return result;
    });

    TorchAllocatorDumpRequestPB first_request;
    first_request.set_auth_token("secret");
    first_request.set_dump_id("dump-in-flight");
    grpc::ServerContext          first_context;
    TorchAllocatorDumpResponsePB first_response;
    auto                         first_call = std::async(
        std::launch::async, [&] { return server.DumpTorchAllocator(&first_context, &first_request, &first_response); });
    const auto dump_started_status = dump_started.get_future().wait_for(std::chrono::seconds(5));
    if (dump_started_status != std::future_status::ready) {
        release_dump.set_value();
        first_call.wait();
        ADD_FAILURE() << "first allocator dump did not start";
        return;
    }

    TorchAllocatorDumpRequestPB second_request;
    second_request.set_auth_token("secret");
    second_request.set_dump_id("dump-concurrent");
    grpc::ServerContext          second_context;
    TorchAllocatorDumpResponsePB second_response;
    EXPECT_EQ(server.DumpTorchAllocator(&second_context, &second_request, &second_response).error_code(),
              grpc::StatusCode::ABORTED);

    release_dump.set_value();
    ASSERT_EQ(first_call.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_TRUE(first_call.get().ok());
    EXPECT_EQ(dump_calls.load(), 1);
}

TEST(LocalRpcServerTest, InternalAllocatorDumpRequiresTrustedTpPeer) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "secret");
    server.configureInternalAllocatorDumpPeer(false);
    TorchAllocatorDumpRequestPB request;
    request.set_auth_token("secret");
    request.set_dump_id("dump-internal");
    grpc::ServerContext        context;
    TorchAllocatorDumpResultPB response;

    const auto status = server.DumpTorchAllocatorInternal(&context, &request, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::PERMISSION_DENIED);
    EXPECT_TRUE(response.dump_id().empty());
}

TEST(LocalRpcServerTest, InternalAllocatorDumpEnforcesAdmissionAndReplayProtection) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "secret");
    server.configureInternalAllocatorDumpPeer(true);
    std::promise<void> dump_started;
    std::promise<void> release_dump;
    auto               release_future = release_dump.get_future().share();
    server.setAllocatorDumpCallback([&](const std::string& dump_id) {
        dump_started.set_value();
        release_future.wait();
        TorchAllocatorDumpResultPB result;
        result.set_dump_id(dump_id);
        result.set_success(true);
        return result;
    });

    TorchAllocatorDumpRequestPB first_request;
    first_request.set_auth_token("secret");
    first_request.set_dump_id("dump-internal-first");
    grpc::ServerContext        first_context;
    TorchAllocatorDumpResultPB first_response;
    auto                       first_call          = std::async(std::launch::async, [&] {
        return server.DumpTorchAllocatorInternal(&first_context, &first_request, &first_response);
    });
    const auto                 dump_started_status = dump_started.get_future().wait_for(std::chrono::seconds(5));
    if (dump_started_status != std::future_status::ready) {
        release_dump.set_value();
        first_call.wait();
        ADD_FAILURE() << "internal allocator dump did not start";
        return;
    }

    TorchAllocatorDumpRequestPB second_request;
    second_request.set_auth_token("secret");
    second_request.set_dump_id("dump-internal-second");
    grpc::ServerContext        second_context;
    TorchAllocatorDumpResultPB second_response;
    EXPECT_EQ(server.DumpTorchAllocatorInternal(&second_context, &second_request, &second_response).error_code(),
              grpc::StatusCode::ABORTED);

    release_dump.set_value();
    ASSERT_EQ(first_call.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_TRUE(first_call.get().ok());
    EXPECT_EQ(first_response.dump_id(), first_request.dump_id());

    grpc::ServerContext        replay_context;
    TorchAllocatorDumpResultPB replay_response;
    EXPECT_EQ(server.DumpTorchAllocatorInternal(&replay_context, &first_request, &replay_response).error_code(),
              grpc::StatusCode::ALREADY_EXISTS);
}

TEST(LocalRpcServerTest, InternalAllocatorDumpEnforcesCooldownOnEachRank) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "secret", 60.0);
    server.configureInternalAllocatorDumpPeer(true);
    TorchAllocatorDumpRequestPB request;
    request.set_auth_token("secret");
    request.set_dump_id("dump-internal-first");
    grpc::ServerContext        first_context;
    TorchAllocatorDumpResultPB first_response;
    ASSERT_TRUE(server.DumpTorchAllocatorInternal(&first_context, &request, &first_response).ok());

    request.set_dump_id("dump-internal-second");
    grpc::ServerContext        second_context;
    TorchAllocatorDumpResultPB second_response;
    EXPECT_EQ(server.DumpTorchAllocatorInternal(&second_context, &request, &second_response).error_code(),
              grpc::StatusCode::RESOURCE_EXHAUSTED);
}

TEST(LocalRpcServerTest, LeaderAdmissionAllowsExactlyOneMatchingInternalFanout) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "secret");
    server.configureInternalAllocatorDumpPeer(true);
    ASSERT_TRUE(server.beginAllocatorDump("dump-fanout").ok());

    TorchAllocatorDumpRequestPB request;
    request.set_auth_token("secret");
    request.set_dump_id("dump-fanout");
    grpc::ServerContext        context;
    TorchAllocatorDumpResultPB response;
    EXPECT_TRUE(server.DumpTorchAllocatorInternal(&context, &request, &response).ok());
    EXPECT_EQ(response.dump_id(), request.dump_id());

    grpc::ServerContext        replay_context;
    TorchAllocatorDumpResultPB replay_response;
    EXPECT_EQ(server.DumpTorchAllocatorInternal(&replay_context, &request, &replay_response).error_code(),
              grpc::StatusCode::ALREADY_EXISTS);

    EXPECT_EQ(server.beginAllocatorDump("dump-other").error_code(), grpc::StatusCode::ABORTED);
    server.finishAllocatorDump();
}

TEST(LocalRpcServerTest, AllocatorDumpReplayHistoryIsBounded) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "secret");
    for (int index = 0; index <= 1024; ++index) {
        ASSERT_TRUE(server.beginAllocatorDump("dump-" + std::to_string(index)).ok());
        server.finishAllocatorDump();
    }

    EXPECT_EQ(server.allocatorDumpReplayHistorySize(), 1024);
    EXPECT_TRUE(server.beginAllocatorDump("dump-0").ok());
    server.finishAllocatorDump();
    EXPECT_EQ(server.allocatorDumpReplayHistorySize(), 1024);
}

TEST(LocalRpcServerTest, AllocatorDumpAggregationRejectsMismatchedIds) {
    TestLocalRpcServer         server;
    TorchAllocatorDumpResultPB matching;
    matching.set_dump_id("expected-id");
    matching.set_world_rank(0);
    TorchAllocatorDumpResultPB mismatched;
    mismatched.set_dump_id("other-id");
    mismatched.set_world_rank(1);
    TorchAllocatorDumpResponsePB response;
    response.add_results()->set_dump_id("stale-id");

    const auto status = server.aggregateAllocatorDumpResults("expected-id", {matching, mismatched}, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::DATA_LOSS);
    EXPECT_EQ(response.results_size(), 0);
}

TEST(LocalRpcServerTest, PrepareInputReturnsInvalidParamsMappedToInvalidArgument) {
    TestLocalRpcServer server;
    GenerateInputPB    input;
    input.add_token_ids(0);
    input.mutable_generate_config()->set_max_new_tokens(-1);
    auto output = std::make_shared<GenerateInput>();

    ErrorInfo result;
    EXPECT_NO_THROW(result = server.prepare(input, output));

    EXPECT_EQ(result.code(), ErrorCode::INVALID_PARAMS);
    EXPECT_THAT(result.ToString(), HasSubstr("max_new_tokens"));
    EXPECT_EQ(output, nullptr);
    EXPECT_EQ(transErrorCodeToGrpc(result.code()), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(LocalRpcServerTest, GenerateStreamCallQueryConversionFailureUsesInvalidArgumentWithDetails) {
    TestLocalRpcServer server;
    GenerateInputPB    input;
    input.add_token_ids(0);
    input.mutable_generate_config()->set_max_new_tokens(-1);
    grpc::ServerContext context;

    const auto status = server.GenerateStreamCall(&context, &input, nullptr);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    ErrorDetailsPB error_details;
    ASSERT_TRUE(error_details.ParseFromString(status.error_details()));
    EXPECT_EQ(error_details.error_code(), static_cast<int>(ErrorCode::INVALID_PARAMS));
    EXPECT_THAT(error_details.error_message(), HasSubstr("max_new_tokens"));
}

TEST(LocalRpcServerTest, BatchGenerateCallSerializesUnexpectedEngineExceptions) {
    for (const auto mode : {EngineFailureMode::THROW_STD_EXCEPTION, EngineFailureMode::THROW_UNKNOWN_EXCEPTION}) {
        TestLocalRpcServer server;
        server.setEngineForTest(std::make_shared<FailingEngine>(mode));
        BatchGenerateInputPB request;
        for (int i = 0; i < 3; ++i) {
            auto* input = request.add_inputs();
            input->set_request_id(123 + i);
            input->add_token_ids(i);
        }
        grpc::ServerContext    context;
        BatchGenerateOutputsPB response;

        grpc::Status status;
        EXPECT_NO_THROW(status = server.BatchGenerateCall(&context, &request, &response));
        EXPECT_TRUE(status.ok());
        ASSERT_EQ(response.results_size(), request.inputs_size());
        for (const auto& result : response.results()) {
            ASSERT_TRUE(result.has_error_info());
            EXPECT_EQ(result.error_info().error_code(), ErrorCodePB::EXECUTION_EXCEPTION);
            EXPECT_THAT(result.error_info().error_message(),
                        HasSubstr(mode == EngineFailureMode::THROW_STD_EXCEPTION ? "injected engine enqueue failure" :
                                                                                   "unknown exception"));
        }
    }
}

TEST(LocalRpcServerTest, GenerateHandlersRejectNullRpcArgumentsWithoutDereferencingThem) {
    TestLocalRpcServer  server;
    grpc::ServerContext context;

    auto stream_status = server.GenerateStreamCall(&context, nullptr, nullptr);
    EXPECT_EQ(stream_status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    ErrorDetailsPB stream_details;
    ASSERT_TRUE(stream_details.ParseFromString(stream_status.error_details()));
    EXPECT_EQ(stream_details.error_code(), static_cast<int>(ErrorCode::INVALID_PARAMS));
    EXPECT_THAT(stream_details.error_message(), HasSubstr("must not be null"));

    BatchGenerateOutputsPB response;
    auto                   batch_status = server.BatchGenerateCall(&context, nullptr, &response);
    EXPECT_EQ(batch_status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);

    BatchGenerateInputPB request;
    batch_status = server.BatchGenerateCall(&context, &request, nullptr);
    EXPECT_EQ(batch_status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(LocalRpcServerTest, EmptyBatchPreservesLegacyNoOp) {
    TestLocalRpcServer     server;
    BatchGenerateInputPB   request;
    BatchGenerateOutputsPB response;
    response.add_results();
    grpc::ServerContext context;

    const auto status = server.BatchGenerateCall(&context, &request, &response);

    EXPECT_TRUE(status.ok());
    EXPECT_EQ(response.results_size(), 1);
}

TEST(LocalRpcServerTest, GenerateStreamCallRejectsNullWriterBeforeEngineAccess) {
    TestLocalRpcServer server;
    GenerateInputPB    input;
    input.set_request_id(123);
    input.add_token_ids(1);
    grpc::ServerContext context;

    const auto status = server.GenerateStreamCall(&context, &input, nullptr);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    ErrorDetailsPB details;
    ASSERT_TRUE(details.ParseFromString(status.error_details()));
    EXPECT_EQ(details.error_code(), static_cast<int>(ErrorCode::INVALID_PARAMS));
    EXPECT_THAT(details.error_message(), HasSubstr("writer must not be null"));
}

TEST(RpcErrorCodeTest, InvalidParamsUsesDeclaredRpcEnum) {
    EXPECT_EQ(transErrorCodeToRPC(ErrorCode::INVALID_PARAMS), ErrorCodePB::INVALID_PARAMS);
    EXPECT_EQ(transRPCErrorCode(ErrorCodePB::INVALID_PARAMS), ErrorCode::INVALID_PARAMS);
    EXPECT_NE(static_cast<int>(ErrorCodePB::INVALID_PARAMS), static_cast<int>(ErrorCode::INVALID_PARAMS));
}

TEST(LocalRpcServerTest, UninitializedServerRejectsValidBatchRequests) {
    TestLocalRpcServer  server;
    grpc::ServerContext context;

    BatchGenerateInputPB batch_input;
    batch_input.add_inputs()->add_token_ids(1);
    batch_input.add_inputs()->add_token_ids(2);
    BatchGenerateOutputsPB batch_output;
    const auto             batch_status = server.BatchGenerateCall(&context, &batch_input, &batch_output);
    EXPECT_TRUE(batch_status.ok());
    ASSERT_EQ(batch_output.results_size(), batch_input.inputs_size());
    for (const auto& result : batch_output.results()) {
        ASSERT_TRUE(result.has_error_info());
        EXPECT_EQ(result.error_info().error_code(), ErrorCodePB::EXECUTION_EXCEPTION);
        EXPECT_THAT(result.error_info().error_message(), HasSubstr("engine is not initialized"));
    }
}

TEST(LocalRpcServerTest, BatchPrepareInputFailurePreservesCardinalityAndInvalidParams) {
    TestLocalRpcServer   server;
    BatchGenerateInputPB request;
    for (int i = 0; i < 3; ++i) {
        auto* input = request.add_inputs();
        input->add_token_ids(i);
        input->mutable_generate_config()->set_max_new_tokens(1);
    }
    auto* invalid_config = request.mutable_inputs(1)->mutable_generate_config();
    invalid_config->set_max_new_tokens(0);
    invalid_config->set_prefill_only(true);
    invalid_config->set_min_new_tokens(1);

    grpc::ServerContext    context;
    BatchGenerateOutputsPB response;
    const auto             status = server.BatchGenerateCall(&context, &request, &response);

    EXPECT_TRUE(status.ok());
    ASSERT_EQ(response.results_size(), request.inputs_size());
    for (int i = 0; i < response.results_size(); ++i) {
        SCOPED_TRACE(i);
        const auto& result = response.results(i);
        ASSERT_TRUE(result.has_error_info());
        EXPECT_EQ(result.error_info().error_code(), ErrorCodePB::INVALID_PARAMS);
        EXPECT_EQ(transRPCErrorCode(result.error_info().error_code()), ErrorCode::INVALID_PARAMS);
        EXPECT_THAT(result.error_info().error_message(), Not(HasSubstr("multimodal")));
        if (i == 1) {
            EXPECT_THAT(result.error_info().error_message(), HasSubstr("min_new_tokens"));
        } else {
            EXPECT_THAT(result.error_info().error_message(), HasSubstr("invalid parameters at index 1"));
        }
    }
}

TEST(LocalRpcServerTest, BatchGenerateCallReturnsTerminalErrorForRejectedEnqueue) {
    TestLocalRpcServer server;
    auto               engine  = std::make_shared<FailingEngine>(EngineFailureMode::REJECT_ALL);
    auto               stream0 = createMockStream();
    auto               stream1 = createMockStream();
    auto               stream2 = createMockStream();
    engine->streams            = {stream0, stream1, stream2};
    EXPECT_CALL(*stream0, nextOutput(_)).Times(0);
    EXPECT_CALL(*stream1, nextOutput(_)).Times(0);
    EXPECT_CALL(*stream2, nextOutput(_)).Times(0);
    server.setEngineForTest(engine);

    BatchGenerateInputPB request;
    for (int i = 0; i < 3; ++i) {
        request.add_inputs()->add_token_ids(i);
    }
    BatchGenerateOutputsPB response;
    grpc::ServerContext    context;

    const auto status = server.BatchGenerateCall(&context, &request, &response);

    EXPECT_TRUE(status.ok());
    ASSERT_EQ(response.results_size(), request.inputs_size());
    for (int i = 0; i < response.results_size(); ++i) {
        SCOPED_TRACE(i);
        const auto& result = response.results(i);
        ASSERT_TRUE(result.has_error_info());
        EXPECT_EQ(result.error_info().error_code(), ErrorCodePB::EXECUTION_EXCEPTION);
        EXPECT_THAT(result.error_info().error_message(), HasSubstr("rejected input at index " + std::to_string(i)));
    }
}

TEST(LocalRpcServerTest, BatchGenerateCallPassesThroughRejectedStreamError) {
    TestLocalRpcServer server;
    auto               engine  = std::make_shared<FailingEngine>(EngineFailureMode::REJECT_ALL_WITH_STREAM_ERROR);
    auto               stream0 = createMockStream();
    auto               stream1 = createMockStream();
    engine->streams            = {stream0, stream1};
    EXPECT_CALL(*stream0, nextOutput(_)).Times(0);
    EXPECT_CALL(*stream1, nextOutput(_)).Times(0);
    server.setEngineForTest(engine);

    BatchGenerateInputPB request;
    request.add_inputs()->add_token_ids(1);
    request.add_inputs()->add_token_ids(2);
    BatchGenerateOutputsPB response;
    grpc::ServerContext    context;

    const auto status = server.BatchGenerateCall(&context, &request, &response);

    EXPECT_TRUE(status.ok());
    ASSERT_EQ(response.results_size(), request.inputs_size());
    for (const auto& result : response.results()) {
        ASSERT_TRUE(result.has_error_info());
        EXPECT_EQ(result.error_info().error_code(), ErrorCodePB::INVALID_PARAMS);
        EXPECT_THAT(result.error_info().error_message(), HasSubstr("injected invalid batch parameters"));
    }
}

TEST(LocalRpcServerTest, BatchGenerateCallCancelsEnqueuedStreamsOnOuterExceptionAndPreservesCardinality) {
    TestLocalRpcServer server;
    auto               engine = std::make_shared<FailingEngine>(EngineFailureMode::BATCH_SIZE_MISMATCH);
    for (int i = 0; i < 4; ++i) {
        engine->streams.push_back(createMockStream());
    }
    engine->streams.front()->reportEvent(StreamEvents::GenerateDone);
    server.setEngineForTest(engine);

    BatchGenerateInputPB request;
    for (int i = 0; i < 3; ++i) {
        auto* input = request.add_inputs();
        input->set_request_id(100 + i);
        input->add_token_ids(i);
    }
    BatchGenerateOutputsPB response;
    response.add_results();
    grpc::ServerContext context;

    grpc::Status status;
    EXPECT_NO_THROW(status = server.BatchGenerateCall(&context, &request, &response));
    EXPECT_TRUE(status.ok());
    ASSERT_EQ(response.results_size(), request.inputs_size());
    for (const auto& result : response.results()) {
        ASSERT_TRUE(result.has_error_info());
        EXPECT_EQ(result.error_info().error_code(), ErrorCodePB::EXECUTION_EXCEPTION);
        EXPECT_THAT(result.error_info().error_message(), HasSubstr("enqueueMultiple result size mismatch"));
    }
    ASSERT_EQ(engine->streams.size(), request.inputs_size() + 1);
    ASSERT_NE(engine->streams.front(), nullptr);
    EXPECT_FALSE(engine->streams.front()->hasError());
    EXPECT_TRUE(engine->streams.front()->hasEvent(StreamEvents::GenerateDone));
    for (size_t i = 1; i < engine->streams.size(); ++i) {
        ASSERT_NE(engine->streams[i], nullptr);
        EXPECT_TRUE(engine->streams[i]->hasError());
        EXPECT_EQ(engine->streams[i]->statusInfo().code(), ErrorCode::CANCELLED);
    }
}

TEST(LocalRpcServerTest, PollChecksCancellationBeforeHandlingEveryWakeReason) {
    for (const auto reason :
         std::array{WakeReason::OUTPUT, WakeReason::FINISHED, WakeReason::STREAM_ERROR, WakeReason::TIMEOUT}) {
        TestLocalRpcServer server;
        auto               mock_stream     = createMockStream();
        auto*              mock_stream_ptr = mock_stream.get();
        EXPECT_CALL(*mock_stream, nextOutput(_)).WillOnce(InvokeWithoutArgs([&server, mock_stream_ptr, reason] {
            publishWakeError(mock_stream_ptr, reason);
            server.cancelled = true;
            return wakeResult(reason);
        }));
        std::shared_ptr<GenerateStream> stream = mock_stream;

        const auto status = server.poll(stream);

        EXPECT_EQ(status.error_code(), grpc::StatusCode::CANCELLED);
        EXPECT_EQ(stream->statusInfo().code(), expectedStreamError(reason));
    }
}

TEST(LocalRpcServerTest, CollectChecksCancellationBeforeHandlingEveryWakeReason) {
    for (const auto reason :
         std::array{WakeReason::OUTPUT, WakeReason::FINISHED, WakeReason::STREAM_ERROR, WakeReason::TIMEOUT}) {
        TestLocalRpcServer server;
        auto               mock_stream     = createMockStream();
        auto*              mock_stream_ptr = mock_stream.get();
        EXPECT_CALL(*mock_stream, nextOutput(_)).WillOnce(InvokeWithoutArgs([&server, mock_stream_ptr, reason] {
            publishWakeError(mock_stream_ptr, reason);
            server.cancelled = true;
            return wakeResult(reason);
        }));
        std::shared_ptr<GenerateStream> stream = mock_stream;

        const auto status = server.collect(stream);

        EXPECT_EQ(status.code(), ErrorCode::CANCELLED);
        EXPECT_EQ(stream->statusInfo().code(), expectedStreamError(reason));
    }
}

TEST(LocalRpcServerTest, PollInterruptsBlockedNextOutputAfterClientCancellation) {
    TestLocalRpcServer              server;
    auto                            cancellation_checked = server.cancellationChecked();
    std::shared_ptr<GenerateStream> stream               = createNormalStream();
    auto poll_result = std::async(std::launch::async, [&server, &stream] { return server.poll(stream); });

    EXPECT_EQ(cancellation_checked.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    server.cancelled = true;

    const auto wait_status = poll_result.wait_for(std::chrono::seconds(5));
    if (wait_status != std::future_status::ready) {
        stream->reportError(ErrorCode::EXECUTION_EXCEPTION, "test poll cancellation timed out");
    }
    EXPECT_EQ(wait_status, std::future_status::ready);
    EXPECT_EQ(poll_result.get().error_code(), grpc::StatusCode::CANCELLED);
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
}

TEST(LocalRpcServerTest, PollClassifiesClosedResponseStreamAsCancelled) {
    TestLocalRpcServer server;
    FailingWriter      writer;
    auto               mock_stream = createMockStream();
    EXPECT_CALL(*mock_stream, nextOutput(_)).WillOnce(InvokeWithoutArgs([] {
        GenerateOutputs outputs;
        outputs.request_id = 1;
        return ErrorResult<GenerateOutputs>(std::move(outputs));
    }));
    std::shared_ptr<GenerateStream> stream = mock_stream;

    const auto status = server.poll(&writer, stream);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::CANCELLED);
    EXPECT_EQ(status.error_message(), "request output consumer closed");
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
    EXPECT_EQ(writer.write_count_, 1);
}

TEST(LocalRpcServerTest, CollectInterruptsBlockedNextOutputAfterClientCancellation) {
    TestLocalRpcServer              server;
    auto                            cancellation_checked = server.cancellationChecked();
    std::shared_ptr<GenerateStream> stream               = createNormalStream();
    auto collect_result = std::async(std::launch::async, [&server, &stream] { return server.collect(stream); });

    EXPECT_EQ(cancellation_checked.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    server.cancelled = true;

    const auto wait_status = collect_result.wait_for(std::chrono::seconds(5));
    if (wait_status != std::future_status::ready) {
        stream->reportError(ErrorCode::EXECUTION_EXCEPTION, "test collect cancellation timed out");
    }
    EXPECT_EQ(wait_status, std::future_status::ready);
    EXPECT_EQ(collect_result.get().code(), ErrorCode::CANCELLED);
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
}

TEST(LocalRpcServerTest, PollWritesFinalLocalOutputBeforeRemoteHandoff) {
    TestLocalRpcServer              server;
    RecordingWriter                 writer;
    auto                            normal_stream = createNormalStream();
    std::shared_ptr<GenerateStream> stream        = normal_stream;
    normal_stream->setNeedReleaseResource(true);
    normal_stream->generate_status_->status.store(StreamState::RUNNING);

    {
        std::lock_guard<std::mutex> lock(*normal_stream->mutex_);
        GenerateOutputs             outputs;
        outputs.request_id = 123;
        normal_stream->enqueueGenerateOutput(std::move(outputs));
        normal_stream->reportEventWithoutLock(StreamEvents::NeedRemoteGenerate);
    }

    const auto status = server.poll(&writer, stream);

    EXPECT_TRUE(status.ok());
    ASSERT_EQ(writer.outputs_.size(), 1);
    EXPECT_EQ(writer.outputs_[0].request_id(), 123);
    EXPECT_TRUE(stream->hasEvent(StreamEvents::NeedRemoteGenerate));
    EXPECT_EQ(stream->getStatus(), StreamState::RUNNING);
    EXPECT_FALSE(normal_stream->stream_cache_resource_->isResourceReleased());
    EXPECT_FALSE(normal_stream->hasOutput());
}

TEST(PressureRpcTest, UnsupportedEngineDoesNotReturnEmptySuccess) {
    TestLocalRpcServer server;
    server.setEngineForTest(std::make_shared<FailingEngine>(EngineFailureMode::REJECT_ALL));
    CachePressureRequestPB request;
    request.set_session_id("pressure-test");
    CachePressureResponsePB response;
    grpc::ServerContext     context;
    EXPECT_EQ(server.GetCachePressureInternal(&context, &request, &response).error_code(),
              grpc::StatusCode::UNIMPLEMENTED);
    EXPECT_EQ(server.GetCachePressure(&context, &request, &response).error_code(), grpc::StatusCode::UNIMPLEMENTED);
    ASSERT_EQ(response.ranks_size(), 1);
    EXPECT_FALSE(response.ranks(0).error().empty());
}

TEST(PressureRpcTest, SnapshotSerializationPreservesPhysicalDimensionsAndPresence) {
    auto                      engine = std::make_shared<PressureEngine>();
    CachePressurePoolSnapshot pool;
    pool.pool_id                    = "physical-host";
    pool.pool_name                  = "host-name";
    pool.tier                       = Tier::HOST;
    pool.block_size_bytes           = 4096;
    pool.total_blocks               = 100;
    pool.used_blocks                = 70;
    pool.free_blocks                = 30;
    pool.available_blocks           = 40;
    pool.pending_release_blocks     = 7;
    pool.published_host_blocks      = 55;
    pool.active_blocks              = 10;
    pool.request_ref_blocks         = 12;
    pool.block_cache_ref_blocks     = 65;
    pool.load_ref_blocks            = 13;
    pool.eviction_target_ref_blocks = 14;
    pool.store_ref_blocks           = 15;
    engine->snapshot.pools.push_back(pool);
    pool.pool_id                      = "device";
    pool.tier                         = Tier::DEVICE;
    pool.request_ref_blocks_available = true;
    engine->snapshot.pools.push_back(pool);
    for (auto type : {CacheGroupType::FULL, CacheGroupType::SWA, CacheGroupType::LINEAR}) {
        CachePressureGroupSnapshot group;
        group.group_id                = engine->snapshot.groups.size();
        group.group_type              = type;
        group.device_pool_ids         = {"device"};
        group.host_pool_id            = "physical-host";
        group.device_watermark        = {0.1, 0.2};
        group.host_watermark          = {0.3, 0.4};
        group.disk_watermark          = {0.5, 0.6};
        group.watermark_trigger_count = 17;
        engine->snapshot.groups.push_back(group);
    }
    for (auto tier : {Tier::DEVICE, Tier::HOST, Tier::DISK, Tier::REMOTE, Tier::NONE}) {
        CachePressureTransferSnapshot transfer;
        transfer.group_id          = 2;
        transfer.pool_id           = "endpoint";
        transfer.pool_tier         = tier;
        transfer.source_tier       = Tier::DEVICE;
        transfer.target_tier       = Tier::HOST;
        transfer.operation         = CacheTransferOperation::EVICT;
        transfer.copied_blocks     = 2;
        transfer.copied_bytes      = 8192;
        transfer.published_blocks  = 1;
        transfer.published_bytes   = 4096;
        transfer.failed_operations = 3;
        engine->snapshot.transfers.push_back(transfer);
    }
    for (auto op : {CacheTransferOperation::LOAD, CacheTransferOperation::STORE}) {
        CachePressureTransferSnapshot transfer;
        transfer.operation = op;
        engine->snapshot.transfers.push_back(transfer);
    }
    auto server = std::make_shared<TestLocalRpcServer>();
    server->configurePressure(4);
    server->setEngineForTest(engine);
    PressureService         service(server);
    CachePressureRequestPB  request;
    CachePressureResponsePB response;
    grpc::ServerContext     context;
    ASSERT_TRUE(service.GetCachePressure(&context, &request, &response).ok());
    CachePressureResponsePB roundtrip;
    ASSERT_TRUE(roundtrip.ParseFromString(response.SerializeAsString()));
    ASSERT_EQ(roundtrip.ranks_size(), 1);
    EXPECT_EQ(roundtrip.requested_ranks(0), 4);
    EXPECT_EQ(roundtrip.schema_version(), 1);
    const auto& rank = roundtrip.ranks(0);
    EXPECT_EQ(rank.world_rank(), 4);
    EXPECT_FALSE(rank.service_instance().empty());
    EXPECT_EQ(rank.sequence(), 1);
    EXPECT_GT(rank.sampled_monotonic_ns(), 0);
    EXPECT_GE(rank.collect_duration_ns(), 0);
    EXPECT_LE(rank.coordinator_send_ns(), rank.sampled_monotonic_ns());
    EXPECT_GE(rank.coordinator_receive_ns(), rank.sampled_monotonic_ns() + rank.collect_duration_ns());
    EXPECT_FALSE(rank.has_activity());
    EXPECT_FALSE(rank.step_history_available());
    ASSERT_EQ(rank.pools_size(), 2);
    const auto& pb = rank.pools(0);
    EXPECT_EQ(pb.pool_id(), "physical-host");
    EXPECT_EQ(pb.pool_name(), "host-name");
    EXPECT_EQ(pb.tier(), PRESSURE_HOST);
    EXPECT_EQ(pb.block_size_bytes(), 4096);
    EXPECT_EQ(pb.total_blocks(), 100);
    EXPECT_EQ(pb.used_blocks(), 70);
    EXPECT_EQ(pb.free_blocks(), 30);
    EXPECT_EQ(pb.available_blocks(), 40);
    EXPECT_EQ(pb.pending_release_blocks(), 7);
    EXPECT_EQ(pb.published_host_blocks(), 55);
    EXPECT_EQ(pb.active_blocks(), 10);
    EXPECT_EQ(pb.block_cache_ref_blocks(), 65);
    EXPECT_EQ(pb.load_ref_blocks(), 13);
    EXPECT_EQ(pb.eviction_target_ref_blocks(), 14);
    EXPECT_EQ(pb.store_ref_blocks(), 15);
    EXPECT_FALSE(pb.has_request_ref_blocks());
    EXPECT_TRUE(rank.pools(1).has_request_ref_blocks());
    EXPECT_EQ(rank.pools(1).request_ref_blocks().value(), 12);
    EXPECT_EQ(rank.pools(1).tier(), PRESSURE_DEVICE);
    ASSERT_EQ(rank.groups_size(), 3);
    EXPECT_EQ(rank.groups(0).group_type(), PRESSURE_FULL);
    EXPECT_EQ(rank.groups(1).group_type(), PRESSURE_SWA);
    EXPECT_EQ(rank.groups(2).group_type(), PRESSURE_LINEAR);
    EXPECT_EQ(rank.groups(2).group_id(), 2);
    EXPECT_EQ(rank.groups(0).device_pool_ids(0), "device");
    EXPECT_EQ(rank.groups(0).host_pool_id().value(), "physical-host");
    EXPECT_FALSE(rank.groups(0).has_disk_pool_id());
    EXPECT_DOUBLE_EQ(rank.groups(0).device_watermark().low_ratio(), 0.1);
    EXPECT_DOUBLE_EQ(rank.groups(0).host_watermark().high_ratio(), 0.4);
    EXPECT_DOUBLE_EQ(rank.groups(0).disk_watermark().high_ratio(), 0.6);
    EXPECT_EQ(rank.groups(0).watermark_trigger_count(), 17);
    ASSERT_EQ(rank.transfers_size(), 7);
    const std::array<PressureTierPB, 5> tiers = {
        PRESSURE_DEVICE, PRESSURE_HOST, PRESSURE_DISK, PRESSURE_REMOTE, PRESSURE_NONE};
    for (int i = 0; i < 5; ++i) {
        const auto& transfer = rank.transfers(i);
        EXPECT_EQ(transfer.group_id(), 2);
        EXPECT_EQ(transfer.pool_id(), "endpoint");
        EXPECT_EQ(transfer.pool_tier(), tiers[i]);
        EXPECT_EQ(transfer.source_tier(), PRESSURE_DEVICE);
        EXPECT_EQ(transfer.target_tier(), PRESSURE_HOST);
        EXPECT_EQ(transfer.operation(), PRESSURE_EVICT);
        EXPECT_EQ(transfer.copied_blocks(), 2);
        EXPECT_EQ(transfer.copied_bytes(), 8192);
        EXPECT_EQ(transfer.published_blocks(), 1);
        EXPECT_EQ(transfer.published_bytes(), 4096);
        EXPECT_EQ(transfer.failed_operations(), 3);
    }
    EXPECT_EQ(rank.transfers(5).operation(), PRESSURE_LOAD);
    EXPECT_EQ(rank.transfers(6).operation(), PRESSURE_STORE);
}

TEST(PressureRpcTest, ValidatesTargetsPageCapAndRankZero) {
    TestLocalRpcServer server;
    server.configurePressure(4, 0, 2);
    auto engine = std::make_shared<PressureEngine>();
    server.setEngineForTest(engine);
    CachePressureRequestPB  request;
    CachePressureResponsePB response;
    grpc::ServerContext     context;
    request.add_target_ranks(4);
    request.add_target_ranks(4);
    EXPECT_EQ(server.GetCachePressure(&context, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    request.mutable_target_ranks()->Set(1, 6);
    EXPECT_EQ(server.GetCachePressure(&context, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    request.clear_target_ranks();
    request.set_max_step_records(1025);
    EXPECT_EQ(server.GetCachePressure(&context, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(server.GetCachePressureInternal(&context, &request, &response).error_code(),
              grpc::StatusCode::INVALID_ARGUMENT);
    request.set_max_step_records(1024);
    EXPECT_TRUE(server.GetCachePressureInternal(&context, &request, &response).ok());
    server.configurePressure(5, 1, 2);
    EXPECT_EQ(server.GetCachePressure(&context, &request, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_TRUE(server.GetCachePressureInternal(&context, &request, &response).ok());
    EXPECT_EQ(response.ranks(0).world_rank(), 5);
}

TEST(PressureRpcTest, SequenceAndInstanceResetAreExplicit) {
    auto                    engine = std::make_shared<PressureEngine>();
    CachePressureRequestPB  request;
    CachePressureResponsePB response;
    grpc::ServerContext     context;
    TestLocalRpcServer      first;
    first.setEngineForTest(engine);
    ASSERT_TRUE(first.GetCachePressureInternal(&context, &request, &response).ok());
    const auto instance = response.ranks(0).service_instance();
    ASSERT_TRUE(first.GetCachePressureInternal(&context, &request, &response).ok());
    EXPECT_EQ(response.ranks(0).sequence(), 2);
    EXPECT_EQ(response.ranks(0).service_instance(), instance);
    TestLocalRpcServer restarted;
    restarted.setEngineForTest(engine);
    ASSERT_TRUE(restarted.GetCachePressureInternal(&context, &request, &response).ok());
    EXPECT_EQ(response.ranks(0).sequence(), 1);
    EXPECT_NE(response.ranks(0).service_instance(), instance);
}

TEST(PressureRpcTest, MissingRankPreservesLocalEvidence) {
    TestLocalRpcServer server;
    server.configurePressure(4, 0, 2);
    server.setEngineForTest(std::make_shared<PressureEngine>());
    grpc::ServerContext     context;
    CachePressureRequestPB  request;
    CachePressureResponsePB response;
    ASSERT_TRUE(server.GetCachePressure(&context, &request, &response).ok());
    ASSERT_EQ(response.ranks_size(), 2);
    EXPECT_EQ(response.requested_ranks(0), 4);
    EXPECT_EQ(response.requested_ranks(1), 5);
    EXPECT_EQ(response.ranks(0).sequence(), 1);
    EXPECT_TRUE(response.ranks(0).error().empty());
    EXPECT_EQ(response.ranks(1).world_rank(), 5);
    EXPECT_FALSE(response.ranks(1).error().empty());
}

TEST(PressureRpcTest, HealthAndOrdinaryStatusNeverSnapshot) {
    TestLocalRpcServer server;
    server.configurePressure(0);
    auto engine = std::make_shared<PressureEngine>();
    server.setEngineForTest(engine);
    grpc::ServerContext   context;
    EmptyPB               empty;
    CheckHealthResponsePB health;
    CacheVersionPB        version;
    CacheStatusPB         cache;
    StatusVersionPB       worker_version;
    WorkerStatusPB        worker;
    EXPECT_TRUE(server.CheckHealth(&context, &empty, &health).ok());
    EXPECT_TRUE(server.GetCacheStatus(&context, &version, &cache).ok());
    EXPECT_TRUE(server.GetWorkerStatus(&context, &worker_version, &worker).ok());
    EXPECT_EQ(engine->snapshot_calls, 0);
}

TEST(PressureRpcTest, FanoutUsesOnlyInternalMethodAndChecksReturnedIdentity) {
    auto peer_server = std::make_shared<TestLocalRpcServer>();
    peer_server->configurePressure(5, 1, 2);
    auto peer_engine = std::make_shared<PressureEngine>();
    peer_server->setEngineForTest(peer_engine);
    PressureService     peer_service(peer_server);
    grpc::ServerBuilder builder;
    int                 port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&peer_service);
    auto grpc_server = builder.BuildAndStart();
    ASSERT_NE(grpc_server, nullptr);
    TestLocalRpcServer coordinator;
    coordinator.configurePressure(4, 0, 2, {"", "127.0.0.1:" + std::to_string(port)});
    coordinator.setEngineForTest(std::make_shared<PressureEngine>());
    CachePressureRequestPB  request;
    CachePressureResponsePB response;
    grpc::ServerContext     context;
    ASSERT_TRUE(coordinator.GetCachePressure(&context, &request, &response).ok());
    ASSERT_EQ(response.ranks_size(), 2);
    EXPECT_EQ(response.ranks(1).world_rank(), 5);
    EXPECT_TRUE(response.ranks(1).error().empty());
    EXPECT_EQ(peer_engine->snapshot_calls, 1);
    EXPECT_LE(response.ranks(1).coordinator_send_ns(), response.ranks(1).coordinator_receive_ns());
    peer_server->configurePressure(6, 1, 2);
    ASSERT_TRUE(coordinator.GetCachePressure(&context, &request, &response).ok());
    EXPECT_EQ(response.ranks(1).world_rank(), 5);
    EXPECT_EQ(response.ranks(1).error(), "invalid rank response");
    grpc_server->Shutdown();
}

TEST(PressureRpcTest, RepeatedThreePeerControlAndSnapshotPreserveTransportProgress) {
    std::vector<std::shared_ptr<TestLocalRpcServer>> peers;
    std::vector<std::unique_ptr<PressureService>>    services;
    std::vector<std::unique_ptr<grpc::Server>>       servers;
    std::vector<std::string>                         addresses(1, "");
    for (int rank = 1; rank < 4; ++rank) {
        auto peer = std::make_shared<TestLocalRpcServer>();
        peer->configurePressure(rank, rank, 4);
        peer->setEngineForTest(std::make_shared<PressureEngine>());
        services.push_back(std::make_unique<PressureService>(peer));
        grpc::ServerBuilder builder;
        int                 port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(services.back().get());
        servers.push_back(builder.BuildAndStart());
        ASSERT_NE(servers.back(), nullptr);
        addresses.push_back("127.0.0.1:" + std::to_string(port));
        peers.push_back(std::move(peer));
    }
    TestLocalRpcServer coordinator;
    coordinator.configurePressure(0, 0, 4, addresses);
    coordinator.setEngineForTest(std::make_shared<PressureEngine>());
    for (int iteration = 0; iteration < 16; ++iteration) {
        SCOPED_TRACE(iteration);
        PressureSessionControlRequestPB control;
        control.set_session_id("repeated-transport");
        control.set_action(PressureSessionControlRequestPB::ENABLE);
        PressureSessionControlResponsePB acknowledgement;
        grpc::ServerContext              control_context;
        ASSERT_TRUE(coordinator.ControlPressureDiagnostics(&control_context, &control, &acknowledgement).ok());
        ASSERT_EQ(acknowledgement.ranks_size(), 4);
        for (int rank = 0; rank < 4; ++rank) {
            EXPECT_TRUE(acknowledgement.ranks(rank).error().empty())
                << "control rank " << rank << ": " << acknowledgement.ranks(rank).error();
        }
        CachePressureRequestPB request;
        request.set_session_id("repeated-transport");
        CachePressureResponsePB response;
        grpc::ServerContext     context;
        const auto              begin = std::chrono::steady_clock::now();
        ASSERT_TRUE(coordinator.GetCachePressure(&context, &request, &response).ok());
        EXPECT_LT(std::chrono::steady_clock::now() - begin, std::chrono::milliseconds(500));
        ASSERT_EQ(response.ranks_size(), 4);
        for (int rank = 0; rank < 4; ++rank) {
            EXPECT_EQ(response.ranks(rank).world_rank(), rank);
            EXPECT_TRUE(response.ranks(rank).error().empty())
                << "snapshot rank " << rank << ": " << response.ranks(rank).error();
        }
    }
    for (auto& server : servers) {
        server->Shutdown();
    }
}

class TimeoutPressureService: public RpcService::Service {
public:
    grpc::Status ControlPressureDiagnosticsInternal(grpc::ServerContext* context,
                                                    const PressureSessionControlRequestPB*,
                                                    PressureSessionControlResponsePB*) override {
        while (!context->IsCancelled()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return {grpc::StatusCode::CANCELLED, "deadline"};
    }
    grpc::Status GetCachePressureInternal(grpc::ServerContext* context,
                                          const CachePressureRequestPB*,
                                          CachePressureResponsePB*) override {
        while (!context->IsCancelled()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return {grpc::StatusCode::CANCELLED, "deadline"};
    }
};
TEST(PressureRpcTest, PeerDeadlineRetainsSuccessfulRankAndBoundsWait) {
    TimeoutPressureService service;
    grpc::ServerBuilder    builder;
    int                    port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    auto grpc_server = builder.BuildAndStart();
    ASSERT_NE(grpc_server, nullptr);
    TestLocalRpcServer coordinator;
    coordinator.configurePressure(0, 0, 2, {"", "127.0.0.1:" + std::to_string(port)});
    coordinator.setEngineForTest(std::make_shared<PressureEngine>());
    CachePressureRequestPB  request;
    CachePressureResponsePB response;
    grpc::ServerContext     context;
    const auto              begin = std::chrono::steady_clock::now();
    ASSERT_TRUE(coordinator.GetCachePressure(&context, &request, &response).ok());
    EXPECT_LT(std::chrono::steady_clock::now() - begin, std::chrono::seconds(3));
    ASSERT_EQ(response.ranks_size(), 2);
    EXPECT_TRUE(response.ranks(0).error().empty());
    EXPECT_FALSE(response.ranks(1).error().empty());
    EXPECT_GE(response.ranks(1).coordinator_receive_ns(), response.ranks(1).coordinator_send_ns());
    grpc_server->Shutdown();
}

TEST(PressureRpcTest, ControlWrappersValidateAndKeepActiveProjectionAcrossCursors) {
    auto server = std::make_shared<TestLocalRpcServer>();
    server->configurePressure(0);
    auto engine = std::make_shared<PressureEngine>();
    server->setEngineForTest(engine);
    PressureService                  service(server);
    grpc::ServerContext              context;
    PressureSessionControlRequestPB  control;
    PressureSessionControlResponsePB ack;
    EXPECT_EQ(service.ControlPressureDiagnostics(&context, &control, &ack).error_code(),
              grpc::StatusCode::INVALID_ARGUMENT);
    control.set_session_id("s");
    control.set_action(PressureSessionControlRequestPB::ENABLE);
    control.set_recorder_capacity(65537);
    EXPECT_EQ(service.ControlPressureDiagnosticsInternal(&context, &control, &ack).error_code(),
              grpc::StatusCode::INVALID_ARGUMENT);
    control.set_recorder_capacity(2);
    ASSERT_TRUE(service.ControlPressureDiagnostics(&context, &control, &ack).ok());
    ASSERT_EQ(ack.ranks_size(), 1);
    EXPECT_GT(ack.ranks(0).applied_monotonic_ns(), 0);
    EXPECT_FALSE(ack.ranks(0).service_instance().empty());
    control.set_action(PressureSessionControlRequestPB::MARK_PHASE);
    control.set_recorder_capacity(0);
    control.set_phase("measurement");
    ASSERT_TRUE(service.ControlPressureDiagnosticsInternal(&context, &control, &ack).ok());
    PressureStepRecord step;
    step.step_id              = 1;
    step.context_batch        = 2;
    step.tokens               = 8;
    step.begin_ns             = 10;
    step.forward_enter_ns     = 20;
    step.forward_enter_cpu_ns = 5;
    step.request_ids          = {"s:measurement:1", "s:measurement:2"};
    step.stream_ids           = {101, 102};
    engine->recorder.begin(step);
    CachePressureRequestPB request;
    request.set_session_id("s");
    (*request.mutable_after_step_sequence())[0] = 99;
    request.set_max_step_records(1);
    CachePressureResponsePB response;
    ASSERT_TRUE(service.GetCachePressure(&context, &request, &response).ok());
    const auto& rank = response.ranks(0);
    EXPECT_TRUE(rank.step_history_available());
    ASSERT_EQ(rank.steps_size(), 1);
    EXPECT_FALSE(rank.steps(0).completed());
    EXPECT_EQ(rank.steps(0).sequence(), 0);
    EXPECT_EQ(rank.next_step_sequence(), 99);
    EXPECT_EQ(rank.steps(0).source_request_ids(0), "s:measurement:1");
    EXPECT_EQ(rank.steps(0).request_ids(0), 101);
    EXPECT_EQ(rank.steps(0).preparation_begin_ns(), 10);
    EXPECT_EQ(rank.steps(0).wall_start_ns(), 20);
    EXPECT_EQ(rank.steps(0).executed_tokens(), 8);
    EXPECT_EQ(rank.activity().running_requests().value(), 2);
    step.complete          = true;
    step.forward_return_ns = step.end_ns = 40;
    step.forward_return_cpu_ns           = 15;
    engine->recorder.append(step);
    (*request.mutable_after_step_sequence())[0] = 0;
    ASSERT_TRUE(service.GetCachePressureInternal(&context, &request, &response).ok());
    EXPECT_TRUE(response.ranks(0).steps(0).completed());
    EXPECT_EQ(response.ranks(0).steps(0).cpu_end_ns(), 15);
    EXPECT_EQ(response.ranks(0).activity().running_requests().value(), 0);
    EXPECT_EQ(response.ranks(0).next_step_sequence(), 1);
    request.set_session_id("other");
    ASSERT_TRUE(service.GetCachePressureInternal(&context, &request, &response).ok());
    EXPECT_FALSE(response.ranks(0).step_history_available());
    EXPECT_EQ(response.ranks(0).steps_size(), 0);
    EXPECT_FALSE(response.ranks(0).has_activity());
    control.set_action(PressureSessionControlRequestPB::DISABLE);
    control.set_session_id("other");
    EXPECT_EQ(service.ControlPressureDiagnostics(&context, &control, &ack).error_code(),
              grpc::StatusCode::FAILED_PRECONDITION);
    control.set_session_id("s");
    ASSERT_TRUE(service.ControlPressureDiagnostics(&context, &control, &ack).ok());
    ASSERT_TRUE(service.ControlPressureDiagnosticsInternal(&context, &control, &ack).ok());
}

TEST(PressureRpcTest, ProjectsObservedWaitingRequestsWithoutInventingUnsupportedZero) {
    TestLocalRpcServer server;
    server.configurePressure(0);
    auto engine = std::make_shared<PressureEngine>();
    engine->recorder.enable("waiting", 8);
    server.setEngineForTest(engine);
    grpc::ServerContext    context;
    CachePressureRequestPB request;
    request.set_session_id("waiting");
    CachePressureResponsePB response;
    for (const auto count : {0u, 3u}) {
        engine->waiting_requests = count;
        ASSERT_TRUE(server.GetCachePressureInternal(&context, &request, &response).ok());
        ASSERT_TRUE(response.ranks(0).activity().has_waiting_requests());
        EXPECT_EQ(response.ranks(0).activity().waiting_requests().value(), count);
    }
    engine->waiting_requests.reset();
    ASSERT_TRUE(server.GetCachePressureInternal(&context, &request, &response).ok());
    EXPECT_FALSE(response.ranks(0).activity().has_waiting_requests());
}

TEST(PressureRpcTest, ExplicitSharedOwnerZeroRoundTripsWithoutChangingLocalCounters) {
    TestLocalRpcServer server;
    server.configurePressure(1, 1, 2);
    auto                   engine = std::make_shared<PressureEngine>();
    CachePressureOwnership ownership;
    ownership.mode                        = CachePressureOwnership::Mode::SHARED_TP;
    ownership.owner_world_rank            = 0;
    ownership.participant_world_ranks     = {0, 1};
    ownership.all_participant_publication = true;
    engine->snapshot.ownership            = ownership;
    CachePressurePoolSnapshot pool;
    pool.pool_id      = "peer-local";
    pool.total_blocks = pool.free_blocks = 127;
    engine->snapshot.pools.push_back(pool);
    CachePressureGroupSnapshot group;
    group.group_tags    = {"full"};
    group.block_tokens  = 512;
    group.payload_bytes = 4096;
    group.storage_bytes = 4112;
    group.crc_enabled   = true;
    engine->snapshot.groups.push_back(group);
    server.setEngineForTest(engine);
    grpc::ServerContext     context;
    CachePressureRequestPB  request;
    CachePressureResponsePB response;
    ASSERT_TRUE(server.GetCachePressureInternal(&context, &request, &response).ok());
    CachePressureResponsePB decoded;
    ASSERT_TRUE(decoded.ParseFromString(response.SerializeAsString()));
    const auto& rank = decoded.ranks(0);
    ASSERT_TRUE(rank.has_ownership());
    ASSERT_TRUE(rank.ownership().has_owner_world_rank());
    EXPECT_EQ(rank.ownership().owner_world_rank().value(), 0);
    EXPECT_EQ(rank.ownership().mode(), PressureOwnershipPB::SHARED_TP);
    EXPECT_EQ(rank.ownership().publication_scope(), PressureOwnershipPB::ALL_PARTICIPANTS);
    EXPECT_EQ(rank.ownership().participant_world_ranks_size(), 2);
    EXPECT_EQ(rank.pools(0).used_blocks(), 0);
    EXPECT_EQ(rank.pools(0).published_host_blocks(), 0);
    EXPECT_EQ(rank.groups(0).block_tokens(), 512);
    EXPECT_EQ(rank.groups(0).payload_bytes(), 4096);
    EXPECT_EQ(rank.groups(0).storage_bytes(), 4112);
    EXPECT_TRUE(rank.groups(0).crc_enabled());
}

TEST(PressureRpcTest, ControlDeadlineRetainsSuccessfulRankAndUnsupportedIsExplicit) {
    TimeoutPressureService service;
    grpc::ServerBuilder    builder;
    int                    port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    auto grpc_server = builder.BuildAndStart();
    ASSERT_TRUE(grpc_server);
    TestLocalRpcServer coordinator;
    coordinator.configurePressure(0, 0, 2, {"", "127.0.0.1:" + std::to_string(port)});
    coordinator.setEngineForTest(std::make_shared<PressureEngine>());
    PressureSessionControlRequestPB request;
    request.set_session_id("deadline");
    request.set_action(PressureSessionControlRequestPB::ENABLE);
    PressureSessionControlResponsePB response;
    grpc::ServerContext              context;
    const auto                       begin = std::chrono::steady_clock::now();
    ASSERT_TRUE(coordinator.ControlPressureDiagnostics(&context, &request, &response).ok());
    EXPECT_LT(std::chrono::steady_clock::now() - begin, std::chrono::seconds(3));
    ASSERT_EQ(response.ranks_size(), 2);
    EXPECT_TRUE(response.ranks(0).error().empty());
    EXPECT_GT(response.ranks(0).applied_monotonic_ns(), 0);
    EXPECT_FALSE(response.ranks(1).error().empty());
    EXPECT_EQ(response.ranks(1).applied_monotonic_ns(), 0);
    coordinator.configurePressure(0);
    coordinator.setEngineForTest(std::make_shared<FailingEngine>(EngineFailureMode::REJECT_ALL));
    EXPECT_EQ(coordinator.ControlPressureDiagnostics(&context, &request, &response).error_code(),
              grpc::StatusCode::UNIMPLEMENTED);
    EXPECT_EQ(coordinator.ControlPressureDiagnosticsInternal(&context, &request, &response).error_code(),
              grpc::StatusCode::UNIMPLEMENTED);
    grpc_server->Shutdown();
}

TEST(PressureRpcTest, ControlFanoutUsesInternalWrapperAndPreservesMissingRank) {
    auto peer = std::make_shared<TestLocalRpcServer>();
    peer->configurePressure(1, 1, 3);
    auto peer_engine = std::make_shared<PressureEngine>();
    peer->setEngineForTest(peer_engine);
    PressureService     service(peer);
    grpc::ServerBuilder builder;
    int                 port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    auto grpc_server = builder.BuildAndStart();
    ASSERT_TRUE(grpc_server);
    TestLocalRpcServer server;
    server.configurePressure(0, 0, 3, {"", "127.0.0.1:" + std::to_string(port), ""});
    server.setEngineForTest(std::make_shared<PressureEngine>());
    grpc::ServerContext             context;
    PressureSessionControlRequestPB request;
    request.set_session_id("fanout");
    request.set_action(PressureSessionControlRequestPB::ENABLE);
    PressureSessionControlResponsePB response;
    ASSERT_TRUE(server.ControlPressureDiagnostics(&context, &request, &response).ok());
    ASSERT_EQ(response.ranks_size(), 3);
    EXPECT_GT(response.ranks(0).applied_monotonic_ns(), 0);
    EXPECT_GT(response.ranks(1).applied_monotonic_ns(), 0);
    EXPECT_NE(response.ranks(0).service_instance(), response.ranks(1).service_instance());
    EXPECT_FALSE(response.ranks(2).error().empty());
    EXPECT_TRUE(peer_engine->recorder.enabled());
    request.add_target_ranks(1);
    request.add_target_ranks(1);
    EXPECT_EQ(server.ControlPressureDiagnostics(&context, &request, &response).error_code(),
              grpc::StatusCode::INVALID_ARGUMENT);
    grpc_server->Shutdown();
}

}  // namespace rtp_llm
