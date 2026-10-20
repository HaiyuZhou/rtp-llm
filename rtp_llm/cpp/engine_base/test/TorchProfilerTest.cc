#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <future>
#include <filesystem>

#include "c10/util/Exception.h"
#include "gtest/gtest.h"
#include "kineto/GenericTraceActivity.h"
#include "kineto/IActivityProfiler.h"
#include "kineto/libkineto.h"
#include "kineto/output_base.h"
#include "rtp_llm/cpp/engine_base/TorchProfiler.h"

namespace rtp_llm {

struct TorchProfileTestPeer {
    static void useCpuActivities(TorchProfile& profile) {
        profile.activities_ = {tpi::ActivityType::CPU};
    }
    static std::thread::id owner(const StepWindowProfiler& profile) {
        return profile.owner_thread_;
    }
    static int64_t steps(const StepWindowProfiler& profile) {
        return profile.profiled_steps_;
    }
};

namespace {

namespace tap = torch::autograd::profiler;
namespace tpi = torch::profiler::impl;

class DuplicateFlowSession: public libkineto::IActivityProfilerSession {
public:
    DuplicateFlowSession(): span_(0, 0, "duplicate_flow_span") {
        for (int64_t id = 1; id <= 2; ++id) {
            libkineto::GenericTraceActivity activity(
                span_, libkineto::ActivityType::CUDA_RUNTIME, "duplicate_flow_launch_" + std::to_string(id));
            activity.startTime = 100 + id;
            activity.endTime   = 200 + id;
            // Match CuptiActivity exactly: flowId() returns correlationId().
            // The two distinct API activities carry the same bad backend ID.
            activity.id         = 42;
            activity.flow.id    = 42;
            activity.flow.type  = libkineto::kLinkAsyncCpuGpu;
            activity.flow.start = true;
            activities_.push_back(std::move(activity));
        }
    }

    void start() override {
        status_ = libkineto::TraceStatus::RECORDING;
    }

    void stop() override {
        status_ = libkineto::TraceStatus::PROCESSING;
    }

    std::vector<std::string> errors() override {
        return {};
    }

    void processTrace(libkineto::ActivityLogger& logger) override {
        for (const auto& activity : activities_) {
            logger.handleGenericActivity(activity);
        }
    }

    std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override {
        return nullptr;
    }

    std::vector<libkineto::ResourceInfo> getResourceInfos() override {
        return {};
    }

    std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override {
        return nullptr;
    }

private:
    libkineto::TraceSpan                         span_;
    std::vector<libkineto::GenericTraceActivity> activities_;
};

class DuplicateFlowProfiler: public libkineto::IActivityProfiler {
public:
    const std::string& name() const override {
        return name_;
    }

    const std::set<libkineto::ActivityType>& availableActivities() const override {
        return activities_;
    }

    std::unique_ptr<libkineto::IActivityProfilerSession> configure(const std::set<libkineto::ActivityType>&,
                                                                   const libkineto::Config&) override {
        return std::make_unique<DuplicateFlowSession>();
    }

    std::unique_ptr<libkineto::IActivityProfilerSession>
    configure(int64_t,
              int64_t,
              const std::set<libkineto::ActivityType>& activity_types,
              const libkineto::Config&                 config) override {
        return configure(activity_types, config);
    }

private:
    std::string                       name_{"duplicate_flow_profiler"};
    std::set<libkineto::ActivityType> activities_{libkineto::ActivityType::CUDA_RUNTIME};
};

void startCpuProfiler() {
    tpi::ProfilerConfig         config(tpi::ProfilerState::KINETO, /*report_input_shapes=*/false);
    std::set<tpi::ActivityType> activities{tpi::ActivityType::CPU};
    tap::prepareProfiler(config, activities);
    tap::enableProfiler(config, activities);
}

std::unique_ptr<tap::ProfilerResult> throwProfilerError() {
    throw std::runtime_error("synthetic profiler finalization failure");
}

std::unique_ptr<tap::ProfilerResult> throwUnknownProfilerError() {
    throw 1;
}

std::unique_ptr<tap::ProfilerResult> returnProfilerResult() {
    return std::make_unique<tap::ProfilerResult>();
}

TEST(TorchProfilerTest, DisableResultIsReturned) {
    EXPECT_NE(detail::tryDisableProfiler(&returnProfilerResult), nullptr);
}

TEST(TorchProfilerTest, DisableFailureIsNonFatal) {
    EXPECT_EQ(detail::tryDisableProfiler(&throwProfilerError), nullptr);
}

TEST(TorchProfilerTest, UnknownDisableFailureIsNonFatal) {
    EXPECT_EQ(detail::tryDisableProfiler(&throwUnknownProfilerError), nullptr);
}

TEST(TorchProfilerTest, DrainCompletesBeforeProfilerIsDisabled) {
    static bool drained;
    drained      = false;
    auto disable = +[]() {
        EXPECT_TRUE(drained);
        return returnProfilerResult();
    };
    auto result = detail::tryDisableProfiler(disable, [] { drained = true; });
    EXPECT_TRUE(drained);
    EXPECT_NE(result, nullptr);
}

TEST(TorchProfilerTest, DrainFailureStillDisablesOnceAndDropsIncompleteTrace) {
    static int disable_calls;
    disable_calls = 0;
    auto disable  = +[]() {
        ++disable_calls;
        return returnProfilerResult();
    };
    auto result = detail::tryDisableProfiler(disable, [] { throw std::runtime_error("synthetic device failure"); });
    EXPECT_EQ(disable_calls, 1);
    EXPECT_EQ(result, nullptr);
}

TEST(TorchProfilerTest, UnknownDrainFailureStillDisablesOnce) {
    static int disable_calls;
    disable_calls = 0;
    auto disable  = +[]() {
        ++disable_calls;
        return returnProfilerResult();
    };
    EXPECT_EQ(detail::tryDisableProfiler(disable, [] { throw 1; }), nullptr);
    EXPECT_EQ(disable_calls, 1);
}

TEST(TorchProfilerTest, CpuStopRecordsTeardownAndIsIdempotent) {
    TorchProfile profile("teardown_test_");
    TorchProfileTestPeer::useCpuActivities(profile);
    profile.start();
    auto [result, name] = profile.stopAndCollect();
    ASSERT_NE(result, nullptr);
    EXPECT_FALSE(name.empty());
    size_t teardown_events = 0;
    for (const auto& event : result->events()) {
        teardown_events += event.name() == "profiler.teardown(device_drain)";
    }
    EXPECT_EQ(teardown_events, 1);
    auto [second_result, second_name] = profile.stopAndCollect();
    EXPECT_EQ(second_result, nullptr);
    EXPECT_TRUE(second_name.empty());
}

TEST(TorchProfilerTest, DestructorDoesNotThrowWhenTraceCannotBeSaved) {
    EXPECT_NO_THROW({
        // /dev/null is a file, so no trace file can be created beneath it.
        TorchProfile profile("teardown_test_", "/dev/null/not_a_directory");
        TorchProfileTestPeer::useCpuActivities(profile);
        profile.start();
    });
}

TEST(TorchProfilerTest, WorkerCancellationCollectsWithoutAnotherForwardAndExports) {
    StepWindowProfiler profile("/tmp");
    profile.configure(true, "task12_worker", 0, 20);
    std::promise<void> started, service;
    auto               ready   = started.get_future();
    auto               proceed = service.get_future();
    std::thread        worker([&] {
        profile.startStep();
        profile.finishStep();
        started.set_value();
        proceed.wait();
        profile.serviceControl();
    });
    ready.wait();
    EXPECT_EQ(profile.status().state, "active");
    EXPECT_EQ(TorchProfileTestPeer::owner(profile), worker.get_id());
    EXPECT_FALSE(profile.requestStop("other_window"));
    EXPECT_TRUE(profile.requestStop("task12_worker"));
    EXPECT_EQ(profile.status().state, "stopping");
    profile.serviceControl();  // Wrong-thread service must leave the collector alone.
    EXPECT_FALSE(profile.status().collection_complete);
    profile.configure(true, "replacement", 0, 1);
    EXPECT_EQ(profile.status().trace_name, "task12_worker");
    EXPECT_EQ(ready.wait_for(std::chrono::milliseconds(1)), std::future_status::ready);
    service.set_value();
    worker.join();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (profile.status().state != "finished" && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    const auto status = profile.status();
    EXPECT_EQ(status.state, "finished");
    EXPECT_TRUE(status.collection_complete);
    EXPECT_TRUE(status.collection_error.empty()) << status.collection_error;
    EXPECT_TRUE(status.export_complete) << status.export_error;
    EXPECT_TRUE(std::filesystem::exists(status.file_name));
    EXPECT_EQ(TorchProfileTestPeer::steps(profile), 1);
    EXPECT_TRUE(profile.requestStop("task12_worker"));
    if (const auto* fixture_dir = std::getenv("RTP_PROFILE_FIXTURE_DIR"); fixture_dir && status.export_complete) {
        std::filesystem::create_directories(fixture_dir);
        std::filesystem::copy_file(status.file_name,
                                   std::string(fixture_dir) + "/task12-worker-toolchain.json",
                                   std::filesystem::copy_options::overwrite_existing);
    }
    if (!status.file_name.empty())
        std::filesystem::remove(status.file_name);
}

TEST(TorchProfilerTest, AsyncSaverFailureIsObservable) {
    StepWindowProfiler profile("/dev/null/not_a_directory");
    profile.configure(true, "task12_save_failure", 0, 1);
    std::thread worker([&] {
        profile.startStep();
        profile.finishStep();
    });
    worker.join();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (profile.status().state != "finished" && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_EQ(profile.status().state, "finished");
    EXPECT_TRUE(profile.status().collection_complete);
    EXPECT_TRUE(profile.status().collection_error.empty());
    EXPECT_FALSE(profile.status().export_complete);
    EXPECT_FALSE(profile.status().export_error.empty());
}

TEST(TorchProfilerTest, RealDuplicateFlowDoesNotEscapeWrapper) {
    libkineto::api().registerProfilerFactory([]() { return std::make_unique<DuplicateFlowProfiler>(); });

    startCpuProfiler();
    try {
        // Newer PyTorch versions handle this malformed backend trace and
        // return a result. The incident version takes the catch branch below.
        EXPECT_NE(tap::disableProfiler(), nullptr);
    } catch (const c10::Error& e) {
        EXPECT_NE(std::string(e.what()).find("collection.cpp"), std::string::npos);
    }

    startCpuProfiler();
    EXPECT_NO_THROW((void)detail::tryDisableProfiler(&tap::disableProfiler));
}

}  // namespace
}  // namespace rtp_llm
