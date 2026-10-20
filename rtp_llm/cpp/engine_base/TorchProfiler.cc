#include "rtp_llm/cpp/engine_base/TorchProfiler.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "autil/TimeUtility.h"
#include <c10/core/impl/VirtualGuardImpl.h>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <string>

namespace rtp_llm {
namespace tap = torch::autograd::profiler;

namespace {
// Trust-boundary limits for configure(). Every profiling config source (HTTP after
// Python sanitize, gRPC/ARPC without Python involvement, service-wide config) funnels
// through StepWindowProfiler::configure, so enforcing them here covers all callers.
constexpr size_t kMaxTraceNameLen = 64;
constexpr int    kMaxNumSteps     = 1000;

int currentProfileDevice() {
#if USING_CUDA || USING_ROCM
    // PyTorch's ROCm backend also registers the CUDA device type.
    c10::impl::VirtualGuardImpl guard(c10::DeviceType::CUDA);
    if (guard.deviceCount() > 0) {
        return guard.getDevice().index();
    }
#endif
    return -1;
}

void drainProfileDevice(int device) {
#if USING_CUDA || USING_ROCM
    if (device >= 0) {
        // Drain all streams on the device recorded at profiler start, including
        // asynchronous draft/commit work. A stream-only wait can miss its tail.
        c10::impl::VirtualGuardImpl(c10::DeviceType::CUDA).synchronizeDevice(device);
    }
#else
    (void)device;
#endif
}

std::string sanitizeTraceName(const std::string& trace_name) {
    std::string out;
    out.reserve(std::min(trace_name.size(), kMaxTraceNameLen));
    for (char c : trace_name) {
        if (out.size() >= kMaxTraceNameLen) {
            break;
        }
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-') {
            out.push_back(c);
        }
    }
    return out;
}
}  // namespace
namespace detail {

std::unique_ptr<tap::ProfilerResult> tryDisableProfiler(DisableProfilerFn            disable_profiler,
                                                        const std::function<void()>& drain_device) {
    bool drained = true;
    if (drain_device) {
        try {
            // This scope must end before disableProfiler finalizes its events.
            // It is profiling teardown, not part of the measured forward.
            RTP_LLM_PROFILE_SCOPE("profiler.teardown(device_drain)");
            drain_device();
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("failed to drain profiling device; dropping incomplete trace: %s", e.what());
            drained = false;
        } catch (...) {
            RTP_LLM_LOG_WARNING("failed to drain profiling device with unknown exception; dropping incomplete trace");
            drained = false;
        }
    }
    // Even a failed drain must pop the active Kineto state exactly once. In
    // particular, unwinding a device failure must not retry from the destructor.
    try {
        auto result = disable_profiler();
        return drained ? std::move(result) : nullptr;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("failed to disable torch profiler; dropping trace and continuing inference: %s", e.what());
    } catch (...) {
        RTP_LLM_LOG_WARNING(
            "failed to disable torch profiler with unknown exception; dropping trace and continuing inference");
    }
    return nullptr;
}

}  // namespace detail

// ---- TorchProfile ----

std::atomic<size_t> TorchProfile::count_{0};

TorchProfile::TorchProfile(const std::string& prefix, std::string output_dir):
    prefix_(prefix), output_dir_(output_dir.empty() ? "." : std::move(output_dir)) {}

TorchProfile::~TorchProfile() {
    if (!stopped_) {
        if (owner_thread_ != std::this_thread::get_id()) {
            RTP_LLM_LOG_ERROR("active torch profiler destroyed off owner thread; shutdown was not serviced");
            return;
        }
        try {
            stop();
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("failed to save torch profiler trace during destruction: %s", e.what());
        } catch (...) {
            RTP_LLM_LOG_WARNING("failed to save torch profiler trace during destruction with unknown exception");
        }
    }
}

void TorchProfile::start() {
    owner_thread_   = std::this_thread::get_id();
    profile_device_ = activities_.count(tpi::ActivityType::CUDA) ? currentProfileDevice() : -1;
    count_ += 1;
    stopped_ = false;
    tap::prepareProfiler(config_, activities_);
    tap::enableProfiler(config_, activities_);
}

std::pair<std::unique_ptr<tap::ProfilerResult>, std::string> TorchProfile::stopAndCollect() {
    if (stopped_) {
        return {nullptr, ""};
    }
    if (owner_thread_ != std::this_thread::get_id()) {
        throw std::logic_error("torch profiler collection requires its owner thread");
    }
    // disableProfiler() pops the active profiler state before finalizing the
    // trace. Mark this wrapper stopped first so an exception is never retried
    // from the destructor.
    stopped_ = true;
    auto res = detail::tryDisableProfiler(&tap::disableProfiler, [this] { drainProfileDevice(profile_device_); });
    if (!res) {
        return {nullptr, ""};
    }
    std::string file_name = output_dir_ + "/" + prefix_ + std::to_string(count_) + ".json";
    return {std::move(res), std::move(file_name)};
}

void TorchProfile::stop() {
    auto [res, file_name] = stopAndCollect();
    if (res) {
        res->save(file_name);
    }
}

// ---- ProfilerSaveWorker ----

ProfilerSaveWorker::ProfilerSaveWorker(): thread_([this] { run(); }) {}

ProfilerSaveWorker::~ProfilerSaveWorker() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    cv_.notify_one();
    thread_.join();
}

void ProfilerSaveWorker::enqueue(std::unique_ptr<tap::ProfilerResult>    result,
                                 std::string                             file_name,
                                 std::function<void(const std::string&)> completed) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        tasks_.push({std::move(result), std::move(file_name), std::move(completed)});
    }
    cv_.notify_one();
}

void ProfilerSaveWorker::run() {
    while (true) {
        SaveTask task;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) {
                return;
            }
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        RTP_LLM_LOG_INFO("saving profiler trace to %s (async)", task.file_name.c_str());
        std::string error;
        try {
            // Kineto's JSON logger can report I/O failures only to stderr.
            // Truncate first to rule out a stale artifact from a prior process.
            {
                std::ofstream output(task.file_name, std::ios::trunc);
                if (!output)
                    throw std::runtime_error("cannot open trace output");
            }
            task.result->save(task.file_name);
            std::ifstream output(task.file_name, std::ios::binary | std::ios::ate);
            if (!output || output.tellg() <= 0)
                throw std::runtime_error("trace output missing or empty");
            auto end  = output.tellg();
            char last = 0;
            do {
                end -= std::streamoff(1);
                output.seekg(end);
                output.get(last);
            } while (end > 0 && std::isspace(static_cast<unsigned char>(last)));
            if (!output || last != '}')
                throw std::runtime_error("trace output was not fully written");
            RTP_LLM_LOG_INFO("profiler trace saved: %s", task.file_name.c_str());
        } catch (const std::exception& e) {
            error = e.what();
            RTP_LLM_LOG_ERROR("failed to save profiler trace %s: %s", task.file_name.c_str(), e.what());
        } catch (...) {
            error = "unknown trace export failure";
        }
        if (task.completed) {
            task.completed(error);
        }
    }
}

// ---- StepWindowProfiler ----

StepWindowProfiler::StepWindowProfiler(const std::string& default_output_dir, int world_rank):
    default_output_dir_(default_output_dir.empty() ? "." : default_output_dir), world_rank_(world_rank) {
    status_.supported = true;
}

StepWindowProfiler::StepScope::StepScope(StepWindowProfiler& profiler): profiler_(profiler) {
    profiler_.beginStep();
}

StepWindowProfiler::StepScope::~StepScope() {
    try {
        profiler_.endStep();
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("timeline profiler endStep failed: %s", e.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("timeline profiler endStep failed with unknown exception");
    }
}

void StepWindowProfiler::configure(bool enable, const std::string& trace_name, int start_step, int num_steps) {
    const auto safe_name = sanitizeTraceName(trace_name);
    if (!enable) {
        requestStop(safe_name);
        return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    // Completed legacy prefixes may be reused for counter-suffixed captures.
    // Only actual cancellation prevents a window from being resurrected.
    if (status_.state != "finished" || (!safe_name.empty() && safe_name == last_cancelled_trace_)) {
        return;
    }
    trace_name_        = safe_name;
    status_            = ProfileStatus{};
    status_.supported  = true;
    status_.state      = "armed";
    status_.trace_name = safe_name;
    start_step_.store(std::max(0, start_step));
    num_steps_.store(num_steps > 0 ? std::min(num_steps, kMaxNumSteps) : 3);
    waited_steps_   = 0;
    profiled_steps_ = 0;
    enabled_.store(true, std::memory_order_release);
}

bool StepWindowProfiler::requestStop(const std::string& trace_name) {
    std::lock_guard<std::mutex> lock(mu_);
    if (trace_name != trace_name_
        || (trace_name.empty() && status_.trace_name.empty() && status_.state == "finished"
            && !status_.collection_complete)) {
        return !trace_name.empty() && trace_name == last_finished_trace_;
    }
    if (status_.state != "finished" && !status_.collection_complete) {
        last_cancelled_trace_ = trace_name;
    }
    enabled_.store(false, std::memory_order_release);
    status_.stop_requested = true;
    if (status_.state == "armed") {
        status_.state               = "finished";
        status_.collection_complete = true;
        status_.export_complete     = true;  // No collector or artifact was created.
        last_finished_trace_        = trace_name_;
    } else if (status_.state == "active") {
        status_.state = "stopping";
    }
    return true;
}

ProfileStatus StepWindowProfiler::status() const {
    std::lock_guard<std::mutex> lock(mu_);
    return status_;
}

void StepWindowProfiler::configureFromConfig(const ProfilingDebugLoggingConfig& cfg) {
    if (cfg.gen_timeline_sync) {
        configure(true, cfg.timeline_trace_name, cfg.timeline_start_step, cfg.timeline_num_steps);
    }
}

void StepWindowProfiler::beginStep() {
    startStep();
}
void StepWindowProfiler::endStep() {
    finishStep();
}

void StepWindowProfiler::startStep() {
    serviceControl();
    if (!enabled_.load(std::memory_order_acquire)) {
        return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (!enabled_.load(std::memory_order_acquire) || profiler_) {
        return;
    }
    if (waited_steps_ < start_step_.load()) {
        ++waited_steps_;
        return;
    }
    owner_thread_      = std::this_thread::get_id();
    std::string prefix = trace_name_;
    if (prefix.empty()) {
        prefix = "profiler_ts" + std::to_string(autil::TimeUtility::currentTimeInMicroSeconds());
    }
    if (prefix.back() != '_') {
        prefix += "_";
    }
    prefix += "wr" + std::to_string(world_rank_) + "_";
    profiler_ = std::make_shared<TorchProfile>(prefix, default_output_dir_);
    try {
        profiler_->start();
        has_profiler_.store(true, std::memory_order_release);
        status_.state = "active";
    } catch (const std::exception& error) {
        enabled_.store(false, std::memory_order_release);
        profiler_.reset();
        status_.state               = "finished";
        status_.collection_complete = true;
        status_.collection_error    = error.what();
        last_finished_trace_        = trace_name_;
    }
}

void StepWindowProfiler::finishStep() {
    if (!has_profiler_.load(std::memory_order_acquire)) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (owner_thread_ != std::this_thread::get_id()) {
            return;
        }
        if (++profiled_steps_ >= num_steps_.load()) {
            enabled_.store(false, std::memory_order_release);
        }
    }
    serviceControl();
}

void StepWindowProfiler::serviceControl() {
    if (!enabled_.load(std::memory_order_acquire) && has_profiler_.load(std::memory_order_acquire)) {
        stopProfiler("control_or_window_complete");
    }
}

StepWindowProfiler::~StepWindowProfiler() {
    // Engines must service shutdown before joining their loop. Never pop
    // another thread's Kineto state from a control-thread destructor.
    if (owner_thread_ == std::this_thread::get_id()) {
        stopProfiler("owner_destructor");
    }
}

void StepWindowProfiler::stopProfiler(const char* reason) {
    std::shared_ptr<TorchProfile> profiler;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!profiler_ || owner_thread_ != std::this_thread::get_id()) {
            return;
        }
        enabled_.store(false, std::memory_order_release);
        status_.state = "stopping";
        profiler      = std::move(profiler_);
    }
    // Device drain/finalization can stall. Keep snapshot and requestStop usable
    // so external callers can enforce their own bounded cleanup deadlines.
    auto [result, file_name] = profiler->stopAndCollect();
    {
        std::lock_guard<std::mutex> lock(mu_);
        has_profiler_.store(false, std::memory_order_release);
        status_.collection_complete = true;
        status_.file_name           = file_name;
        if (!result) {
            status_.collection_error = "profiler collection failed";
            status_.state            = "finished";
            last_finished_trace_     = trace_name_;
            return;
        }
    }
    save_worker_.enqueue(std::move(result), std::move(file_name), [this](const std::string& error) {
        std::lock_guard<std::mutex> lock(mu_);
        status_.export_complete = error.empty();
        status_.export_error    = error;
        status_.state           = "finished";
        last_finished_trace_    = trace_name_;
    });
    RTP_LLM_LOG_INFO("timeline profiler collected: reason=%s", reason);
}

}  // namespace rtp_llm
