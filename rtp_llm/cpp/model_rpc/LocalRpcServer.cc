#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <unistd.h>
#include <c10/core/InferenceMode.h>
#include "autil/Scope.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/utils/TorchCudaOom.h"
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/multimodal_processor/MMProcessorConfig.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/config/EplbConfig.h"
#include "rtp_llm/cpp/config/RoleTypes.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/telemetry/PhaseSpanSynthesizer.h"

using namespace std;

namespace rtp_llm {

namespace {
constexpr int64_t kRpcOutputWaitTimeoutMs        = 500;
constexpr size_t  kAllocatorDumpReplayHistoryMax = 1024;

std::string endpointHost(std::string endpoint) {
    const auto scheme_separator = endpoint.find(':');
    if (endpoint.compare(0, scheme_separator, "ipv4") == 0 || endpoint.compare(0, scheme_separator, "ipv6") == 0) {
        endpoint.erase(0, scheme_separator + 1);
    }
    if (!endpoint.empty() && endpoint.front() == '[') {
        const auto closing_bracket = endpoint.find(']');
        return closing_bracket == std::string::npos ? std::string() : endpoint.substr(1, closing_bracket - 1);
    }
    const auto port_separator = endpoint.rfind(':');
    return port_separator == std::string::npos ? endpoint : endpoint.substr(0, port_separator);
}

bool constantTimeEquals(const std::string& lhs, const std::string& rhs) {
    size_t       difference = lhs.size() ^ rhs.size();
    const size_t length     = std::max(lhs.size(), rhs.size());
    for (size_t index = 0; index < length; ++index) {
        const unsigned char left  = index < lhs.size() ? static_cast<unsigned char>(lhs[index]) : 0;
        const unsigned char right = index < rhs.size() ? static_cast<unsigned char>(rhs[index]) : 0;
        difference |= left ^ right;
    }
    return difference == 0;
}

bool validAllocatorDumpId(const std::string& dump_id) {
    if (dump_id.empty() || dump_id.size() > 64) {
        return false;
    }
    return std::all_of(dump_id.begin(), dump_id.end(), [](unsigned char character) {
        return (character >= '0' && character <= '9') || (character >= 'A' && character <= 'Z')
               || (character >= 'a' && character <= 'z') || character == '-';
    });
}

std::string formatRequestLogTag(const std::string& request_key, const RequestInfo& request_info) {
    std::string tag = "request [" + request_key + "]";
    if (!request_info.trace_id.empty()) {
        tag += ", trace id [" + request_info.trace_id + "]";
    }
    if (!request_info.request_id.empty()) {
        tag += ", source request id [" + request_info.request_id + "]";
    }
    if (!request_info.frontend_ip.empty()) {
        tag += ", frontend ip [" + request_info.frontend_ip + "]";
    }
    if (!request_info.dash_ip.empty()) {
        tag += ", dash ip [" + request_info.dash_ip + "]";
    }
    if (!request_info.source_role.empty()) {
        tag += ", source role [" + request_info.source_role + "]";
    }
    return tag;
}

class BatchStreamCleanupGuard {
public:
    explicit BatchStreamCleanupGuard(std::vector<GenerateStreamPtr>& streams): streams_(streams) {}

    ~BatchStreamCleanupGuard() {
        if (dismissed_ || streams_.empty()) {
            return;
        }

        size_t cancelled_streams = 0;
        for (size_t i = handled_stream_count_; i < streams_.size(); ++i) {
            const auto& stream = streams_[i];
            if (!stream) {
                continue;
            }
            try {
                if (stream->getStatus() == StreamState::FINISHED || stream->hasError()
                    || stream->hasEvent(StreamEvents::GenerateDone)
                    || stream->hasEvent(StreamEvents::NeedRemoteGenerate)) {
                    continue;
                }
                stream->reportError(ErrorCode::CANCELLED, "batch generate call exited before all streams finished");
                ++cancelled_streams;
            } catch (const std::exception& e) {
                RTP_LLM_LOG_ERROR("failed to cancel batch stream [%ld]: %s", stream->streamId(), e.what());
            } catch (...) {
                RTP_LLM_LOG_ERROR("failed to cancel batch stream [%ld]: unknown exception", stream->streamId());
            }
        }
        RTP_LLM_LOG_WARNING("BatchGenerateCall cleanup completed, cancelled_streams=%zu", cancelled_streams);
    }

    void markHandled(size_t stream_index) noexcept {
        if (handled_stream_count_ <= stream_index) {
            handled_stream_count_ = stream_index + 1;
        }
    }

    void dismiss() noexcept {
        dismissed_ = true;
    }

private:
    std::vector<GenerateStreamPtr>& streams_;
    size_t                          handled_stream_count_{0};
    bool                            dismissed_{false};
};

}  // namespace

grpc::Status LocalRpcServer::init(const EngineInitParams&                       maga_init_params,
                                  std::unique_ptr<ProposeModelEngineInitParams> propose_params,
                                  py::object                                    mm_process_engine) {
    return initWithEngineLoop(maga_init_params, std::move(propose_params), std::move(mm_process_engine), true);
}

grpc::Status LocalRpcServer::initWithEngineLoop(const EngineInitParams&                       maga_init_params,
                                                std::unique_ptr<ProposeModelEngineInitParams> propose_params,
                                                py::object                                    mm_process_engine,
                                                bool                                          start_engine_loop) {
    meta_.reset(new RpcServerRuntimeMeta());
    maga_init_params_                      = maga_init_params;
    weight_manager_                        = maga_init_params.weight_manager;
    metrics_reporter_                      = maga_init_params.metrics_reporter;
    torch_allocator_dump_enabled_          = false;
    torch_allocator_dump_auth_token_       = "";
    torch_allocator_dump_cooldown_seconds_ = 60.0;
    {
        std::lock_guard<std::mutex> lock(torch_allocator_dump_mutex_);
        torch_allocator_dump_in_progress_              = false;
        torch_allocator_dump_has_completed_            = false;
        torch_allocator_dump_active_started_by_public_ = false;
        torch_allocator_dump_active_internal_started_  = false;
        torch_allocator_dump_active_id_.clear();
        torch_allocator_dump_ids_.clear();
        torch_allocator_dump_id_order_.clear();
    }
    if (!maga_init_params_.server_config.is_none()) {
        const auto& server_config = maga_init_params_.server_config;
        if (py::hasattr(server_config, "enable_torch_allocator_dump")) {
            torch_allocator_dump_enabled_ = server_config.attr("enable_torch_allocator_dump").cast<bool>();
        }
        if (py::hasattr(server_config, "torch_allocator_dump_auth_token")) {
            torch_allocator_dump_auth_token_ =
                server_config.attr("torch_allocator_dump_auth_token").cast<std::string>();
        }
        if (py::hasattr(server_config, "torch_allocator_dump_cooldown_seconds")) {
            torch_allocator_dump_cooldown_seconds_ =
                server_config.attr("torch_allocator_dump_cooldown_seconds").cast<double>();
        }
    }
    if (!std::isfinite(torch_allocator_dump_cooldown_seconds_) || torch_allocator_dump_cooldown_seconds_ < 0) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            "allocator dump cooldown must be finite and non-negative");
    }
    if (torch_allocator_dump_enabled_ && torch_allocator_dump_auth_token_.empty()) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            "allocator dump authentication token is required when enabled");
    }
    RTP_LLM_LOG_INFO("LocalRpcServer aux_string %s", maga_init_params_.misc_config.aux_string.c_str());
    propose_maga_init_params_ = propose_params.get();
    if (maga_init_params_.parallelism_config.tp_rank == 0
        && !maga_init_params_.runtime_config.worker_grpc_addrs.empty()) {
        tp_broadcaster_ = std::make_shared<BroadcastManager>(maga_init_params_.runtime_config.worker_grpc_addrs);
        if (!tp_broadcaster_->init()) {
            RTP_LLM_LOG_WARNING("failed to init TP broadcaster");
            tp_broadcaster_.reset();
        }
    }

    const auto mm_decision = resolveAndLogMMProcessorKind(maga_init_params.model_config_.mm_model_config.is_multimodal,
                                                          maga_init_params.vit_config.vit_separation,
                                                          !mm_process_engine.is_none(),
                                                          maga_init_params.pd_sep_config.role_type,
                                                          maga_init_params.parallelism_config.tp_rank,
                                                          maga_init_params.model_config_.model_type,
                                                          "LocalRpcServer");
    const auto mm_kind     = mm_decision.kind;
    if (!mm_decision.ok()) {
        RTP_LLM_LOG_ERROR("%s", mm_decision.error.c_str());
        return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, mm_decision.error);
    }

    {
        pybind11::gil_scoped_release release;
        RTP_LLM_CHECK_WITH_INFO(!PyGILState_Check(),
                                "running engine init with gil held may cause program hang, please check");
        engine_.reset(new NormalEngine(maga_init_params, std::move(propose_params), start_engine_loop));
    }
    if (mm_kind == MMProcessorKind::LOCAL) {
        mm_processor_.reset(new LocalMultimodalProcessor(mm_process_engine,
                                                         maga_init_params.model_config_.mm_model_config,
                                                         maga_init_params.model_config_.max_seq_len));
    } else if (mm_kind == MMProcessorKind::REMOTE) {
        mm_processor_.reset(new RemoteMultimodalProcessor(maga_init_params.model_config_.mm_model_config,
                                                          maga_init_params.model_config_.max_seq_len,
                                                          maga_init_params.vit_config.output_transport,
                                                          metrics_reporter_,
                                                          maga_init_params.parallelism_config.local_rank));
    }

    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::serializeErrorMsg(const string& request_key, ErrorInfo error_info) {
    return serializeErrorMsg(request_key, RequestInfo(), error_info);
}

grpc::Status
LocalRpcServer::serializeErrorMsg(const string& request_key, const RequestInfo& request_info, ErrorInfo error_info) {
    const auto& error_msg       = error_info.ToString();
    const auto  request_log_tag = formatRequestLogTag(request_key, request_info);
    RTP_LLM_LOG_WARNING("%s, error code [%s], error message [%s]",
                        request_log_tag.c_str(),
                        ErrorCodeToString(error_info.code()).c_str(),
                        error_msg.c_str());
    auto           grpc_error_code = transErrorCodeToGrpc(error_info.code());
    ErrorDetailsPB error_details;
    error_details.set_error_code(static_cast<int>(error_info.code()));
    error_details.set_error_message(error_msg);
    std::string error_details_serialized;
    if (error_details.SerializeToString(&error_details_serialized)) {
        return grpc::Status(grpc_error_code, error_msg, error_details_serialized);
    } else {
        RTP_LLM_LOG_WARNING("%s error details serialize to string failed", request_log_tag.c_str());
        return grpc::Status(grpc_error_code, error_msg);
    }
}

grpc::Status LocalRpcServer::pollStreamOutput(grpc::ServerContext*             context,
                                              const string&                    request_key,
                                              WriterInterface*                 writer,
                                              std::shared_ptr<GenerateStream>& stream) {
    RTP_LLM_PROFILE_FUNCTION();
    while (true) {
        const auto result = stream->nextOutput(kRpcOutputWaitTimeoutMs);
        if (isCancelled(context)) {
            stream->reportError(ErrorCode::CANCELLED, "request cancelled by user");
            RTP_LLM_LOG_WARNING("request [%s] cancelled by user", request_key.c_str());
            return grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled by user");
        }
        if (!result.ok()) {
            if (result.status().code() == ErrorCode::OUTPUT_QUEUE_NO_UPDATE) {
                continue;
            }
            if (result.status().code() != ErrorCode::FINISHED) {
                return serializeErrorMsg(request_key, stream->generateInput()->request_info, result.status());
            } else {
                break;
            }
        }
        RTP_LLM_LOG_DEBUG("request [%s] generate next output success", request_key.c_str());
        GenerateOutputsPB outputs_pb;

        QueryConverter::transResponse(&outputs_pb,
                                      &(result.value()),
                                      stream->generateConfig()->aux_info,
                                      maga_init_params_.misc_config.aux_string,
                                      stream->specialTokens().eos_token_id);
        if (!writer->Write(outputs_pb)) {
            stream->reportError(ErrorCode::CANCELLED, "write outputs pb failed");
            RTP_LLM_LOG_WARNING("request [%s] write outputs pb failed", request_key.c_str());
            // WriterInterface uses false to signal that its downstream consumer has closed or cancelled.
            return grpc::Status(grpc::StatusCode::CANCELLED, "request output consumer closed");
        }
        if (stream->hasEvent(StreamEvents::NeedRemoteGenerate)) {
            break;
        }
    }
    RTP_LLM_LOG_DEBUG("request [%s] local generate done", request_key.c_str());

    return grpc::Status::OK;
}

ErrorInfo LocalRpcServer::updateMultimodalFeaturesWithTrace(
    std::shared_ptr<GenerateInput>&                                     input,
    const opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& parent_span,
    grpc::ServerContext*                                                server_context) {
    if (!telemetry::TelemetryRuntime::isActive() || parent_span == nullptr || !input->multimodal_inputs
        || input->multimodal_inputs->empty()) {
        return mm_processor_->updateMultimodalFeatures(input, server_context);
    }

    opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span> span;
    try {
        opentelemetry::trace::StartSpanOptions options;
        options.parent = parent_span->GetContext();
        options.kind   = opentelemetry::trace::SpanKind::kInternal;
        span           = telemetry::TelemetryRuntime::tracer()->StartSpan("rtp_llm.vit", options);
    } catch (...) {}
    grpc::Status                   status;
    telemetry::GrpcStatusSpanGuard guard(span, &status, telemetry::SpanStatusSemantics::Logical);
    try {
        guard.setAttribute(telemetry::kAttrRequestId, std::to_string(input->request_id));
    } catch (...) {}

    // Include preprocessing, embedding transport and token expansion in one operation.
    auto result = mm_processor_->updateMultimodalFeatures(input, server_context);
    if (!result.ok()) {
        status = grpc::Status(grpc::StatusCode::INTERNAL, "");
        try {
            const auto reason = ErrorCodeToString(result.code());
            guard.setLogicalErrorType(reason);
            guard.setAttribute(telemetry::kAttrRtpLlmErrorCode, static_cast<int64_t>(result.code()));
            guard.setAttribute(telemetry::kAttrRtpLlmErrorReason, reason);
        } catch (...) {}
    }
    return result;
}

ErrorInfo LocalRpcServer::prepareInput(const GenerateInputPB&                                              input_pb,
                                       std::shared_ptr<GenerateInput>&                                     output,
                                       const opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& parent_span,
                                       grpc::ServerContext* server_context) {
    output.reset();
    try {
        output = QueryConverter::transQuery(&input_pb);
        if (mm_processor_ != nullptr && output->multimodal_inputs && !output->multimodal_inputs->empty()) {
            RTP_LLM_PROFILE_SCOPE("rpc.mm_update_features");
            auto mm_res = updateMultimodalFeaturesWithTrace(output, parent_span, server_context);
            if (!mm_res.ok()) {
                return mm_res;
            }
        }
    } catch (const std::invalid_argument& e) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, e.what());
    } catch (const std::exception& e) {
        return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, e.what());
    } catch (...) {
        return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "input preparation failed with an unknown exception");
    }
    return ErrorInfo::OkStatus();
}

ErrorInfo LocalRpcServer::collectStreamOutput(grpc::ServerContext*                  context,
                                              std::shared_ptr<GenerateStream>&      stream,
                                              const std::shared_ptr<GenerateInput>& input,
                                              GenerateOutputs&                      last_outputs) {
    while (true) {
        const auto output_result = stream->nextOutput(kRpcOutputWaitTimeoutMs);
        if (isCancelled(context)) {
            stream->reportError(ErrorCode::CANCELLED, "request cancelled by client");
            return ErrorInfo(ErrorCode::CANCELLED, "request cancelled by client");
        }
        if (!output_result.ok()) {
            if (output_result.status().code() == ErrorCode::OUTPUT_QUEUE_NO_UPDATE) {
                continue;
            }
            if (output_result.status().code() != ErrorCode::FINISHED) {
                return output_result.status();
            }
            break;
        }
        last_outputs = output_result.value();
    }
    return ErrorInfo::OkStatus();
}

grpc::Status LocalRpcServer::GenerateStreamCall(grpc::ServerContext*                   context,
                                                const GenerateInputPB*                 request,
                                                grpc::ServerWriter<GenerateOutputsPB>* writer) try {
    RTP_LLM_PROFILE_SCOPE("rpc.generate_stream_call");
    c10::InferenceMode inference_guard(true);
    AtomicGuard        request_guard(onflight_requests_);
    if (request == nullptr) {
        return serializeErrorMsg("unknown", ErrorInfo(ErrorCode::INVALID_PARAMS, "generate request must not be null"));
    }
    auto request_id = request->request_id();
    RTP_LLM_LOG_DEBUG("receive request %ld", request_id);
    auto generate_context =
        GenerateContext(request_id, request->generate_config().timeout_ms(), context, metrics_reporter_, meta_);
    generate_context.onflight_requests    = &onflight_requests_;
    const int         uncaught_exceptions = std::uncaught_exceptions();
    autil::ScopeGuard rpc_completion_guard([&generate_context, uncaught_exceptions] {
        if (std::uncaught_exceptions() == uncaught_exceptions) {
            generate_context.markRpcHandlingCompleted();
        }
    });
    // gRPC SERVER span doubles as the request span on the fusion path; guard
    // destruction covers CHECK_ERROR_STATUS early returns.
    if (telemetry::TelemetryRuntime::isActive()) {
        auto span = telemetry::startRpcServerSpan(
            "rtp_llm.generate_stream_call", context, false, "RpcService/GenerateStreamCall");
        generate_context.trace_span_guard =
            std::make_unique<telemetry::GrpcStatusSpanGuard>(span, &generate_context.error_status);
        // `request_id` is the Bailian Unitrace index key (string, verified);
        generate_context.trace_span_guard->setAttribute(telemetry::kAttrRequestId, std::to_string(request_id));
    }
    telemetry::PhaseSpanSynthesisScope phase_span_scope([&generate_context](bool exception_unwinding) {
        if (!generate_context.trace_span_guard || !generate_context.trace_span_guard->valid()) {
            return;
        }
        auto& stream = generate_context.getStream();
        if (!stream) {
            return;
        }
        const auto             time_info = stream->getTimeInfo();
        telemetry::PhaseTiming phase_timing;
        phase_timing.begin_time_us           = time_info.begin_time_us;
        phase_timing.running_started         = time_info.running_started;
        phase_timing.running_started_time_us = time_info.running_started_time_us;
        phase_timing.first_token_committed   = time_info.first_token_committed;
        phase_timing.first_token_time_us     = time_info.first_token_time_us;
        phase_timing.generation_done         = time_info.generation_done;
        phase_timing.generation_done_time_us = time_info.generation_done_time_us;
        phase_timing.synthesis_end_time_us   = currentTimeUs();
        phase_timing.request_id              = generate_context.request_id;
        const bool request_ok                = generate_context.error_status.ok() && !exception_unwinding;
        if (request_ok && time_info.generation_done) {
            // Beam rows are an internal search width; Fusion exposes one primary
            // sequence (the remaining candidates live in beam_responses). Only
            // ordinary multi-return requests aggregate all active rows.
            const auto returned_sequence_count =
                stream->hasNumBeams() ? std::max(stream->numReturnSequences(), 1) : stream->currentBatchSize();
            telemetry::setUsageTokenAttributes(*generate_context.trace_span_guard,
                                               (int64_t)stream->inputLength(),
                                               (int64_t)(stream->outputTokenLen() * returned_sequence_count));
        }
        phase_timing.error_type = request_ok ?
                                      nullptr :
                                      (!generate_context.error_status.ok() ?
                                           telemetry::grpcStatusCodeName(generate_context.error_status.error_code()) :
                                           "Exception");
        telemetry::synthesizePhaseSpans(
            generate_context.trace_span_guard->sharedSpan(), phase_timing, telemetry::PhaseRole::Fusion, request_ok);
    });
    std::shared_ptr<GenerateInput>     input;
    {
        auto mm_res = prepareInput(*request,
                                   input,
                                   generate_context.trace_span_guard ?
                                       generate_context.trace_span_guard->sharedSpan() :
                                       opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>{},
                                   context);
        if (!mm_res.ok()) {
            generate_context.error_info = mm_res;
            generate_context.error_status =
                serializeErrorMsg(generate_context.request_key, generate_context.request_info, mm_res);
        }
    }
    if (input) {
        generate_context.request_info = input->request_info;
    }
    CHECK_ERROR_STATUS(generate_context);
    if (writer == nullptr) {
        generate_context.error_info = ErrorInfo(ErrorCode::INVALID_PARAMS, "generate response writer must not be null");
        generate_context.error_status =
            serializeErrorMsg(generate_context.request_key, generate_context.request_info, generate_context.error_info);
        return generate_context.error_status;
    }
    if (!engine_) {
        throw std::runtime_error("LocalRpcServer engine is not initialized");
    }

    RTP_LLM_LOG_DEBUG("request [%ld] trans to stream success", request_id);
    {
        RTP_LLM_PROFILE_SCOPE("rpc.enqueue_engine");
        generate_context.setStream(engine_->enqueue(input));
    }

    RTP_LLM_LOG_DEBUG("request [%ld] enqueue success", request_id);

    generate_context.error_status =
        pollStreamOutput(context, generate_context.request_key, writer, generate_context.getStream());
    if (generate_context.hasError() && generate_context.getStream() && generate_context.getStream()->hasError()) {
        generate_context.error_info = generate_context.getStream()->statusInfo();
    }
    meta_->dequeue(generate_context.request_id, generate_context.getStream());
    return generate_context.error_status;
} catch (const std::exception& e) {
    const auto request_key = request ? std::to_string(request->request_id()) : "unknown";
    return serializeErrorMsg(request_key,
                             ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                                       "GenerateStreamCall failed with exception: " + std::string(e.what())));
} catch (...) {
    const auto request_key = request ? std::to_string(request->request_id()) : "unknown";
    return serializeErrorMsg(
        request_key, ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "GenerateStreamCall failed with unknown exception"));
}

grpc::Status LocalRpcServer::BatchGenerateCall(grpc::ServerContext*        context,
                                               const BatchGenerateInputPB* request,
                                               BatchGenerateOutputsPB*     response) {
    std::vector<GenerateStreamPtr> streams;
    BatchStreamCleanupGuard        stream_cleanup_guard(streams);

    try {
        RTP_LLM_PROFILE_SCOPE("rpc.batch_generate_call");
        c10::InferenceMode inference_guard(true);
        AtomicGuard        request_guard(onflight_requests_);
        if (request == nullptr || response == nullptr) {
            return serializeErrorMsg(
                "unknown",
                ErrorInfo(ErrorCode::INVALID_PARAMS, "batch generate request and response must not be null"));
        }
        const int batch_size = request->inputs_size();
        RTP_LLM_LOG_INFO("receive batch generate request, batch_size=%d", batch_size);
        if (batch_size == 0) {
            return grpc::Status::OK;
        }
        response->clear_results();

        std::vector<std::shared_ptr<GenerateInput>> inputs;
        inputs.reserve(batch_size);
        for (int i = 0; i < batch_size; i++) {
            std::shared_ptr<GenerateInput> input;
            auto                           err = prepareInput(request->inputs(i), input, {}, context);
            if (!err.ok()) {
                // Fill error results for all requests (0..batch_size-1) to maintain 1:1 mapping.
                const bool invalid_params = err.code() == ErrorCode::INVALID_PARAMS;
                for (int j = 0; j < batch_size; j++) {
                    auto* result = response->add_results();
                    auto* err_pb = result->mutable_error_info();
                    // Internal ErrorCode values and RPC enum values are independent wire contracts.
                    err_pb->set_error_code(transErrorCodeToRPC(err.code()));
                    if (j == i) {
                        err_pb->set_error_message(invalid_params ? err.ToString() :
                                                                   "input preparation failed: " + err.ToString());
                    } else if (invalid_params) {
                        err_pb->set_error_message("batch aborted due to invalid parameters at index "
                                                  + std::to_string(i));
                    } else {
                        err_pb->set_error_message("batch aborted due to input preparation failure at index "
                                                  + std::to_string(i));
                    }
                }
                return grpc::Status::OK;
            }
            inputs.push_back(input);
        }

        if (!engine_) {
            for (int i = 0; i < batch_size; ++i) {
                auto* error_info = response->add_results()->mutable_error_info();
                error_info->set_error_code(transErrorCodeToRPC(ErrorCode::EXECUTION_EXCEPTION));
                error_info->set_error_message("LocalRpcServer engine is not initialized");
            }
            return grpc::Status::OK;
        }

        const auto  enqueue_result    = engine_->enqueueMultiple(inputs);
        const auto& enqueue_successes = enqueue_result.first;
        streams                       = enqueue_result.second;
        if (enqueue_successes.size() != inputs.size() || streams.size() != inputs.size()) {
            throw std::runtime_error("enqueueMultiple result size mismatch: input=" + std::to_string(inputs.size())
                                     + " status=" + std::to_string(enqueue_successes.size())
                                     + " stream=" + std::to_string(streams.size()));
        }

        // collectStreamOutput is currently SERIAL: streams[0] must finish before streams[1] is drained.
        // For batch decode this is bounded (all streams advance together), but TODO: parallelize for
        // mixed-length batches.
        for (int i = 0; i < (int)streams.size(); i++) {
            if (!streams[i]) {
                throw std::runtime_error("enqueueMultiple returned a null stream at index " + std::to_string(i));
            }
            auto* result = response->add_results();
            if (!enqueue_successes[i]) {
                auto* error_info = result->mutable_error_info();
                if (streams[i]->hasError()) {
                    const auto stream_error = streams[i]->statusInfo();
                    error_info->set_error_code(transErrorCodeToRPC(stream_error.code()));
                    error_info->set_error_message(stream_error.ToString());
                } else {
                    error_info->set_error_code(transErrorCodeToRPC(ErrorCode::EXECUTION_EXCEPTION));
                    error_info->set_error_message("enqueueMultiple rejected input at index " + std::to_string(i));
                }
                stream_cleanup_guard.markHandled(static_cast<size_t>(i));
                continue;
            }

            GenerateOutputs last_outputs;
            auto            err = collectStreamOutput(context, streams[i], inputs[i], last_outputs);
            stream_cleanup_guard.markHandled(static_cast<size_t>(i));
            if (!err.ok()) {
                auto* err_pb = result->mutable_error_info();
                err_pb->set_error_code(transErrorCodeToRPC(err.code()));
                err_pb->set_error_message(err.ToString());
            } else {
                auto* output_pb = result->mutable_final_output();
                QueryConverter::transResponse(output_pb,
                                              &last_outputs,
                                              inputs[i]->generate_config->aux_info,
                                              maga_init_params_.misc_config.aux_string,
                                              streams[i]->specialTokens().eos_token_id);
            }
        }

        stream_cleanup_guard.dismiss();
        RTP_LLM_LOG_INFO("batch generate done, batch_size=%d", batch_size);
        return grpc::Status::OK;
    } catch (const std::exception& e) {
        const std::string error_message = "BatchGenerateCall failed with exception: " + std::string(e.what());
        const int         batch_size    = request ? request->inputs_size() : 0;
        RTP_LLM_LOG_ERROR("%s, batch_size=%d", error_message.c_str(), batch_size);
        if (response == nullptr) {
            return serializeErrorMsg("unknown", ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error_message));
        }
        response->clear_results();
        for (int i = 0; i < batch_size; ++i) {
            auto* error_info = response->add_results()->mutable_error_info();
            error_info->set_error_code(transErrorCodeToRPC(ErrorCode::EXECUTION_EXCEPTION));
            error_info->set_error_message(error_message);
        }
        return grpc::Status::OK;
    } catch (...) {
        const std::string error_message = "BatchGenerateCall failed with unknown exception";
        const int         batch_size    = request ? request->inputs_size() : 0;
        RTP_LLM_LOG_ERROR("%s, batch_size=%d", error_message.c_str(), batch_size);
        if (response == nullptr) {
            return serializeErrorMsg("unknown", ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error_message));
        }
        response->clear_results();
        for (int i = 0; i < batch_size; ++i) {
            auto* error_info = response->add_results()->mutable_error_info();
            error_info->set_error_code(transErrorCodeToRPC(ErrorCode::EXECUTION_EXCEPTION));
            error_info->set_error_message(error_message);
        }
        return grpc::Status::OK;
    }
}

grpc::Status
LocalRpcServer::GetCacheStatus(grpc::ServerContext* context, const CacheVersionPB* request, CacheStatusPB* response) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("receive cacheStatus rpc request from client: %s, request cache version: [%d]",
                      context->peer().c_str(),
                      request->latest_cache_version());
    KVCacheInfo cache_status = getCacheStatusInfo(request->latest_cache_version(), request->need_cache_keys());
    response->set_available_kv_cache(cache_status.available_kv_cache);
    response->set_total_kv_cache(cache_status.total_kv_cache);
    response->set_block_size(cache_status.block_size);
    response->set_version(cache_status.version);
    auto* cache_map = response->mutable_cache_keys();
    for (const auto& key : cache_status.cached_keys) {
        (*cache_map)[static_cast<int64_t>(key)] = true;
    }
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::GetWorkerStatus(grpc::ServerContext*   context,
                                             const StatusVersionPB* request,
                                             WorkerStatusPB*        response) {
    RTP_LLM_PROFILE_FUNCTION();
    int64_t request_begin_time_us   = currentTimeUs();
    int64_t latest_finished_version = request->latest_finished_version();
    RTP_LLM_LOG_DEBUG(
        "receive workerStatus rpc request from client: %s, latest_finished_version: %ld, config role_type: %d",
        context->peer().c_str(),
        latest_finished_version,
        maga_init_params_.pd_sep_config.role_type);

    WorkerStatusInfo status_info              = getWorkerStatusInfo(latest_finished_version);
    int64_t          request_after_ws_time_us = currentTimeUs();
    RTP_LLM_LOG_DEBUG("getWorkerStatusInfo took %ld us", request_after_ws_time_us - request_begin_time_us);

    const auto& engine_schedule_info = status_info.engine_schedule_info;
    // Field 1 is consumed as an opaque string by legacy Masters, whose role
    // codes are "RoleType.PREFILL" etc. Keep that exact spelling while the
    // typed field 20 is rolled out.
    response->set_role("RoleType." + roleTypeToString(status_info.role));
    response->set_role_type(static_cast<RoleTypePB>(status_info.role));

    for (const auto& task : engine_schedule_info.running_task_info_list) {
        TaskInfoPB* task_info = response->add_running_task_info();
        task_info->set_request_id(task.request_id);
        task_info->set_prefix_length(task.prefix_length);
        task_info->set_input_length(task.input_length);
        task_info->set_waiting_time_ms(task.waiting_time_ms);
        task_info->set_iterate_count(task.iterate_count);
        task_info->set_end_time_ms(task.end_time_ms);
        task_info->set_dp_rank(status_info.dp_rank);
        task_info->set_phase(static_cast<::TaskPhase>(task.phase));
        task_info->set_is_waiting(task.phase != rtp_llm::TaskPhase::RUNNING);
        task_info->set_batch_id(task.batch_id);
        task_info->set_execution_time_ms(task.execution_time_ms);
        task_info->set_priority_preemption_progress(
            static_cast<PriorityPreemptionProgressPB>(task.priority_preemption_progress));
        if (task.error_code != 0) {
            task_info->mutable_error_info()->set_error_code(task.error_code);
            task_info->mutable_error_info()->set_error_message(task.error_message);
        }
    }

    for (const auto& task : engine_schedule_info.finished_task_info_list) {
        TaskInfoPB* task_info = response->add_finished_task_list();
        task_info->set_request_id(task.request_id);
        task_info->set_prefix_length(task.prefix_length);
        task_info->set_input_length(task.input_length);
        task_info->set_waiting_time_ms(task.waiting_time_ms);
        task_info->set_iterate_count(task.iterate_count);
        task_info->set_end_time_ms(task.end_time_ms);
        task_info->set_dp_rank(status_info.dp_rank);
        task_info->set_phase(static_cast<::TaskPhase>(task.phase));
        task_info->set_is_waiting(task.phase != rtp_llm::TaskPhase::RUNNING);
        task_info->set_batch_id(task.batch_id);
        task_info->set_execution_time_ms(task.execution_time_ms);
        task_info->set_priority_preemption_progress(
            static_cast<PriorityPreemptionProgressPB>(task.priority_preemption_progress));
        if (task.error_code != 0) {
            task_info->mutable_error_info()->set_error_code(task.error_code);
            task_info->mutable_error_info()->set_error_message(task.error_message);
        }
    }

    response->set_dp_size(status_info.dp_size);
    response->set_tp_size(status_info.tp_size);
    response->set_status_version(status_info.status_version);
    response->set_latest_finished_version(status_info.latest_finished_version);
    response->set_alive(status_info.alive);
    response->set_precision(status_info.precision);
    response->set_dp_rank(status_info.dp_rank);
    response->set_max_seq_len(maga_init_params_.model_config_.max_seq_len);
    response->set_max_batch_tokens_size(maga_init_params_.runtime_config.fifo_scheduler_config.max_batch_tokens_size);
    auto kv_info = engine_->getCacheStatusInfo(-1, false);
    response->set_available_kv_cache(kv_info.available_kv_cache);
    response->set_total_kv_cache(kv_info.total_kv_cache);
    reportWorkerStatusTime(request_begin_time_us, request_after_ws_time_us);
    return grpc::Status::OK;
}

WorkerStatusInfo LocalRpcServer::getWorkerStatusInfo(int64_t latest_finished_version) {
    WorkerStatusInfo status_info;
    status_info.engine_schedule_info    = getEngineScheduleInfo(latest_finished_version);
    status_info.role                    = maga_init_params_.pd_sep_config.role_type;
    status_info.dp_size                 = maga_init_params_.parallelism_config.dp_size;
    status_info.tp_size                 = maga_init_params_.parallelism_config.tp_size;
    status_info.dp_rank                 = maga_init_params_.parallelism_config.dp_rank;
    status_info.status_version          = currentTimeUs();
    status_info.latest_finished_version = status_info.engine_schedule_info.latest_finished_version;
    status_info.alive                   = true;
    auto quant_method                   = maga_init_params_.model_config_.quant_algo.getQuantMethod();

    switch (quant_method) {
        case QuantMethod::WeightOnlyPerCol:
            status_info.precision = "WeightOnlyPerCol";
            break;
        case QuantMethod::GptQ:
            status_info.precision = "GptQ";
            break;
        case QuantMethod::Awq:
            status_info.precision = "Awq";
            break;
        case QuantMethod::SmoothQuant:
            status_info.precision = "SmoothQuant";
            break;
        case QuantMethod::OmniQuant:
            status_info.precision = "OmniQuant";
            break;
        case QuantMethod::PerTensorQuant:
            status_info.precision = "PerTensorQuant";
            break;
        case QuantMethod::FP8Quant:
            status_info.precision = "FP8Quant";
            break;
        case QuantMethod::FP8PTPC:
            status_info.precision = "FP8PTPC";
            break;
        case QuantMethod::W4A8INT4PTPC:
            status_info.precision = "W4A8INT4PTPC";
            break;
        case QuantMethod::W8A8INT8PTPC:
            status_info.precision = "W8A8INT8PTPC";
            break;
        case QuantMethod::None:
            status_info.precision = "FP16";
            break;
        default:
            RTP_LLM_LOG_ERROR("unknown quant method: %d", static_cast<int>(quant_method));
            status_info.precision = "UNKNOWN";
    }
    return status_info;
}

KVCacheInfo LocalRpcServer::getCacheStatusInfo(int64_t latest_version, bool need_cache_keys) {
    int64_t     request_begin_time_us = currentTimeUs();
    const auto& cache_info            = engine_->getCacheStatusInfo(latest_version, need_cache_keys);
    reportCacheStatusTime(request_begin_time_us);
    return cache_info;
}

size_t LocalRpcServer::onflightRequestNum() {
    return onflight_requests_;
}

EngineScheduleInfo LocalRpcServer::getEngineScheduleInfo(int64_t latest_finished_version) {
    EngineScheduleInfo info               = meta_->getEngineScheduleInfo(latest_finished_version);
    auto               last_schedule_time = engine_->getLastScheduleTime();
    info.last_schedule_delta =
        std::max((int64_t)0, autil::TimeUtility::currentTimeInMilliSeconds() - last_schedule_time);
    return info;
}

grpc::Status LocalRpcServer::UpdateSchedulerInfo(grpc::ServerContext*                context,
                                                 const UpdateSchedulerInfoRequestPB* request,
                                                 EmptyPB*                            response) {
    const std::string scheduler_info = request->scheduler_info();
    engine_->getScheduler().updateSchedulerInfo(scheduler_info);
    return grpc::Status::OK;
}

grpc::Status
LocalRpcServer::SetLogLevel(grpc::ServerContext* context, const SetLogLevelRequestPB* request, EmptyPB* response) {
    std::string log_level_str = request->log_level();
    uint32_t    log_level     = alog::LOG_LEVEL_INFO;
    if (log_level_str == "INFO" || log_level_str == "info") {
        log_level = alog::LOG_LEVEL_INFO;
    } else if (log_level_str == "WARNING" || log_level_str == "warning") {
        log_level = alog::LOG_LEVEL_WARN;
    } else if (log_level_str == "DEBUG" || log_level_str == "debug") {
        log_level = alog::LOG_LEVEL_DEBUG;
    } else if (log_level_str == "TRACE" || log_level_str == "trace") {
        log_level = alog::LOG_LEVEL_TRACE1;
    } else {
        RTP_LLM_LOG_WARNING("set log level failed, unknown log level: %s", log_level_str.c_str());
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Invalid log_level format");
    }
    auto& logger = rtp_llm::Logger::getEngineLogger();
    logger.setBaseLevel(log_level);
    return grpc::Status::OK;
}

namespace {
int64_t pressureNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
void profileStatusToProto(const ProfileStatus& value, ProfileStatusPB* pb) {
    pb->set_supported(value.supported);
    pb->set_state(value.state);
    pb->set_trace_name(value.trace_name);
    pb->set_stop_requested(value.stop_requested);
    pb->set_collection_complete(value.collection_complete);
    pb->set_export_complete(value.export_complete);
    pb->set_collection_error(value.collection_error);
    pb->set_export_error(value.export_error);
    pb->set_file_name(value.file_name);
    pb->set_capture_begin_ns(value.capture_begin_ns);
    pb->set_capture_end_ns(value.capture_end_ns);
}
PressureTierPB pressureTier(Tier tier) {
    switch (tier) {
        case Tier::DEVICE:
            return PRESSURE_DEVICE;
        case Tier::HOST:
            return PRESSURE_HOST;
        case Tier::DISK:
            return PRESSURE_DISK;
        case Tier::REMOTE:
            return PRESSURE_REMOTE;
        case Tier::NONE:
            return PRESSURE_NONE;
    }
    return PRESSURE_TIER_UNSPECIFIED;
}
PressureGroupTypePB pressureGroup(CacheGroupType type) {
    switch (type) {
        case CacheGroupType::FULL:
            return PRESSURE_FULL;
        case CacheGroupType::SWA:
            return PRESSURE_SWA;
        case CacheGroupType::LINEAR:
            return PRESSURE_LINEAR;
    }
    return PRESSURE_GROUP_UNSPECIFIED;
}
PressureOperationPB pressureOperation(CacheTransferOperation operation) {
    switch (operation) {
        case CacheTransferOperation::LOAD:
            return PRESSURE_LOAD;
        case CacheTransferOperation::EVICT:
            return PRESSURE_EVICT;
        case CacheTransferOperation::STORE:
            return PRESSURE_STORE;
    }
    return PRESSURE_OPERATION_UNSPECIFIED;
}
void pressureWatermark(const TierWatermark& value, PressureWatermarkPB* output) {
    output->set_low_ratio(value.low_ratio);
    output->set_high_ratio(value.high_ratio);
}
void pressureSnapshotToProto(const CachePressureSnapshot& snapshot, CachePressureRankPB* output) {
    for (const auto& pool : snapshot.pools) {
        auto* pb = output->add_pools();
        pb->set_pool_id(pool.pool_id);
        pb->set_tier(pressureTier(pool.tier));
        pb->set_pool_name(pool.pool_name);
        pb->set_block_size_bytes(pool.block_size_bytes);
        pb->set_total_blocks(pool.total_blocks);
        pb->set_free_blocks(pool.free_blocks);
        pb->set_used_blocks(pool.used_blocks);
        pb->set_available_blocks(pool.available_blocks);
        pb->set_active_blocks(pool.active_blocks);
        pb->set_block_cache_ref_blocks(pool.block_cache_ref_blocks);
        pb->set_load_ref_blocks(pool.load_ref_blocks);
        pb->set_eviction_target_ref_blocks(pool.eviction_target_ref_blocks);
        pb->set_store_ref_blocks(pool.store_ref_blocks);
        pb->set_pending_release_blocks(pool.pending_release_blocks);
        pb->set_published_host_blocks(pool.published_host_blocks);
        if (pool.request_ref_blocks_available) {
            pb->mutable_request_ref_blocks()->set_value(pool.request_ref_blocks);
        }
    }
    for (const auto& group : snapshot.groups) {
        auto* pb = output->add_groups();
        pb->set_group_id(group.group_id);
        pb->set_group_type(pressureGroup(group.group_type));
        for (const auto& id : group.device_pool_ids) {
            pb->add_device_pool_ids(id);
        }
        if (group.host_pool_id) {
            pb->mutable_host_pool_id()->set_value(*group.host_pool_id);
        }
        if (group.disk_pool_id) {
            pb->mutable_disk_pool_id()->set_value(*group.disk_pool_id);
        }
        pressureWatermark(group.device_watermark, pb->mutable_device_watermark());
        pressureWatermark(group.host_watermark, pb->mutable_host_watermark());
        pressureWatermark(group.disk_watermark, pb->mutable_disk_watermark());
        pb->set_watermark_trigger_count(group.watermark_trigger_count);
    }
    for (const auto& transfer : snapshot.transfers) {
        auto* pb = output->add_transfers();
        pb->set_group_id(transfer.group_id);
        pb->set_pool_id(transfer.pool_id);
        pb->set_pool_tier(pressureTier(transfer.pool_tier));
        pb->set_operation(pressureOperation(transfer.operation));
        pb->set_source_tier(pressureTier(transfer.source_tier));
        pb->set_target_tier(pressureTier(transfer.target_tier));
        pb->set_copied_blocks(transfer.copied_blocks);
        pb->set_copied_bytes(transfer.copied_bytes);
        pb->set_published_blocks(transfer.published_blocks);
        pb->set_published_bytes(transfer.published_bytes);
        pb->set_failed_operations(transfer.failed_operations);
    }
}
}  // namespace

std::string LocalRpcServer::makePressureInstance() {
    return std::to_string(getpid()) + "-" + std::to_string(pressureNowNs()) + "-"
           + std::to_string(std::random_device{}());
}

namespace {
grpc::Status validatePressureControl(const PressureSessionControlRequestPB& request) {
    if (request.session_id().empty() || request.session_id().size() > 256 || request.phase().size() > 256
        || request.action() < PressureSessionControlRequestPB::ENABLE
        || request.action() > PressureSessionControlRequestPB::DISABLE
        || request.recorder_capacity() > PressureStepRecorder::kMaxCapacity
        || (request.action() != PressureSessionControlRequestPB::ENABLE && request.recorder_capacity())
        || (request.action() == PressureSessionControlRequestPB::MARK_PHASE && request.phase().empty())) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "invalid pressure session control"};
    }
    return grpc::Status::OK;
}
}  // namespace

grpc::Status LocalRpcServer::ControlPressureDiagnosticsInternal(grpc::ServerContext*                   context,
                                                                const PressureSessionControlRequestPB* request,
                                                                PressureSessionControlResponsePB*      response) {
    response->Clear();
    response->set_schema_version(1);
    auto status = validatePressureControl(*request);
    if (!status.ok()) {
        return status;
    }
    auto* rank = response->add_ranks();
    rank->set_world_rank(maga_init_params_.parallelism_config.world_rank);
    rank->set_service_instance(pressure_instance_);
    if (context->IsCancelled()) {
        return {grpc::StatusCode::CANCELLED, "pressure control cancelled"};
    }
    PressureSessionControl control{request->session_id(),
                                   static_cast<PressureControlAction>(request->action()),
                                   request->phase(),
                                   request->recorder_capacity()};
    const auto             deadline = std::min(context->deadline(),
                                   std::chrono::system_clock::now() + std::chrono::milliseconds(kPressureRpcTimeoutMs));
    auto                   result   = engine_ ? engine_->controlPressureDiagnostics(control, deadline) :
                                                ErrorResult<int64_t>(kPressureDiagnosticsUnsupported, "pressure diagnostics unsupported");
    if (!result.ok()) {
        rank->set_error(result.status().ToString());
        auto code = grpc::StatusCode::FAILED_PRECONDITION;
        if (result.status().code() == kPressureDiagnosticsUnsupported) {
            code = grpc::StatusCode::UNIMPLEMENTED;
        } else if (result.status().code() == ErrorCode::DEADLINE_EXCEEDED) {
            code = grpc::StatusCode::DEADLINE_EXCEEDED;
        }
        return {code, result.status().ToString()};
    }
    rank->set_applied_monotonic_ns(result.value());
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::ControlPressureDiagnostics(grpc::ServerContext*                   context,
                                                        const PressureSessionControlRequestPB* request,
                                                        PressureSessionControlResponsePB*      response) {
    response->Clear();
    response->set_schema_version(1);
    auto status = validatePressureControl(*request);
    if (!status.ok()) {
        return status;
    }
    const auto& topology = maga_init_params_.parallelism_config;
    if (topology.tp_rank != 0) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "pressure control fan-out requires tp_rank 0"};
    }
    const int        base = topology.world_rank - topology.tp_rank;
    std::set<int>    seen;
    std::vector<int> ranks(request->target_ranks().begin(), request->target_ranks().end());
    if (ranks.empty()) {
        for (int i = 0; i < topology.tp_size; ++i) {
            ranks.push_back(base + i);
        }
    }
    for (int rank : ranks) {
        if (rank < base || rank >= base + topology.tp_size || !seen.insert(rank).second) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "duplicate or out-of-topology target rank"};
        }
    }
    struct Pending {
        grpc::ClientContext                                                                context;
        PressureSessionControlResponsePB                                                   response;
        grpc::Status                                                                       status;
        std::shared_ptr<RpcService::Stub>                                                  stub;
        std::unique_ptr<grpc::ClientAsyncResponseReader<PressureSessionControlResponsePB>> reader;
        PressureSessionControlRankPB*                                                      record;
        int                                                                                rank;
    };
    grpc::CompletionQueue                 queue;
    std::vector<std::unique_ptr<Pending>> pending;
    const auto                            deadline     = std::min(context->deadline(),
                                   std::chrono::system_clock::now() + std::chrono::milliseconds(kPressureRpcTimeoutMs));
    const auto&                           addresses    = maga_init_params_.runtime_config.worker_grpc_addrs;
    PressureSessionControlRankPB*         local_record = nullptr;
    // Launch peers before waiting for the local engine boundary. Every rank shares the deadline.
    for (int rank : ranks) {
        auto* record = response->add_ranks();
        record->set_world_rank(rank);
        if (rank == topology.world_rank) {
            local_record = record;
            continue;
        }
        const int index = rank - base;
        if (index >= addresses.size() || addresses[index].empty()) {
            record->set_error("rank endpoint unavailable");
            continue;
        }
        auto connection = pressure_rpc_pool_.getConnection(addresses[index]);
        if (!connection.ok()) {
            record->set_error(connection.status().ToString());
            continue;
        }
        auto call    = std::make_unique<Pending>();
        call->record = record;
        call->rank   = rank;
        call->stub   = connection.value().stub;
        call->context.set_deadline(deadline);
        call->reader = call->stub->AsyncControlPressureDiagnosticsInternal(&call->context, *request, &queue);
        call->reader->Finish(&call->response, &call->status, call.get());
        pending.push_back(std::move(call));
    }
    grpc::Status local_status;
    if (local_record) {
        if (context->IsCancelled()) {
            local_status = {grpc::StatusCode::CANCELLED, "pressure control cancelled"};
        } else {
            PressureSessionControl control{request->session_id(),
                                           static_cast<PressureControlAction>(request->action()),
                                           request->phase(),
                                           request->recorder_capacity()};
            auto                   result = engine_ ?
                                                engine_->controlPressureDiagnostics(control, deadline) :
                                                ErrorResult<int64_t>(kPressureDiagnosticsUnsupported, "pressure diagnostics unsupported");
            local_record->set_service_instance(pressure_instance_);
            if (result.ok()) {
                local_record->set_applied_monotonic_ns(result.value());
            } else {
                local_status = {
                    result.status().code() == kPressureDiagnosticsUnsupported ? grpc::StatusCode::UNIMPLEMENTED :
                    result.status().code() == ErrorCode::DEADLINE_EXCEEDED    ? grpc::StatusCode::DEADLINE_EXCEEDED :
                                                                                grpc::StatusCode::FAILED_PRECONDITION,
                    result.status().ToString()};
            }
        }
        if (!local_status.ok()) {
            local_record->set_error(local_status.error_message());
        }
    }
    bool cancellation_sent = false;
    for (size_t i = 0; i < pending.size();) {
        void*      tag = nullptr;
        bool       ok  = false;
        const auto now = std::chrono::system_clock::now();
        if (now >= deadline || context->IsCancelled()) {
            if (!cancellation_sent) {
                for (auto& call : pending) {
                    call->context.TryCancel();
                }
                cancellation_sent = true;
            }
            queue.Next(&tag, &ok);
        } else {
            // A peer may enqueue after the root's first empty TP broadcast, or
            // root may not be a target at all. Keep the existing idle TP loop
            // progressing only while this bounded control fan-out is pending.
            if (engine_) {
                engine_->wakePressureControlProgress();
            }
            if (queue.AsyncNext(&tag, &ok, std::min(deadline, now + std::chrono::milliseconds(10)))
                != grpc::CompletionQueue::GOT_EVENT) {
                continue;
            }
        }
        ++i;
        auto* call = static_cast<Pending*>(tag);
        if (ok && call->status.ok() && call->response.schema_version() == 1 && call->response.ranks_size() == 1
            && call->response.ranks(0).world_rank() == call->rank) {
            *call->record = call->response.ranks(0);
        } else {
            call->record->set_error(call->status.ok() ? "invalid rank response" : call->status.error_message());
        }
    }
    queue.Shutdown();
    void* tag = nullptr;
    bool  ok  = false;
    while (queue.Next(&tag, &ok)) {}
    return ranks.size() == 1 && local_record ? local_status : grpc::Status::OK;
}

grpc::Status LocalRpcServer::GetCachePressureInternal(grpc::ServerContext*          context,
                                                      const CachePressureRequestPB* request,
                                                      CachePressureResponsePB*      response) {
    response->Clear();
    response->set_schema_version(1);
    response->set_coordinator_received_ns(pressureNowNs());
    if (request->max_step_records() > kMaxPressureStepRecords) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "max_step_records exceeds 1024"};
    }
    const int rank = maga_init_params_.parallelism_config.world_rank;
    response->add_requested_ranks(rank);
    auto* record = response->add_ranks();
    record->set_world_rank(rank);
    record->set_service_instance(pressure_instance_);
    record->set_sequence(pressure_sequence_.fetch_add(1) + 1);
    const auto begin = pressureNowNs();
    record->set_sampled_monotonic_ns(begin);
    if (context->IsCancelled()) {
        return {grpc::StatusCode::CANCELLED, "diagnostic request cancelled"};
    }
    auto result = engine_ ? engine_->getCachePressure() : unsupportedCachePressure();
    profileStatusToProto(engine_ ? engine_->getProfileStatus() : ProfileStatus{}, record->mutable_profile());
    record->set_collect_duration_ns(pressureNowNs() - begin);
    response->set_coordinator_sent_ns(pressureNowNs());
    if (!result.ok()) {
        record->set_error(result.status().ToString());
        return {result.status().code() == kPressureDiagnosticsUnsupported ? grpc::StatusCode::UNIMPLEMENTED :
                                                                            grpc::StatusCode::INTERNAL,
                result.status().ToString()};
    }
    pressureSnapshotToProto(result.value(), record);
    const auto cursor = request->after_step_sequence().find(rank);
    auto       page =
        engine_->getPressureSteps(request->session_id(),
                                  cursor == request->after_step_sequence().end() ? 0 : cursor->second,
                                  request->max_step_records() ? request->max_step_records() : kMaxPressureStepRecords);
    record->set_step_history_available(page.available);
    record->set_next_step_sequence(page.next_sequence);
    record->set_step_records_lost(page.lost);
    auto add_step = [&](const PressureStepRecord& step) {
        auto* pb = record->add_steps();
        pb->set_session_id(step.session_id);
        pb->set_phase(step.phase);
        pb->set_step_id(step.step_id);
        pb->set_sequence(step.sequence);
        pb->set_wall_start_ns(step.forward_enter_ns);
        pb->set_wall_end_ns(step.forward_return_ns);
        pb->set_cpu_start_ns(step.forward_enter_cpu_ns);
        pb->set_cpu_end_ns(step.forward_return_cpu_ns);
        pb->set_context_count(step.context_batch);
        pb->set_decode_count(step.decode_batch);
        pb->set_executed_tokens(step.tokens);
        // NormalExecutor executes one token per decode batch row.
        pb->set_context_tokens(step.tokens >= step.decode_batch ? step.tokens - step.decode_batch : 0);
        pb->set_decode_tokens(step.decode_batch);
        for (const auto& id : step.request_ids) {
            pb->add_source_request_ids(id);
        }
        for (const auto id : step.stream_ids) {
            pb->add_request_ids(id);
        }
        pb->set_completed(step.complete);
        pb->set_preparation_begin_ns(step.begin_ns);
        pb->set_forward_enter_ns(step.forward_enter_ns);
        pb->set_forward_return_ns(step.forward_return_ns);
        pb->set_end_ns(step.end_ns);
    };
    for (const auto& step : page.records) {
        add_step(step);
    }
    if (page.available) {
        record->mutable_activity()->mutable_running_requests()->set_value(
            page.active ? page.active->context_batch + page.active->decode_batch : 0);
    }
    if (page.active) {
        add_step(*page.active);
    }
    record->set_collect_duration_ns(pressureNowNs() - begin);
    response->set_coordinator_sent_ns(pressureNowNs());
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::GetCachePressure(grpc::ServerContext*          context,
                                              const CachePressureRequestPB* request,
                                              CachePressureResponsePB*      response) {
    response->Clear();
    response->set_schema_version(1);
    response->set_coordinator_received_ns(pressureNowNs());
    const auto& topology = maga_init_params_.parallelism_config;
    if (topology.tp_rank != 0) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "cache pressure fan-out requires tp_rank 0"};
    }
    if (request->max_step_records() > kMaxPressureStepRecords) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "max_step_records exceeds 1024"};
    }
    const int        base = topology.world_rank - topology.tp_rank;
    std::set<int>    seen;
    std::vector<int> ranks(request->target_ranks().begin(), request->target_ranks().end());
    if (ranks.empty()) {
        for (int i = 0; i < topology.tp_size; ++i) {
            ranks.push_back(base + i);
        }
    }
    for (int rank : ranks) {
        if (rank < base || rank >= base + topology.tp_size || !seen.insert(rank).second) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "duplicate or out-of-topology target rank"};
        }
        response->add_requested_ranks(rank);
    }
    struct Pending {
        grpc::ClientContext                                                       context;
        CachePressureResponsePB                                                   response;
        grpc::Status                                                              status;
        std::shared_ptr<RpcService::Stub>                                         stub;
        std::unique_ptr<grpc::ClientAsyncResponseReader<CachePressureResponsePB>> reader;
        CachePressureRankPB*                                                      record;
        int                                                                       rank;
    };
    grpc::CompletionQueue                 queue;
    std::vector<std::unique_ptr<Pending>> pending;
    const auto                            deadline = std::min(context->deadline(),
                                   std::chrono::system_clock::now() + std::chrono::milliseconds(kPressureRpcTimeoutMs));
    grpc::Status                          local_status;
    const auto&                           addresses = maga_init_params_.runtime_config.worker_grpc_addrs;
    for (int rank : ranks) {
        auto* record = response->add_ranks();
        record->set_world_rank(rank);
        const auto send = pressureNowNs();
        record->set_coordinator_send_ns(send);
        if (rank == topology.world_rank) {
            CachePressureResponsePB local;
            local_status = GetCachePressureInternal(context, request, &local);
            if (local.ranks_size() == 1) {
                *record = local.ranks(0);
            }
            if (!local_status.ok()) {
                record->set_error(local_status.error_message());
            }
            record->set_coordinator_send_ns(send);
            record->set_coordinator_receive_ns(pressureNowNs());
            continue;
        }
        const int index = rank - base;
        if (index >= addresses.size() || addresses[index].empty()) {
            record->set_error("rank endpoint unavailable");
            record->set_coordinator_receive_ns(pressureNowNs());
            continue;
        }
        auto connection = pressure_rpc_pool_.getConnection(addresses[index]);
        if (!connection.ok()) {
            record->set_error(connection.status().ToString());
            record->set_coordinator_receive_ns(pressureNowNs());
            continue;
        }
        auto call    = std::make_unique<Pending>();
        call->record = record;
        call->rank   = rank;
        call->stub   = connection.value().stub;
        call->context.set_deadline(deadline);
        call->reader = call->stub->AsyncGetCachePressureInternal(&call->context, *request, &queue);
        call->reader->Finish(&call->response, &call->status, call.get());
        pending.push_back(std::move(call));
    }
    for (size_t i = 0; i < pending.size(); ++i) {
        void* tag = nullptr;
        bool  ok  = false;
        queue.Next(&tag, &ok);
        auto*      call = static_cast<Pending*>(tag);
        const auto send = call->record->coordinator_send_ns();
        if (ok && call->status.ok() && call->response.schema_version() == 1 && call->response.ranks_size() == 1
            && call->response.ranks(0).world_rank() == call->rank) {
            *call->record = call->response.ranks(0);
        } else {
            call->record->set_error(call->status.ok() ? "invalid rank response" : call->status.error_message());
        }
        call->record->set_coordinator_send_ns(send);
        call->record->set_coordinator_receive_ns(pressureNowNs());
    }
    queue.Shutdown();
    void* drained_tag = nullptr;
    bool  drained_ok  = false;
    while (queue.Next(&drained_tag, &drained_ok)) {}
    response->set_coordinator_sent_ns(pressureNowNs());
    // Preserve peer failures in successful aggregates. A single unsupported local backend is explicit.
    if (ranks.size() == 1 && ranks.front() == topology.world_rank && !local_status.ok()) {
        return local_status;
    }
    return grpc::Status::OK;
}

grpc::Status
LocalRpcServer::StartProfile(grpc::ServerContext* context, const StartProfileRequestPB* request, EmptyPB* response) {
    (void)response;
    RTP_LLM_LOG_INFO("start_profile from %s start_step=%d num_steps=%d enable_all_rank=%d",
                     context->peer().c_str(),
                     request->start_step(),
                     request->num_steps(),
                     int(request->enable_all_rank()));
    if (!request->enable_all_rank()) {
        engine_->startTimelineProfiling(request->trace_name(), request->start_step(), request->num_steps());
        return grpc::Status::OK;
    }
    if (maga_init_params_.parallelism_config.tp_rank != 0) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            "enable_all_rank start_profile must be sent to tp_rank 0");
    }
    if (!tp_broadcaster_) {
        if (maga_init_params_.parallelism_config.tp_size <= 1) {
            RTP_LLM_LOG_INFO("start_profile enable_all_rank with tp_size=1, fallback to local start");
            engine_->startTimelineProfiling(request->trace_name(), request->start_step(), request->num_steps());
            return grpc::Status::OK;
        }
        return grpc::Status(grpc::StatusCode::INTERNAL, "tp broadcaster unavailable for enable_all_rank start_profile");
    }

    std::vector<StartProfileInternalRequestPB> requests(tp_broadcaster_->workerNum());
    for (auto& internal_request : requests) {
        internal_request.set_trace_name(request->trace_name());
        internal_request.set_start_step(request->start_step());
        internal_request.set_num_steps(request->num_steps());
    }
    auto rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& context,
                       const StartProfileInternalRequestPB&        internal_request,
                       grpc::CompletionQueue*                      completion_queue) {
        return stub->AsyncStartProfileInternal(context.get(), internal_request, completion_queue);
    };
    auto broadcast_result =
        tp_broadcaster_->broadcast<StartProfileInternalRequestPB, EmptyPB>(requests, /*timeout_ms=*/3000, rpc_call);
    if (!broadcast_result) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "failed to broadcast start_profile_internal to tp group");
    }
    broadcast_result->waitDone();
    if (!broadcast_result->success()) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "broadcast start_profile_internal to tp group failed");
    }
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::StartProfileInternal(grpc::ServerContext*                 context,
                                                  const StartProfileInternalRequestPB* request,
                                                  EmptyPB*                             response) {
    (void)response;
    RTP_LLM_LOG_INFO("start_profile_internal from %s start_step=%d num_steps=%d",
                     context->peer().c_str(),
                     request->start_step(),
                     request->num_steps());
    engine_->startTimelineProfiling(request->trace_name(), request->start_step(), request->num_steps());
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::StopProfileInternal(grpc::ServerContext*                context,
                                                 const StopProfileInternalRequestPB* request,
                                                 StopProfileResponsePB*              response) {
    return stopProfileLocal(
        context,
        request,
        response,
        std::min(context->deadline(),
                 std::chrono::system_clock::now() + std::chrono::milliseconds(kPressureRpcTimeoutMs)));
}

grpc::Status LocalRpcServer::stopProfileLocal(grpc::ServerContext*                  context,
                                              const StopProfileInternalRequestPB*   request,
                                              StopProfileResponsePB*                response,
                                              std::chrono::system_clock::time_point deadline) {
    response->Clear();
    auto* rank = response->add_ranks();
    rank->set_world_rank(maga_init_params_.parallelism_config.world_rank);
    rank->set_service_instance(pressure_instance_);
    if (!engine_ || !engine_->getProfileStatus().supported) {
        return {grpc::StatusCode::UNIMPLEMENTED, "profile cancellation unsupported"};
    }
    if (request->trace_name().empty() || request->trace_name().size() > 64) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "stop requires the configured trace identity (1..64 bytes)"};
    }
    if (context->IsCancelled() || std::chrono::system_clock::now() >= deadline) {
        return {grpc::StatusCode::DEADLINE_EXCEEDED, "stop request deadline exceeded before acceptance"};
    }
    rank->set_accepted(engine_->stopTimelineProfiling(request->trace_name()));
    if (!rank->accepted()) {
        profileStatusToProto(engine_->getProfileStatus(), rank->mutable_profile());
        return {grpc::StatusCode::FAILED_PRECONDITION, "profile trace identity mismatch"};
    }
    while (true) {
        const auto status = engine_->getProfileStatus();
        profileStatusToProto(status, rank->mutable_profile());
        // A duplicate cleanup of the last finished trace never cancels a newer trace.
        if (status.trace_name != request->trace_name() || status.collection_complete)
            break;
        if (context->IsCancelled() || std::chrono::system_clock::now() >= deadline) {
            rank->set_error("stop accepted; owner-thread collection deadline exceeded");
            break;
        }
        engine_->wakePressureControlProgress();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::StopProfile(grpc::ServerContext*        context,
                                         const StopProfileRequestPB* request,
                                         StopProfileResponsePB*      response) {
    response->Clear();
    StopProfileInternalRequestPB internal;
    internal.set_trace_name(request->trace_name());
    if (!request->enable_all_rank()) {
        if (!request->target_ranks().empty()) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "target_ranks requires enable_all_rank"};
        }
        return StopProfileInternal(context, &internal, response);
    }
    const auto& topology = maga_init_params_.parallelism_config;
    if (topology.tp_rank != 0) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "all-rank stop requires tp_rank 0"};
    }
    const int        base = topology.world_rank;
    std::vector<int> ranks(request->target_ranks().begin(), request->target_ranks().end());
    if (ranks.empty()) {
        for (int i = 0; i < topology.tp_size; ++i)
            ranks.push_back(base + i);
    }
    std::set<int> seen;
    for (int rank : ranks) {
        if (rank < base || rank >= base + topology.tp_size || !seen.insert(rank).second) {
            return {grpc::StatusCode::INVALID_ARGUMENT, "duplicate or out-of-topology stop target"};
        }
    }
    struct Pending {
        grpc::ClientContext                                                     context;
        StopProfileResponsePB                                                   response;
        grpc::Status                                                            status;
        std::shared_ptr<RpcService::Stub>                                       stub;
        std::unique_ptr<grpc::ClientAsyncResponseReader<StopProfileResponsePB>> reader;
        StopProfileRankPB*                                                      record;
        int                                                                     rank;
    };
    const auto                            deadline = std::min(context->deadline(),
                                   std::chrono::system_clock::now() + std::chrono::milliseconds(kPressureRpcTimeoutMs));
    grpc::CompletionQueue                 queue;
    std::vector<std::unique_ptr<Pending>> pending;
    const auto&                           addresses = maga_init_params_.runtime_config.worker_grpc_addrs;
    StopProfileRankPB*                    local     = nullptr;
    for (int rank : ranks) {
        auto* record = response->add_ranks();
        record->set_world_rank(rank);
        if (rank == topology.world_rank) {
            local = record;
            continue;
        }
        const int index = rank - base;
        if (index >= addresses.size() || addresses[index].empty()) {
            record->set_error("rank endpoint unavailable");
            continue;
        }
        auto connection = pressure_rpc_pool_.getConnection(addresses[index]);
        if (!connection.ok()) {
            record->set_error(connection.status().ToString());
            continue;
        }
        auto call    = std::make_unique<Pending>();
        call->record = record;
        call->rank   = rank;
        call->stub   = connection.value().stub;
        call->context.set_deadline(deadline);
        call->reader = call->stub->AsyncStopProfileInternal(&call->context, internal, &queue);
        call->reader->Finish(&call->response, &call->status, call.get());
        pending.push_back(std::move(call));
    }
    if (local) {
        // Give local handling the same bounded deadline as peers.
        StopProfileResponsePB local_response;
        const auto            status = stopProfileLocal(context, &internal, &local_response, deadline);
        if (local_response.ranks_size())
            *local = local_response.ranks(0);
        if (!status.ok())
            local->set_error(status.error_message());
    }
    bool cancelled = false;
    for (size_t completed = 0; completed < pending.size();) {
        void*      tag = nullptr;
        bool       ok  = false;
        const auto now = std::chrono::system_clock::now();
        if (now >= deadline || context->IsCancelled()) {
            if (!cancelled) {
                for (auto& call : pending)
                    call->context.TryCancel();
                cancelled = true;
            }
            queue.Next(&tag, &ok);
        } else {
            // Internal responses wait for owner-thread collection, not merely
            // acceptance. Delayed peer flags therefore still receive TP progress.
            if (engine_)
                engine_->wakePressureControlProgress();
            if (queue.AsyncNext(&tag, &ok, std::min(deadline, now + std::chrono::milliseconds(10)))
                != grpc::CompletionQueue::GOT_EVENT)
                continue;
        }
        ++completed;
        auto* call = static_cast<Pending*>(tag);
        if (ok && call->status.ok() && call->response.ranks_size() == 1
            && call->response.ranks(0).world_rank() == call->rank) {
            *call->record = call->response.ranks(0);
        } else {
            call->record->set_error(call->status.ok() ? "invalid stop response" : call->status.error_message());
        }
    }
    queue.Shutdown();
    void* tag = nullptr;
    bool  ok  = false;
    while (queue.Next(&tag, &ok)) {}
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::authorizeTorchAllocatorDump(const TorchAllocatorDumpRequestPB& request) const {
    if (!torch_allocator_dump_enabled_) {
        return grpc::Status(grpc::StatusCode::PERMISSION_DENIED, "allocator dump unavailable");
    }
    if (torch_allocator_dump_auth_token_.empty()
        || !constantTimeEquals(request.auth_token(), torch_allocator_dump_auth_token_)) {
        return grpc::Status(grpc::StatusCode::UNAUTHENTICATED, "allocator dump authentication failed");
    }
    if (!validAllocatorDumpId(request.dump_id())) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "invalid allocator dump id");
    }
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::authorizeTorchAllocatorDumpInternal(grpc::ServerContext*               context,
                                                                 const TorchAllocatorDumpRequestPB& request) const {
    const auto authorization = authorizeTorchAllocatorDump(request);
    if (!authorization.ok()) {
        return authorization;
    }
    if (maga_init_params_.parallelism_config.tp_size <= 1 || !isTorchAllocatorDumpInternalPeer(context)) {
        return grpc::Status(grpc::StatusCode::PERMISSION_DENIED, "allocator dump internal fanout denied");
    }
    return grpc::Status::OK;
}

bool LocalRpcServer::isTorchAllocatorDumpInternalPeer(grpc::ServerContext* context) const {
    if (context == nullptr) {
        return false;
    }
    const auto peer_host = endpointHost(context->peer());
    if (peer_host.empty()) {
        return false;
    }
    return std::any_of(maga_init_params_.runtime_config.worker_grpc_addrs.begin(),
                       maga_init_params_.runtime_config.worker_grpc_addrs.end(),
                       [&peer_host](const std::string& worker_addr) { return endpointHost(worker_addr) == peer_host; });
}

grpc::Status
LocalRpcServer::beginTorchAllocatorDump(const std::string& dump_id, bool internal_fanout, bool* owns_admission) {
    std::lock_guard<std::mutex> lock(torch_allocator_dump_mutex_);
    if (owns_admission != nullptr) {
        *owns_admission = false;
    }
    if (internal_fanout && torch_allocator_dump_in_progress_ && torch_allocator_dump_active_started_by_public_
        && !torch_allocator_dump_active_internal_started_ && torch_allocator_dump_active_id_ == dump_id) {
        torch_allocator_dump_active_internal_started_ = true;
        return grpc::Status::OK;
    }
    if (torch_allocator_dump_ids_.count(dump_id) != 0) {
        return grpc::Status(grpc::StatusCode::ALREADY_EXISTS, "allocator dump id has already been used");
    }
    if (torch_allocator_dump_in_progress_) {
        return grpc::Status(grpc::StatusCode::ABORTED, "allocator dump already in progress");
    }
    if (torch_allocator_dump_has_completed_) {
        const auto elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - torch_allocator_dump_last_completed_at_)
                .count();
        if (elapsed < torch_allocator_dump_cooldown_seconds_) {
            return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, "allocator dump cooldown active");
        }
    }
    if (torch_allocator_dump_id_order_.size() == kAllocatorDumpReplayHistoryMax) {
        torch_allocator_dump_ids_.erase(torch_allocator_dump_id_order_.front());
        torch_allocator_dump_id_order_.pop_front();
    }
    torch_allocator_dump_ids_.insert(dump_id);
    torch_allocator_dump_id_order_.push_back(dump_id);
    torch_allocator_dump_in_progress_              = true;
    torch_allocator_dump_active_started_by_public_ = !internal_fanout;
    torch_allocator_dump_active_internal_started_  = false;
    torch_allocator_dump_active_id_                = dump_id;
    if (owns_admission != nullptr) {
        *owns_admission = true;
    }
    return grpc::Status::OK;
}

void LocalRpcServer::finishTorchAllocatorDump() {
    std::lock_guard<std::mutex> lock(torch_allocator_dump_mutex_);
    torch_allocator_dump_in_progress_              = false;
    torch_allocator_dump_has_completed_            = true;
    torch_allocator_dump_active_started_by_public_ = false;
    torch_allocator_dump_active_internal_started_  = false;
    torch_allocator_dump_active_id_.clear();
    torch_allocator_dump_last_completed_at_ = std::chrono::steady_clock::now();
}

grpc::Status LocalRpcServer::aggregateTorchAllocatorDumpResults(const std::string&                             dump_id,
                                                                const std::vector<TorchAllocatorDumpResultPB>& results,
                                                                TorchAllocatorDumpResponsePB* response) const {
    for (const auto& result : results) {
        if (result.dump_id() != dump_id) {
            response->Clear();
            RTP_LLM_LOG_WARNING("allocator dump response id mismatch expected=%s actual=%s world_rank=%ld",
                                dump_id.c_str(),
                                result.dump_id().c_str(),
                                result.world_rank());
            return grpc::Status(grpc::StatusCode::DATA_LOSS, "allocator dump response id mismatch");
        }
    }
    for (const auto& result : results) {
        response->add_results()->CopyFrom(result);
    }
    return grpc::Status::OK;
}

TorchAllocatorDumpResultPB LocalRpcServer::dumpTorchAllocatorOnCurrentProcess(const std::string& dump_id) {
    const auto&                parallelism_config = maga_init_params_.parallelism_config;
    TorchAllocatorDumpResultPB result;
    result.set_world_rank(parallelism_config.world_rank);
    result.set_dp_rank(parallelism_config.dp_rank);
    result.set_tp_rank(parallelism_config.tp_rank);
    result.set_local_rank(parallelism_config.local_rank);
    result.set_pid(getpid());
    result.set_dump_id(dump_id);

    auto output_path = dumpTorchCudaOomDiagnostics(parallelism_config.local_rank, dump_id);
    result.set_success(!output_path.empty());
    result.set_file_path(std::move(output_path));
    if (!result.success()) {
        result.set_error("allocator dump failed; see backend log");
    }
    return result;
}

grpc::Status LocalRpcServer::executeAdmittedTorchAllocatorDump(const TorchAllocatorDumpRequestPB& request,
                                                               TorchAllocatorDumpResponsePB*      response) {
    if (maga_init_params_.parallelism_config.tp_size <= 1) {
        return aggregateTorchAllocatorDumpResults(
            request.dump_id(), {dumpTorchAllocatorOnCurrentProcess(request.dump_id())}, response);
    }
    if (!tp_broadcaster_) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "tp broadcaster unavailable for allocator dump");
    }

    std::vector<TorchAllocatorDumpRequestPB> requests(tp_broadcaster_->workerNum(), request);
    auto                                     rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& context,
                       const TorchAllocatorDumpRequestPB&          internal_request,
                       grpc::CompletionQueue*                      completion_queue) {
        return stub->AsyncDumpTorchAllocatorInternal(context.get(), internal_request, completion_queue);
    };
    auto broadcast_result = tp_broadcaster_->broadcast<TorchAllocatorDumpRequestPB, TorchAllocatorDumpResultPB>(
        requests, /*timeout_ms=*/60000, rpc_call);
    if (!broadcast_result) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "failed to broadcast allocator dump to tp group");
    }
    try {
        broadcast_result->waitDone();
    } catch (const std::exception& exception) {
        RTP_LLM_LOG_WARNING(
            "allocator dump broadcast failed dump_id=%s: %s", request.dump_id().c_str(), exception.what());
        return grpc::Status(grpc::StatusCode::INTERNAL, "allocator dump broadcast failed");
    }
    if (!broadcast_result->success()) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "allocator dump broadcast to tp group failed");
    }
    return aggregateTorchAllocatorDumpResults(request.dump_id(), broadcast_result->responses(), response);
}

grpc::Status LocalRpcServer::DumpTorchAllocator(grpc::ServerContext*               context,
                                                const TorchAllocatorDumpRequestPB* request,
                                                TorchAllocatorDumpResponsePB*      response) {
    response->Clear();
    const auto authorization = authorizeTorchAllocatorDump(*request);
    if (!authorization.ok()) {
        RTP_LLM_LOG_WARNING("rejected dump_torch_allocator from %s", context->peer().c_str());
        return authorization;
    }
    RTP_LLM_LOG_INFO("dump_torch_allocator from %s dump_id=%s", context->peer().c_str(), request->dump_id().c_str());
    if (maga_init_params_.parallelism_config.tp_rank != 0) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "dump_torch_allocator must be sent to tp_rank 0");
    }
    const auto admission = beginTorchAllocatorDump(request->dump_id());
    if (!admission.ok()) {
        return admission;
    }

    try {
        const auto result = executeAdmittedTorchAllocatorDump(*request, response);
        finishTorchAllocatorDump();
        return result;
    } catch (...) {
        finishTorchAllocatorDump();
        throw;
    }
}

grpc::Status LocalRpcServer::DumpTorchAllocatorInternal(grpc::ServerContext*               context,
                                                        const TorchAllocatorDumpRequestPB* request,
                                                        TorchAllocatorDumpResultPB*        response) {
    response->Clear();
    const auto authorization = authorizeTorchAllocatorDumpInternal(context, *request);
    if (!authorization.ok()) {
        RTP_LLM_LOG_WARNING("rejected dump_torch_allocator_internal from %s", context->peer().c_str());
        return authorization;
    }
    bool       owns_admission = false;
    const auto admission      = beginTorchAllocatorDump(request->dump_id(), /*internal_fanout=*/true, &owns_admission);
    if (!admission.ok()) {
        return admission;
    }
    RTP_LLM_LOG_INFO(
        "dump_torch_allocator_internal from %s dump_id=%s", context->peer().c_str(), request->dump_id().c_str());
    try {
        response->CopyFrom(dumpTorchAllocatorOnCurrentProcess(request->dump_id()));
        if (owns_admission) {
            finishTorchAllocatorDump();
        }
        return grpc::Status::OK;
    } catch (...) {
        if (owns_admission) {
            finishTorchAllocatorDump();
        }
        throw;
    }
}

grpc::Status
LocalRpcServer::CheckHealth(grpc::ServerContext* context, const EmptyPB* request, CheckHealthResponsePB* response) {
    RTP_LLM_LOG_DEBUG("receive cacheStatus rpc request from client: %s", context->peer().c_str());
    response->set_health("OK");
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::UpdateEplbConfig(grpc::ServerContext*             context,
                                              const UpdateEplbConfigRequestPB* request,
                                              EmptyPB*                         response) {
    RTP_LLM_LOG_DEBUG("receive cacheStatus rpc request from client: %s", context->peer().c_str());
    const string mode_str = request->mode();
    EPLBConfig   config;
    if (mode_str == "EPLB" || mode_str == "eplb") {
        config.eplb_mode = EplbMode::EPLB;
    } else if (mode_str == "STATS" || mode_str == "stats") {
        config.eplb_mode = EplbMode::STATS;
    } else if (mode_str == "NONE" || mode_str == "none") {
        config.eplb_mode = EplbMode::NONE;
    } else if (mode_str == "ALL" || mode_str == "all") {
        config.eplb_mode = EplbMode::ALL;
    } else {
        RTP_LLM_LOG_WARNING("set eplb mode failed, unknown mode : %s", mode_str.c_str());
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Invalid eplb mode");
    }
    config.eplb_update_time = request->update_time();
    engine_->updateEplbConfig(config);
    return grpc::Status::OK;
}

void LocalRpcServer::reportWorkerStatusTime(int64_t request_begin_time_us, int64_t request_after_ws_time_us) {
    RpcWorkerStatusMetricsCollector collector;
    collector.qps         = true;
    collector.total_rt_us = request_after_ws_time_us - request_begin_time_us;
    if (metrics_reporter_) {
        metrics_reporter_->report<RpcWorkerStatusMetrics, RpcWorkerStatusMetricsCollector>(nullptr, &collector);
    }
}

void LocalRpcServer::reportCacheStatusTime(int64_t request_begin_time_us) {
    RpcCacheStatusMetricsCollector collector;
    collector.qps         = true;
    collector.total_rt_us = (currentTimeUs() - request_begin_time_us);
    if (metrics_reporter_) {
        metrics_reporter_->report<RpcCacheStatusMetrics, RpcCacheStatusMetricsCollector>(nullptr, &collector);
    }
}

::grpc::Status LocalRpcServer::ExecuteFunction(::grpc::ServerContext*     context,
                                               const ::FunctionRequestPB* request,
                                               ::FunctionResponsePB*      response) {
    RTP_LLM_LOG_DEBUG("receive execute function request from client: %s, request: [%s]",
                      context->peer().c_str(),
                      request->DebugString().c_str());
    if (context->IsCancelled()) {
        RTP_LLM_LOG_WARNING("execute function failed, request is cancelled");
        return grpc::Status(grpc::StatusCode::CANCELLED, "request is cancelled");
    }
    if (!engine_) {
        RTP_LLM_LOG_WARNING("execute function failed, engine is null");
        return grpc::Status(grpc::StatusCode::INTERNAL, "engine is null");
    }

    auto cache_manager = engine_->getCacheManager();
    if (!cache_manager) {
        RTP_LLM_LOG_WARNING("execute function failed, cache manager is null");
        return grpc::Status(grpc::StatusCode::INTERNAL, "cache manager is null");
    }
    if (!cache_manager->executeFunction(*request, *response)) {
        RTP_LLM_LOG_WARNING("execute function failed, request: [%s]", request->DebugString().c_str());
        const std::string error_msg = "execute function failed, request: [" + request->DebugString() + "]";
        return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
    }
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::SetPause(grpc::ServerContext* context, const EmptyPB* request, EmptyPB* response) {
    RTP_LLM_LOG_DEBUG("receive cacheStatus rpc request from client: %s", context->peer().c_str());
    engine_->pause();
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::SetRestart(grpc::ServerContext* context, const EmptyPB* request, EmptyPB* response) {
    RTP_LLM_LOG_DEBUG("receive cacheStatus rpc request from client: %s,", context->peer().c_str());
    engine_->restart();
    return grpc::Status::OK;
}

grpc::Status
LocalRpcServer::UpdateWeights(grpc::ServerContext* context, const UpdateWeightsRequestPB* request, EmptyPB* response) {
    RTP_LLM_LOG_DEBUG("Receive update weights request from: %s", context->peer().c_str());
    try {
        if (request->name().empty() || request->desc().empty() || request->method().empty()) {
            throw std::runtime_error("Missing required field(s) in request");
        }
        {
            py::gil_scoped_acquire acquire;
            py::dict               req;
            req["name"]   = request->name();
            req["desc"]   = request->desc();
            req["method"] = request->method();
            weight_manager_.attr("update")(req);
        }
        return grpc::Status::OK;
    } catch (const py::error_already_set& e) {
        PyObject *type, *value, *traceback;
        PyErr_Fetch(&type, &value, &traceback);
        std::string err_msg = value ? PyUnicode_AsUTF8(value) : "Unknown Python error";
        return {grpc::StatusCode::INTERNAL, "exception from python: " + err_msg};
    } catch (const std::exception& e) {
        return {grpc::StatusCode::INTERNAL, "exception from C++: " + std::string(e.what())};
    }
}
}  // namespace rtp_llm
