#include "rtp_llm/cpp/normal_engine/EngineModelStartup.h"
#include <c10/core/InferenceMode.h>
#include <cstdlib>
#include <stdexcept>

namespace rtp_llm {
namespace {
bool enabled(const EngineInitParams& params) {
    return params.py_model && !params.py_model.is_none()
           && !py::getattr(params.py_model, "_startup_lifecycle", py::none()).is_none();
}

py::list participants(const EngineInitParams& params) {
    if (params.parallelism_config.pp_size != 1 || params.hw_kernel_config.enable_cuda_graph) {
        throw std::runtime_error("Phase startup requires PP1 and enable_cuda_graph=0");
    }
    py::list models;
    models.append(params.py_model);
    if (params.sp_config.type == SP_TYPE_MTP) {
        if (params.parallelism_config.tp_size > 1 && params.parallelism_config.prefill_cp_config.is_enabled()
            && params.parallelism_config.role_type != RoleType::PREFILL) {
            throw std::runtime_error(
                "Phase MTP with CP currently supports the PREFILL role only; "
                "DSV4 target verification requires a global decode layout. Disable CP for PDFUSION/DECODE");
        }
        const auto enabled_flag = [](const char* name) {
            const char* value = std::getenv(name);
            return value != nullptr && std::string(value) == "1";
        };
        if (params.parallelism_config.tp_size > 1 && params.parallelism_config.prefill_cp_config.is_enabled()
            && params.sp_config.gen_num_per_cycle > 1
            && (enabled_flag("RTP_LLM_STREAM_ASYNC") || enabled_flag("RTP_LLM_MTP_ASYNC_DEVICE_STATE"))) {
            throw std::runtime_error(
                "Phase MTP with CP currently requires COMPACT draft inputs: set RTP_LLM_STREAM_ASYNC=0 "
                "and RTP_LLM_MTP_ASYNC_DEVICE_STATE=0");
        }
        if (!params.py_sp_model || params.py_sp_model.is_none()) {
            throw std::runtime_error("Phase MTP startup requires a draft model before KV");
        }
        models.append(params.py_sp_model);
    } else if (params.sp_config.type != SP_TYPE_NONE) {
        throw std::runtime_error("Phase startup currently supports ordinary MTP or no speculative model");
    }
    return models;
}

py::dict context(const EngineInitParams& params) {
    py::dict resources;
    resources["is_speculative"]         = params.sp_config.type != SP_TYPE_NONE;
    resources["is_decode_role"]         = params.parallelism_config.role_type == RoleType::DECODE;
    resources["max_context_batch_size"] = params.runtime_config.fifo_scheduler_config.max_context_batch_size;
    return resources;
}
}  // namespace

bool prepareEngineModelStartup(const EngineInitParams& params) {
    if (!Py_IsInitialized()) {
        return false;
    }
    py::gil_scoped_acquire gil;
    if (!enabled(params)) {
        return false;
    }
    // Match PyWrappedModel's initialization context when moving conversion
    // and runtime allocation ahead of wrapper construction.
    c10::InferenceMode inference_guard(true);
    auto               models = participants(params);
    auto               bridge = py::module_::import("rtp_llm.utils.model_startup");
    if (!bridge.attr("prepare_engine_models")(models, context(params)).cast<bool>()) {
        throw std::runtime_error("Phase startup preparation did not prepare registered engine models");
    }
    return true;
}

void assertEngineModelStartupReady(const EngineInitParams& params) {
    if (!Py_IsInitialized()) {
        return;
    }
    py::gil_scoped_acquire gil;
    if (!enabled(params)) {
        return;
    }
    auto bridge = py::module_::import("rtp_llm.utils.model_startup");
    if (!bridge.attr("assert_engine_models_ready")(participants(params)).cast<bool>()) {
        throw std::runtime_error("Phase startup models are not ready before KV");
    }
}

bool releaseEngineWarmupKV(const EngineInitParams& params) {
    if (!Py_IsInitialized()) {
        return false;
    }
    py::gil_scoped_acquire gil;
    if (!enabled(params)) {
        return false;
    }
    auto resources        = context(params);
    resources["kv_cache"] = py::none();
    auto empty            = py::module_::import("types").attr("SimpleNamespace")(**resources);
    for (auto model : participants(params)) {
        model.attr("initialize")(empty);
    }
    return true;
}
}  // namespace rtp_llm
