#include "rtp_llm/cpp/normal_engine/EngineModelStartup.h"
#include "gtest/gtest.h"
#include <pybind11/embed.h>
#include <c10/core/InferenceMode.h>
#include "autil/EnvUtil.h"

namespace py = pybind11;
using namespace rtp_llm;

class EngineModelStartupTest: public ::testing::Test {
protected:
    void SetUp() override {
        py::exec(R"(
import sys, types
for name in ('rtp_llm', 'rtp_llm.utils'):
    package = types.ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
bridge = types.ModuleType('rtp_llm.utils.model_startup')
bridge.events = []
def prepare(models, context):
    bridge.events.append(('prepare', len(models), context))
    for model in models: model.ready = True
    return True
def check(models):
    if not all(getattr(m, 'ready', False) for m in models):
        raise RuntimeError('models not ready before KV')
    bridge.events.append(('check', len(models)))
    return True
bridge.prepare_engine_models = prepare
bridge.assert_engine_models_ready = check
sys.modules['rtp_llm.utils.model_startup'] = bridge
)");
    }
    EngineInitParams parameters(bool enabled = true) {
        EngineInitParams p;
        auto             types                                        = py::module_::import("types");
        p.py_model                                                    = types.attr("SimpleNamespace")();
        p.py_model.attr("_startup_lifecycle")                         = enabled ? py::cast(true) : py::none();
        p.py_sp_model                                                 = types.attr("SimpleNamespace")();
        p.py_sp_model.attr("_startup_lifecycle")                      = py::cast(true);
        p.parallelism_config.pp_size                                  = 1;
        p.parallelism_config.role_type                                = RoleType::PDFUSION;
        p.runtime_config.fifo_scheduler_config.max_context_batch_size = 3;
        p.hw_kernel_config.enable_cuda_graph                          = false;
        p.sp_config.type                                              = SP_TYPE_MTP;
        return p;
    }
};

TEST_F(EngineModelStartupTest, PreparesTargetAndDraftWithNativeContext) {
    auto p = parameters();
    EXPECT_TRUE(prepareEngineModelStartup(p));
    auto events = py::module_::import("rtp_llm.utils.model_startup").attr("events").cast<py::list>();
    ASSERT_EQ(events.size(), 1);
    auto e = events[0].cast<py::tuple>();
    EXPECT_EQ(e[0].cast<std::string>(), "prepare");
    EXPECT_EQ(e[1].cast<int>(), 2);
    auto context = e[2].cast<py::dict>();
    EXPECT_TRUE(context["is_speculative"].cast<bool>());
    EXPECT_FALSE(context["is_decode_role"].cast<bool>());
    EXPECT_EQ(context["max_context_batch_size"].cast<int>(), 3);
    EXPECT_NO_THROW(assertEngineModelStartupReady(p));
}

TEST_F(EngineModelStartupTest, RejectsKVReadinessBeforePreparation) {
    auto p = parameters();
    EXPECT_ANY_THROW(assertEngineModelStartupReady(p));
}

TEST_F(EngineModelStartupTest, PreparationPreservesWrapperInferenceContext) {
    auto p                               = parameters();
    bool observed                        = false;
    auto bridge                          = py::module_::import("rtp_llm.utils.model_startup");
    bridge.attr("prepare_engine_models") = py::cpp_function([&observed](py::list, py::dict) {
        observed = c10::InferenceMode::is_enabled();
        return true;
    });
    EXPECT_TRUE(prepareEngineModelStartup(p));
    EXPECT_TRUE(observed);
}

TEST_F(EngineModelStartupTest, DisabledDoesNotCallPythonBridge) {
    auto p = parameters(false);
    EXPECT_FALSE(prepareEngineModelStartup(p));
    EXPECT_NO_THROW(assertEngineModelStartupReady(p));
    auto events = py::module_::import("rtp_llm.utils.model_startup").attr("events").cast<py::list>();
    EXPECT_EQ(events.size(), 0);
}

TEST_F(EngineModelStartupTest, MissingDraftIsRejectedBeforePythonPreparation) {
    auto p        = parameters();
    p.py_sp_model = py::none();
    EXPECT_ANY_THROW(prepareEngineModelStartup(p));
}

TEST_F(EngineModelStartupTest, RejectsCPFixedWidthBeforePreparingModels) {
    for (const auto* enabled : {"RTP_LLM_STREAM_ASYNC", "RTP_LLM_MTP_ASYNC_DEVICE_STATE"}) {
        autil::EnvGuard stream_async("RTP_LLM_STREAM_ASYNC", "0");
        autil::EnvGuard device_state("RTP_LLM_MTP_ASYNC_DEVICE_STATE", "0");
        autil::EnvGuard fixed_width(enabled, "1");
        auto            p                             = parameters();
        p.parallelism_config.tp_size                  = 2;
        p.parallelism_config.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
        p.parallelism_config.role_type                = RoleType::PREFILL;
        p.sp_config.gen_num_per_cycle                 = 3;
        EXPECT_THROW(prepareEngineModelStartup(p), std::runtime_error);
        auto events = py::module_::import("rtp_llm.utils.model_startup").attr("events").cast<py::list>();
        EXPECT_EQ(events.size(), 0);
    }
}

TEST_F(EngineModelStartupTest, RejectsCPMtpDecodeBeforePreparingModels) {
    autil::EnvGuard stream_async("RTP_LLM_STREAM_ASYNC", "0");
    autil::EnvGuard device_state("RTP_LLM_MTP_ASYNC_DEVICE_STATE", "0");
    auto            p                             = parameters();
    p.parallelism_config.tp_size                  = 8;
    p.parallelism_config.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
    p.parallelism_config.role_type                = RoleType::PDFUSION;
    p.sp_config.gen_num_per_cycle                 = 3;
    EXPECT_THROW(prepareEngineModelStartup(p), std::runtime_error);
    auto events = py::module_::import("rtp_llm.utils.model_startup").attr("events").cast<py::list>();
    EXPECT_EQ(events.size(), 0);
}

TEST_F(EngineModelStartupTest, TargetOnlyAndDecodeContext) {
    auto p                         = parameters();
    p.sp_config.type               = SP_TYPE_NONE;
    p.parallelism_config.role_type = RoleType::DECODE;
    EXPECT_TRUE(prepareEngineModelStartup(p));
    auto events = py::module_::import("rtp_llm.utils.model_startup").attr("events").cast<py::list>();
    auto e      = events[0].cast<py::tuple>();
    EXPECT_EQ(e[1].cast<int>(), 1);
    auto context = e[2].cast<py::dict>();
    EXPECT_FALSE(context["is_speculative"].cast<bool>());
    EXPECT_TRUE(context["is_decode_role"].cast<bool>());
}

TEST_F(EngineModelStartupTest, UnsupportedTopologiesRejected) {
    auto p                       = parameters();
    p.parallelism_config.pp_size = 2;
    EXPECT_ANY_THROW(prepareEngineModelStartup(p));
    p.parallelism_config.pp_size         = 1;
    p.hw_kernel_config.enable_cuda_graph = true;
    EXPECT_ANY_THROW(prepareEngineModelStartup(p));
    p.hw_kernel_config.enable_cuda_graph = false;
    p.sp_config.type                     = SP_TYPE_DSPARK;
    EXPECT_ANY_THROW(prepareEngineModelStartup(p));
}

TEST_F(EngineModelStartupTest, WarmupKVReleasedForBothModelsWithOriginalContext) {
    auto p      = parameters();
    auto calls  = py::list();
    auto rebind = py::cpp_function([calls](py::object resources) mutable { calls.append(resources); });
    p.py_model.attr("initialize")    = rebind;
    p.py_sp_model.attr("initialize") = rebind;
    EXPECT_TRUE(releaseEngineWarmupKV(p));
    ASSERT_EQ(calls.size(), 2);
    for (auto resources : calls) {
        EXPECT_TRUE(resources.attr("kv_cache").is_none());
        EXPECT_TRUE(resources.attr("is_speculative").cast<bool>());
        EXPECT_EQ(resources.attr("max_context_batch_size").cast<int>(), 3);
    }
}

TEST_F(EngineModelStartupTest, EmptyPythonModelDisabled) {
    EngineInitParams p;
    EXPECT_FALSE(prepareEngineModelStartup(p));
    EXPECT_NO_THROW(assertEngineModelStartupReady(p));
    EXPECT_FALSE(releaseEngineWarmupKV(p));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    // Native-only engines must not attempt to acquire a nonexistent Python GIL.
    {
        EngineInitParams p;
        if (prepareEngineModelStartup(p) || releaseEngineWarmupKV(p)) {
            return 1;
        }
        assertEngineModelStartupReady(p);
    }
    py::scoped_interpreter interpreter{};
    return RUN_ALL_TESTS();
}
