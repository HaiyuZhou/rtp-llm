#pragma once
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"

namespace rtp_llm {
// Optional Python phase allocation: prepare every active local model before KV.
bool prepareEngineModelStartup(const EngineInitParams& params);
void assertEngineModelStartupReady(const EngineInitParams& params);
bool releaseEngineWarmupKV(const EngineInitParams& params);
}  // namespace rtp_llm
