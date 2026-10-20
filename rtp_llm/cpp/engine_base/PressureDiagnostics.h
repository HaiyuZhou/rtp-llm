#pragma once
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
namespace rtp_llm {
// This dedicated status distinguishes unavailable capability from collection failure.
inline constexpr ErrorCode                kPressureDiagnosticsUnsupported = static_cast<ErrorCode>(55101);
inline constexpr uint32_t                 kMaxPressureStepRecords         = 1024;
inline constexpr int                      kPressureRpcTimeoutMs           = 1000;
inline ErrorResult<CachePressureSnapshot> unsupportedCachePressure() {
    return {kPressureDiagnosticsUnsupported, "cache pressure diagnostics unsupported"};
}
}  // namespace rtp_llm
