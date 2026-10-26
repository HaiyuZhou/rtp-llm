#include "rtp_llm/cpp/cache/MockMemoryHitController.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace rtp_llm {

namespace {
constexpr uint64_t kRatioScale = 1'000'000;
}  // namespace

MockMemoryHitController::MockMemoryHitController(double ratio) {
    if (!std::isfinite(ratio) || ratio < 0.0 || ratio > 1.0) {
        throw std::invalid_argument("mock memory hit ratio must be finite and between zero and one");
    }
    ratio_ppm_ = static_cast<uint64_t>(std::llround(ratio * kRatioScale));
}

MockMemoryHitDecision MockMemoryHitController::plan(size_t                             eligible_blocks,
                                                    const std::function<bool(size_t)>& is_valid_prefix) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (eligible_blocks > std::numeric_limits<uint64_t>::max() - state_.cumulative_eligible_blocks) {
        throw std::overflow_error("mock memory hit cumulative eligible blocks overflow");
    }
    const uint64_t eligible_next = state_.cumulative_eligible_blocks + eligible_blocks;
    const uint64_t target =
        static_cast<uint64_t>((static_cast<unsigned __int128>(eligible_next) * ratio_ppm_) / kRatioScale);
    const uint64_t deficit =
        target > state_.cumulative_planned_host_blocks ? target - state_.cumulative_planned_host_blocks : 0;
    size_t candidate = static_cast<size_t>(std::min<uint64_t>(deficit, eligible_blocks));
    while (!is_valid_prefix(candidate)) {
        if (candidate == 0) {
            throw std::invalid_argument("mock memory hit planning requires a valid prefix");
        }
        --candidate;
    }

    // candidate <= target - planned_so_far, so the planned counter cannot overflow.
    state_.cumulative_eligible_blocks = eligible_next;
    state_.cumulative_planned_host_blocks += candidate;
    return {eligible_blocks, candidate, state_.cumulative_eligible_blocks, state_.cumulative_planned_host_blocks};
}

MockMemoryHitSnapshot MockMemoryHitController::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

}  // namespace rtp_llm
