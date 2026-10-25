#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>

namespace rtp_llm {

struct MockMemoryHitDecision {
    size_t   eligible_blocks{0};
    size_t   planned_host_blocks{0};
    uint64_t cumulative_eligible_blocks{0};
    uint64_t cumulative_planned_host_blocks{0};
};

struct MockMemoryHitSnapshot {
    uint64_t cumulative_eligible_blocks{0};
    uint64_t cumulative_planned_host_blocks{0};
};

// Deterministic cumulative planning with an effective ratio resolution of 1e-6.
// Admission order is the sole input: prefix deficits carry forward to later calls.
class MockMemoryHitController {
public:
    explicit MockMemoryHitController(double ratio);

    // Validation runs under the controller mutex and must not reenter this controller.
    // Throws if the counter would overflow or no prefix (including zero) is valid;
    // failed calls do not change the cumulative counters.
    MockMemoryHitDecision plan(size_t eligible_blocks, const std::function<bool(size_t)>& is_valid_prefix);
    MockMemoryHitSnapshot snapshot() const;

private:
    uint64_t              ratio_ppm_{0};
    mutable std::mutex    mutex_;
    MockMemoryHitSnapshot state_;
};

}  // namespace rtp_llm
