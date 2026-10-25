#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#include "rtp_llm/cpp/cache/MockMemoryHitController.h"

namespace rtp_llm {
namespace test {
namespace {
bool anyPrefix(size_t) {
    return true;
}
}  // namespace

TEST(MockMemoryHitControllerTest, RatiosUseCumulativeTargetsAcrossVariableLengths) {
    struct Case {
        double              ratio;
        std::vector<size_t> planned;
    };
    // Eligible prefixes total 1, 3, 6, 10, 17; expectations are hand-derived.
    for (const auto& test : std::vector<Case>{
             {0.0, {0, 0, 0, 0, 0}}, {0.5, {0, 1, 2, 2, 3}}, {1.0, {1, 2, 3, 4, 7}}, {0.333333, {0, 0, 1, 2, 2}}}) {
        MockMemoryHitController   controller(test.ratio);
        const std::vector<size_t> lengths{1, 2, 3, 4, 7};
        uint64_t                  eligible = 0;
        uint64_t                  planned  = 0;
        for (size_t i = 0; i < lengths.size(); ++i) {
            const auto decision = controller.plan(lengths[i], anyPrefix);
            eligible += lengths[i];
            planned += test.planned[i];
            EXPECT_EQ(decision.eligible_blocks, lengths[i]);
            EXPECT_EQ(decision.planned_host_blocks, test.planned[i]);
            EXPECT_LE(decision.planned_host_blocks, lengths[i]);
            EXPECT_EQ(decision.cumulative_eligible_blocks, eligible);
            EXPECT_EQ(decision.cumulative_planned_host_blocks, planned);
        }
        EXPECT_EQ(controller.snapshot().cumulative_eligible_blocks, 17u);
        EXPECT_EQ(controller.snapshot().cumulative_planned_host_blocks, planned);
    }
}

TEST(MockMemoryHitControllerTest, QuantizesRatioOnceToMillionths) {
    MockMemoryHitController rounded_down(0.3333334);
    MockMemoryHitController rounded_up(0.3333336);
    EXPECT_EQ(rounded_down.plan(1'000'000, anyPrefix).planned_host_blocks, 333'333u);
    EXPECT_EQ(rounded_up.plan(1'000'000, anyPrefix).planned_host_blocks, 333'334u);
}

TEST(MockMemoryHitControllerTest, RejectsRatiosOutsideFiniteUnitInterval) {
    for (double ratio : {-0.1,
                         1.1,
                         std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT_THROW(MockMemoryHitController{ratio}, std::invalid_argument);
    }
}

TEST(MockMemoryHitControllerTest, RoundsDownToLargestValidPrefixAndRecoversDeficit) {
    MockMemoryHitController controller(0.5);
    const auto              multiples_of_four = [](size_t blocks) { return blocks % 4 == 0; };
    // Cumulative targets: 3, 6, 9, 12. Prefixes: 0, 4, 4, 4; deficits: 3, 2, 1, 0.
    const std::vector<size_t>   expected{0, 4, 4, 4};
    const std::vector<uint64_t> targets{3, 6, 9, 12};
    for (size_t i = 0; i < expected.size(); ++i) {
        const auto decision = controller.plan(6, multiples_of_four);
        EXPECT_EQ(decision.planned_host_blocks, expected[i]);
        EXPECT_LT(targets[i] - decision.cumulative_planned_host_blocks, 4u);
    }
    EXPECT_EQ(controller.snapshot().cumulative_planned_host_blocks, 12u);
}

TEST(MockMemoryHitControllerTest, CarriedDeficitNeverExceedsCurrentRequestCapacity) {
    MockMemoryHitController controller(1.0);
    EXPECT_EQ(controller.plan(10, [](size_t n) { return n == 0; }).planned_host_blocks, 0u);
    EXPECT_EQ(controller.plan(2, anyPrefix).planned_host_blocks, 2u);
    EXPECT_EQ(controller.plan(0, anyPrefix).planned_host_blocks, 0u);
    EXPECT_EQ(controller.snapshot().cumulative_eligible_blocks, 12u);
    EXPECT_EQ(controller.snapshot().cumulative_planned_host_blocks, 2u);
}

TEST(MockMemoryHitControllerTest, HandlesNonMonotonicPrefixValidity) {
    MockMemoryHitController controller(1.0);
    std::vector<size_t>     visited;
    const auto              decision = controller.plan(7, [&](size_t n) {
        visited.push_back(n);
        return n == 0 || n == 3 || n == 5;
    });
    EXPECT_EQ(decision.planned_host_blocks, 5u);
    EXPECT_EQ(visited, (std::vector<size_t>{7, 6, 5}));
}

TEST(MockMemoryHitControllerTest, RejectedOrThrowingValidationDoesNotCommitCounters) {
    MockMemoryHitController controller(0.5);
    EXPECT_THROW(controller.plan(4, [](size_t) { return false; }), std::invalid_argument);
    EXPECT_THROW(controller.plan(4, [](size_t) -> bool { throw std::runtime_error("validation failed"); }),
                 std::runtime_error);
    EXPECT_EQ(controller.snapshot().cumulative_eligible_blocks, 0u);
    EXPECT_EQ(controller.snapshot().cumulative_planned_host_blocks, 0u);
    EXPECT_EQ(controller.plan(4, anyPrefix).planned_host_blocks, 2u);
}

TEST(MockMemoryHitControllerTest, WideProductAndOverflowGuardPreserveCounters) {
    static_assert(sizeof(size_t) == sizeof(uint64_t), "Test requires 64-bit request sizes");
    const auto              max_blocks = std::numeric_limits<uint64_t>::max();
    MockMemoryHitController controller(0.5);
    EXPECT_EQ(controller.plan(max_blocks, anyPrefix).planned_host_blocks, 9'223'372'036'854'775'807ULL);
    EXPECT_THROW(controller.plan(1, anyPrefix), std::overflow_error);
    EXPECT_EQ(controller.snapshot().cumulative_eligible_blocks, max_blocks);
    EXPECT_EQ(controller.snapshot().cumulative_planned_host_blocks, 9'223'372'036'854'775'807ULL);
    EXPECT_EQ(controller.plan(0, anyPrefix).planned_host_blocks, 0u);
}

TEST(MockMemoryHitControllerTest, ConcurrentCallersMatchAValidSerializedOrder) {
    MockMemoryHitController            controller(0.5);
    std::atomic<bool>                  start{false};
    std::atomic<bool>                  stop_snapshot{false};
    std::thread                        observer([&] {
        while (!stop_snapshot.load()) {
            const auto state = controller.snapshot();
            EXPECT_EQ(state.cumulative_planned_host_blocks, state.cumulative_eligible_blocks / 2);
        }
    });
    constexpr size_t                   callers          = 8;
    constexpr size_t                   calls_per_thread = 64;
    std::vector<MockMemoryHitDecision> decisions(callers * calls_per_thread);
    std::vector<std::thread>           threads;
    for (size_t t = 0; t < callers; ++t) {
        threads.emplace_back([&, t] {
            while (!start.load()) {
                std::this_thread::yield();
            }
            for (size_t i = 0; i < calls_per_thread; ++i) {
                decisions[t * calls_per_thread + i] = controller.plan(t + 1, anyPrefix);
            }
        });
    }
    start = true;
    for (auto& thread : threads) {
        thread.join();
    }
    stop_snapshot = true;
    observer.join();
    std::sort(decisions.begin(), decisions.end(), [](const auto& a, const auto& b) {
        return a.cumulative_eligible_blocks < b.cumulative_eligible_blocks;
    });
    uint64_t previous_eligible = 0;
    uint64_t previous_planned  = 0;
    for (const auto& decision : decisions) {
        EXPECT_EQ(decision.cumulative_eligible_blocks - previous_eligible, decision.eligible_blocks);
        EXPECT_EQ(decision.cumulative_planned_host_blocks - previous_planned, decision.planned_host_blocks);
        EXPECT_EQ(decision.cumulative_planned_host_blocks, decision.cumulative_eligible_blocks / 2);
        EXPECT_LE(decision.planned_host_blocks, decision.eligible_blocks);
        previous_eligible = decision.cumulative_eligible_blocks;
        previous_planned  = decision.cumulative_planned_host_blocks;
    }
    // 64 * (1 + ... + 8) = 2304 eligible blocks, with exactly half planned.
    EXPECT_EQ(controller.snapshot().cumulative_eligible_blocks, 2304u);
    EXPECT_EQ(controller.snapshot().cumulative_planned_host_blocks, 1152u);
}

}  // namespace test
}  // namespace rtp_llm
