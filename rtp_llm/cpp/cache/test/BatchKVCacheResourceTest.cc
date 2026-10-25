#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"

namespace rtp_llm {
namespace test {

TEST(BatchKVCacheResourcePlanTest, EmptyRetryReusesTheStoredRequestPlan) {
    BatchKVCacheResource resource;
    resource.resetBatchSize(1);
    EXPECT_FALSE(resource.mockMemoryCachePlan().has_value());
    resource.setMockMemoryCachePlan(MockMemoryCachePlan{9, 4});
    // A retry after failed allocation still has no blocks and must retain its decision.
    EXPECT_EQ(resource.curBlocksNum(), 0);
    ASSERT_TRUE(resource.mockMemoryCachePlan().has_value());
    EXPECT_EQ(resource.mockMemoryCachePlan()->eligible_blocks, 9u);
    EXPECT_EQ(resource.mockMemoryCachePlan()->planned_host_blocks, 4u);
}

TEST(BatchKVCacheResourcePlanTest, SwapMovesPlanWithBatchAndCacheKeyState) {
    BatchKVCacheResource first;
    first.resetBatchSize(2);
    first.pushBackCacheKey(0, 42);
    first.markCacheKeysInitialized();
    first.setMockMemoryCachePlan({9, 4});
    BatchKVCacheResource second;
    second.resetBatchSize(1);
    second.setMockMemoryCachePlan({6, 3});
    first.swap(second);
    EXPECT_EQ(first.batchSize(), 1);
    EXPECT_EQ(first.mockMemoryCachePlan()->eligible_blocks, 6u);
    EXPECT_EQ(first.mockMemoryCachePlan()->planned_host_blocks, 3u);
    EXPECT_FALSE(first.cacheKeysInitialized());
    EXPECT_EQ(second.batchSize(), 2);
    EXPECT_EQ(second.cacheKeys(0), (CacheKeysType{42}));
    EXPECT_TRUE(second.cacheKeysInitialized());
    EXPECT_EQ(second.mockMemoryCachePlan()->eligible_blocks, 9u);
    EXPECT_EQ(second.mockMemoryCachePlan()->planned_host_blocks, 4u);
    BatchKVCacheResource empty;
    second.swap(empty);
    EXPECT_FALSE(second.mockMemoryCachePlan().has_value());
    EXPECT_EQ(empty.mockMemoryCachePlan()->planned_host_blocks, 4u);
}

TEST(BatchKVCacheResourcePlanTest, ResetForNewRequestClearsPlanAndReturnsOldResources) {
    BatchKVCacheResource resource;
    resource.resetBatchSize(2);
    resource.pushBackCacheKey(0, 42);
    resource.markCacheKeysInitialized();
    resource.setMockMemoryCachePlan({9, 4});
    std::vector<KVCacheResource> old_resources;
    resource.resetAndReturnOldResources(1, old_resources);
    ASSERT_EQ(old_resources.size(), 2u);
    EXPECT_EQ(old_resources[0].cacheKeys(), (CacheKeysType{42}));
    EXPECT_EQ(resource.batchSize(), 1);
    EXPECT_FALSE(resource.cacheKeysInitialized());
    EXPECT_FALSE(resource.mockMemoryCachePlan().has_value());
}

TEST(BatchKVCacheResourcePlanTest, BatchResizeKeepsOneRequestDecisionUntilExplicitClear) {
    BatchKVCacheResource resource;
    resource.resetBatchSize(1);
    resource.setMockMemoryCachePlan({9, 4});
    for (size_t batch_size : {4, 2, 0, 1}) {
        resource.resetBatchSize(batch_size);
        EXPECT_EQ(resource.batchSize(), batch_size);
        ASSERT_TRUE(resource.mockMemoryCachePlan().has_value());
        EXPECT_EQ(resource.mockMemoryCachePlan()->eligible_blocks, 9u);
        EXPECT_EQ(resource.mockMemoryCachePlan()->planned_host_blocks, 4u);
    }
    resource.clearMockMemoryCachePlan();
    EXPECT_FALSE(resource.mockMemoryCachePlan().has_value());
    resource.setMockMemoryCachePlan({2, 0});
    ASSERT_TRUE(resource.mockMemoryCachePlan().has_value());
    EXPECT_EQ(resource.mockMemoryCachePlan()->eligible_blocks, 2u);
    EXPECT_EQ(resource.mockMemoryCachePlan()->planned_host_blocks, 0u);
}

}  // namespace test
}  // namespace rtp_llm
