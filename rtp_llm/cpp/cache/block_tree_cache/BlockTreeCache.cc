#include <unordered_set>
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"

#include <algorithm>
#include <limits>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/diagnostic/FullPrefixInvariantScanner.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/MockMemoryCacheSource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

BlockTreeCache::BlockTreeCache(std::unique_ptr<BlockTree>                     tree,
                               BlockTreeCacheConfig                           config,
                               std::shared_ptr<StorageBackend>                storage_backend,
                               std::unique_ptr<BlockTransferDispatcher>       transfer_dispatcher,
                               std::unique_ptr<BlockTreeTaskPool>             task_pool,
                               std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter,
                               std::shared_ptr<MockMemoryCacheSource>         mock_memory_cache_source):
    config_(std::move(config)),
    tree_(std::move(tree)),
    storage_backend_(std::move(storage_backend)),
    mock_memory_cache_source_(std::move(mock_memory_cache_source)),
    transfer_dispatcher_(std::move(transfer_dispatcher)),
    task_pool_(std::move(task_pool)),
    metrics_reporter_(std::move(metrics_reporter)),
    evictor_(
        tree_.get(),
        config_.device_eviction_policy,
        config_.host_eviction_policy,
        config_.disk_eviction_policy,
        transfer_dispatcher_.get(),
        task_pool_.get(),
        *metrics_reporter_,
        mutex_,
        config_.host_cache_sync_timeout_ms,
        config_.disk_cache_sync_timeout_ms,
        config_.max_descriptors_per_transfer_batch,
        [this](Tier tier) { return config_.isTierEnabled(tier); },
        [this](bool check_watermark) { onWorkflowSettledLocked(check_watermark); }),
    loader_(
        tree_.get(),
        evictor_,
        transfer_dispatcher_.get(),
        task_pool_.get(),
        *metrics_reporter_,
        mutex_,
        config_.disk_cache_sync_timeout_ms,
        config_.host_cache_sync_timeout_ms,
        config_.enable_device_cache,
        storage_backend_,
        [this](bool check_watermark) { onWorkflowSettledLocked(check_watermark); },
        mock_memory_cache_source_),
    storer_(tree_.get(),
            evictor_,
            transfer_dispatcher_.get(),
            task_pool_.get(),
            *metrics_reporter_,
            mutex_,
            config_.host_cache_sync_timeout_ms,
            config_.disk_cache_sync_timeout_ms,
            storage_backend_,
            [this](bool check_watermark) { onWorkflowSettledLocked(check_watermark); }) {}

bool BlockTreeCache::init() {
    if (initialized_) {
        RTP_LLM_LOG_ERROR("cache is already initialized");
        return false;
    }
    if (!task_pool_->start()) {
        RTP_LLM_LOG_ERROR("failed to start task pool, size=%d", config_.task_pool_size);
        return false;
    }
    RTP_LLM_LOG_INFO("initialized with %zu group sets, %zu reusable topology groups, "
                     "pool_threads=%d, storage_backend=%s, "
                     "device=%s, host=%s, disk=%s, remote=%s",
                     tree_->groupSets().size(),
                     tree_->reusableGroupCount(),
                     config_.task_pool_size,
                     storage_backend_ ? "enabled" : "null",
                     config_.enable_device_cache ? "on" : "off",
                     config_.enable_host_cache ? "on" : "off",
                     config_.enable_disk_cache ? "on" : "off",
                     config_.enable_remote_cache ? "on" : "off");
    for (const GroupSetPtr& group_set : tree_->groupSets()) {
        RTP_LLM_LOG_INFO("  group_set[%zu] type=%s host_pool=%s disk_pool=%s",
                         group_set->groupSetId(),
                         cacheGroupTypeName(group_set->groupType()),
                         group_set->hostPool() ? "enabled" : "null",
                         group_set->diskPool() ? "enabled" : "null");
    }
    if (config_.full_prefix_scan_interval_ms > 0) {
        auto scanner =
            std::make_unique<FullPrefixInvariantScanner>(*tree_, mutex_, config_.full_prefix_scan_interval_ms);
        if (scanner->start()) {
            full_prefix_scanner_ = std::move(scanner);
            RTP_LLM_LOG_INFO("FULL prefix invariant scanner enabled, interval_ms=%ld",
                             config_.full_prefix_scan_interval_ms);
        }
    }
    initialized_ = true;
    if (config_.mock_memory_cache) {
        metrics_reporter_->updateMockMemorySource(tree_->groupSets());
    }
    return true;
}

BlockTreeCache::~BlockTreeCache() {
    full_prefix_scanner_.reset();
    RTP_LLM_LOG_INFO("destroying, closing load tickets...");
    shutdownLoads();
    if (storage_backend_) {
        storage_backend_->shutdown();
    }
    RTP_LLM_LOG_INFO("load tickets closed, stopping store admission and draining cache tasks...");
    {
        std::lock_guard<std::mutex> lock(mutex_);
        storer_.stopAdmissionLocked();
    }
    task_pool_->stopAdmission();
    transfer_dispatcher_->cancelPendingStagingTransfers();
    task_pool_->waitForIdle();
    // Task-body idleness does not imply asynchronous RPC/transfer completion.
    // Keep dispatcher and cache state alive until transfer callbacks return,
    // then drain the settlement tasks those callbacks submitted.
    transfer_dispatcher_->drainTransfers();
    task_pool_->waitForIdle();
    transfer_dispatcher_.reset();
    task_pool_.reset();
    RTP_LLM_LOG_INFO("destroyed");
}

void BlockTreeCache::shutdownLoads() {
    loader_.shutdown();
}

TransferResult BlockTreeCache::executeTransfer(TransferTask task) {
    auto context = transfer_dispatcher_->executePerRank(std::move(task));
    context->waitDone();
    auto result = context->result();
    if (!result.ok()) {
        RTP_LLM_LOG_WARNING("per-rank block transfer failed: %s", result.status.ToString().c_str());
    }
    return result;
}

BlockTreeMatchResult BlockTreeCache::match(const CacheKeysType&               cache_keys,
                                           const std::function<bool(size_t)>& valid_prefix) {
    BlockTreeMatchResult result;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        result = loader_.matchLocked(cache_keys, valid_prefix);
    }
    metrics_reporter_->reportCacheReuseTimeMetrics(result.reuse_time_metrics_snapshots);
    return result;
}

std::shared_ptr<LoadAsyncContext> BlockTreeCache::mockMemoryLoad(size_t logical_blocks) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        context = loader_.mockMemoryLoad(logical_blocks);
    if (context) {
        context->setTransferOutcomeCallback([reporter = metrics_reporter_, logical_blocks](ErrorInfo error) {
            reporter->recordMockMemoryH2D(logical_blocks, error.ok());
        });
    } else if (logical_blocks > 0) {
        metrics_reporter_->recordMockMemoryH2D(logical_blocks, false);
    }
    return context;
}

void BlockTreeCache::reportMockMemoryPlan(const MockMemoryHitSnapshot& snapshot) {
    metrics_reporter_->updateMockMemoryPlan(snapshot);
}

void BlockTreeCache::insert(const CacheKeysType&                              cache_keys,
                            const std::vector<std::vector<GroupSetResource>>& resources,
                            Tier                                              target_tier) {
    (void)insert(cache_keys, resources, target_tier, false);
}

size_t BlockTreeCache::insert(const CacheKeysType&                              cache_keys,
                              const std::vector<std::vector<GroupSetResource>>& resources,
                              Tier                                              target_tier,
                              bool                                              is_resident,
                              bool                                              mock_memory_cache) {
    size_t           resident_prefix_length = 0;
    StorageWriteTask storage_write;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        storage_write = storer_.storeLocked(
            cache_keys, resources, target_tier, is_resident, resident_prefix_length, mock_memory_cache);
    }
    if (storage_write) {
        storage_backend_->write(std::move(storage_write));
    }
    return resident_prefix_length;
}

int BlockTreeCache::evictForGroup(std::string_view group_tag, size_t num_blocks) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!config_.isTierEnabled(Tier::DEVICE)) {
        return 0;
    }
    const ReusableGroupLocation* location = tree_->reusableGroupLocation(group_tag);
    if (location == nullptr) {
        return 0;
    }
    const GroupSetPtr& group_set   = tree_->groupSets()[location->group_set_id];
    const auto&        device_pool = group_set->devicePools()[location->member_group_id];

    const size_t initial_free       = device_pool->freeBlocksNum();
    size_t       reclaimed          = 0;
    bool         eviction_triggered = false;
    while (reclaimed < num_blocks) {
        if (!evictor_.dropLocked(location->group_set_id, Tier::DEVICE, true)) {
            break;
        }
        eviction_triggered        = true;
        const size_t current_free = device_pool->freeBlocksNum();
        reclaimed                 = current_free > initial_free ? current_free - initial_free : 0;
    }
    if (eviction_triggered) {
        metrics_reporter_->reportEvictionTriggered(Tier::DEVICE, group_set->groupType(), /*force_drop=*/true);
    }
    metrics_reporter_->reportEvictionBlocks(Tier::DEVICE,
                                            group_set->groupType(),
                                            /*force_drop=*/true,
                                            num_blocks,
                                            std::min(reclaimed, num_blocks));
    RTP_LLM_LOG_DEBUG("tag=%.*s group_set[%zu] reclaimed %zu/%zu device blocks",
                      static_cast<int>(group_tag.size()),
                      group_tag.data(),
                      location->group_set_id,
                      reclaimed,
                      num_blocks);
    return static_cast<int>(reclaimed);
}

BlockIndicesType BlockTreeCache::matchedBlocksForGroup(std::string_view                      group_tag,
                                                       const std::vector<MultiNodeResource>& matched_resources) const {
    return loader_.matchedBlocksForGroup(group_tag, matched_resources);
}

CacheStats BlockTreeCache::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    CacheStats                  stats;
    stats.tree_node_count           = tree_->size();
    const CandidateStats candidates = evictor_.candidateStats();
    stats.device_heap_total_size    = candidates.device_candidates;
    stats.host_heap_total_size      = candidates.host_candidates;
    stats.disk_heap_total_size      = candidates.disk_candidates;
    return stats;
}

std::vector<BlockTreePoolMetricsSnapshot> BlockTreeCache::poolMetricsSnapshots() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return metrics_reporter_->collectPoolMetricsSnapshots(tree_->groupSets());
}

CachePressureSnapshot BlockTreeCache::pressureSnapshot() const {
    CachePressureSnapshot snapshot;
    snapshot.ownership = config_.pressure_ownership;
    if (config_.mock_memory_cache) {
        snapshot.mock_memory = metrics_reporter_->mockMemoryMetricsSnapshot();
    }
    // Lock order: cache -> individual pool / pending-release / ledger mutex.
    // Each subordinate lock is released before acquiring another. Copy callbacks
    // take only the ledger lock. No serialization or tree walk occurs here.
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& physical : tree_->pressurePools()) {
        const auto*               pool = physical.pool;
        CachePressurePoolSnapshot record;
        record.pool_id                    = physical.id;
        record.tier                       = physical.tier;
        record.pool_name                  = pool->poolName();
        record.block_size_bytes           = pool->blockSizeBytes();
        record.total_blocks               = pool->totalBlocksNum();
        record.free_blocks                = pool->freeBlocksNum();
        record.used_blocks                = record.total_blocks - record.free_blocks;
        record.available_blocks           = pool->availableBlocksNum();
        record.active_blocks              = pool->activeBlocksNum();
        record.block_cache_ref_blocks     = pool->referencedBlocksNum(BlockTreeRefType::CACHE);
        record.load_ref_blocks            = pool->referencedBlocksNum(BlockTreeRefType::LOAD);
        record.eviction_target_ref_blocks = pool->referencedBlocksNum(BlockTreeRefType::EVICTION);
        record.store_ref_blocks           = pool->referencedBlocksNum(BlockTreeRefType::STORE);
        record.mock_source_ref_blocks     = pool->referencedBlocksNum(BlockTreeRefType::MOCK_SOURCE);
        if (physical.tier == Tier::HOST) {
            const size_t stride          = static_cast<const HostBlockPool*>(pool)->strideBytes();
            record.mock_source_ref_bytes = record.mock_source_ref_blocks > std::numeric_limits<size_t>::max() / stride ?
                                               std::numeric_limits<size_t>::max() :
                                               record.mock_source_ref_blocks * stride;
        }
        if (physical.tier == Tier::DEVICE) {
            record.request_ref_blocks           = static_cast<const DeviceBlockPool*>(pool)->referencedBlocksNum();
            record.request_ref_blocks_available = true;
        }
        record.pending_release_blocks = evictor_.pendingReleaseBlocks(pool);
        record.published_host_blocks  = tree_->pressureState().pool(physical.id).published_host_blocks;
        snapshot.pools.push_back(std::move(record));
    }
    for (const auto& group : tree_->groupSets()) {
        CachePressureGroupSnapshot record;
        record.group_id      = group->groupSetId();
        record.group_type    = group->groupType();
        record.group_tags    = group->groupTags();
        record.block_tokens  = group->group(group->groupTags().front()).seqSizePerBlock();
        record.payload_bytes = group->payloadBytes();
        record.storage_bytes = group->storageBytes();
        record.crc_enabled   = group->crcEnabled();
        std::vector<const IBlockPool*> pools;
        for (const auto& pool : group->devicePools()) {
            if (pool) {
                record.device_pool_ids.push_back(tree_->pressurePoolId(pool.get()));
                pools.push_back(pool.get());
            }
        }
        if (group->hostPool()) {
            record.host_pool_id = tree_->pressurePoolId(group->hostPool().get());
            pools.push_back(group->hostPool().get());
        }
        if (group->diskPool()) {
            record.disk_pool_id = tree_->pressurePoolId(group->diskPool().get());
            pools.push_back(group->diskPool().get());
        }
        record.device_watermark        = config_.watermark_device;
        record.host_watermark          = config_.watermark_host;
        record.disk_watermark          = config_.watermark_disk;
        record.watermark_trigger_count = tree_->pressureState().watermarkTriggers(record.group_id);
        snapshot.groups.push_back(record);
        // Fixed key universe: topology pools x three operations x six directions.
        // Deduplicate member pools shared within a group as well as across groups.
        std::unordered_set<const IBlockPool*> visited;
        for (const auto* pool : pools) {
            if (!visited.insert(pool).second) {
                continue;
            }
            const auto& id        = tree_->pressurePoolId(pool);
            Tier        pool_tier = Tier::DEVICE;
            if (group->hostPool().get() == pool) {
                pool_tier = Tier::HOST;
            } else if (group->diskPool().get() == pool) {
                pool_tier = Tier::DISK;
            }
            for (auto operation :
                 {CacheTransferOperation::LOAD, CacheTransferOperation::EVICT, CacheTransferOperation::STORE}) {
                for (auto source : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
                    for (auto target : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
                        if (source == target || (pool_tier != source && pool_tier != target))
                            continue;
                        const auto evidence =
                            tree_->pressureState().transfer({record.group_id, id, operation, source, target});
                        if (!evidence.copied_blocks && !evidence.published_blocks && !evidence.failed_operations)
                            continue;
                        CachePressureTransferSnapshot transfer;
                        static_cast<TransferEvidence&>(transfer) = evidence;
                        transfer.group_id                        = record.group_id;
                        transfer.pool_id                         = id;
                        transfer.pool_tier                       = pool_tier;
                        transfer.operation                       = operation;
                        transfer.source_tier                     = source;
                        transfer.target_tier                     = target;
                        snapshot.transfers.push_back(std::move(transfer));
                    }
                }
            }
        }
    }
    return snapshot;
}

void BlockTreeCache::reportMetrics() const {
    std::vector<BlockTreeEvictableMetricsSnapshot> snapshots;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshots = metrics_reporter_->collectEvictableMetricsSnapshots(tree_->groupSets(), evictor_);
    }
    metrics_reporter_->reportEvictableCandidateCount(snapshots);
    metrics_reporter_->reportQueueBacklog(task_pool_->queueSizes(), "business");
    metrics_reporter_->reportQueueBacklog(transfer_dispatcher_->queueSizes(), "transfer");
    if (config_.mock_memory_cache) {
        metrics_reporter_->updateMockMemorySource(tree_->groupSets());
        metrics_reporter_->reportMockMemoryMetrics();
    }
}

void BlockTreeCache::setEventPublisher(KVCacheEventPublisherPtr        publisher,
                                       const std::vector<std::string>& required_group_tags) {
    std::lock_guard<std::mutex> lock(mutex_);
    tree_->setEventPublisher(std::move(publisher), required_group_tags);
}

KVCacheSnapshot BlockTreeCache::logicalCacheSnapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tree_->logicalCacheSnapshot();
}

BlockTreeKeySnapshot BlockTreeCache::getKeySnapshot(int64_t latest_version) const {
    std::lock_guard<std::mutex> lock(mutex_);
    BlockTreeKeySnapshot        snapshot;
    snapshot.version = tree_->keyVersion();
    if (latest_version != snapshot.version) {
        snapshot.keys = tree_->cacheKeys();
    }
    return snapshot;
}

bool BlockTreeCache::abortPendingLoad(const std::shared_ptr<AsyncContext>& context) {
    return loader_.abortPendingLoad(context);
}

void BlockTreeCache::onWorkflowSettledLocked(bool check_watermark) {
    if (check_watermark) {
        checkWatermark();
    }
}

void BlockTreeCache::checkWatermark() {
    for (Tier tier : {Tier::DISK, Tier::HOST, Tier::DEVICE}) {
        const auto watermark = config_.watermarkForTier(tier);
        if (!config_.isTierEnabled(tier) || !watermark.enabled()) {
            continue;
        }
        evictor_.scheduleWatermarkEvictionsLocked(tier, watermark);
    }
}

}  // namespace rtp_llm
