#include "rtp_llm/cpp/cache/block_tree_cache/store/StoreTaskRunner.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"

#include <exception>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageState.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

void StoreTaskRunner::Task::recordMockMemorySubmission(BlockTreeCacheMetricsReporter& metrics_reporter) {
    if (!mock_memory_cache || target_tier != Tier::HOST) {
        return;
    }
    std::lock_guard<std::mutex> lock(mock_memory_outcome_mutex);
    if (!mock_memory_submission_recorded) {
        metrics_reporter.recordMockMemoryD2HSubmitted(descriptors().size());
        mock_memory_submission_recorded = true;
    }
}

void StoreTaskRunner::Task::recordMockMemoryTransferOutcome(BlockTreeCacheMetricsReporter& metrics_reporter,
                                                            bool                           success) {
    if (!mock_memory_cache || target_tier != Tier::HOST) {
        return;
    }
    // A callback proves acceptance, even when an already-expired timeout runs
    // synchronously before submit() returns. Publish submission before outcome.
    recordMockMemorySubmission(metrics_reporter);
    std::lock_guard<std::mutex> lock(mock_memory_outcome_mutex);
    if (!mock_memory_outcome_recorded) {
        metrics_reporter.recordMockMemoryD2HFinished(descriptors().size(), success);
        mock_memory_outcome_recorded = true;
    }
}

StoreTaskRunner::StoreTaskRunner(const std::vector<GroupSetPtr>& group_sets, BlockTree* pressure_tree):
    pressure_tree_(pressure_tree), group_sets_(group_sets) {}

bool StoreTaskRunner::prepareTask(Task& task, const std::vector<std::vector<GroupSetResource>>& resources) {
    for (size_t key_index = 0; key_index < task.cache_keys.size(); ++key_index) {
        for (size_t group_set_id = 0; group_set_id < group_sets_.size(); ++group_set_id) {
            const GroupSetResource& source = resources[key_index][group_set_id];
            if (source.device_blocks.empty()) {
                continue;
            }
            const GroupSetPtr& group_set    = group_sets_[group_set_id];
            const BlockIdxType target_block = group_set->allocateSingleBlock(task.target_tier, BlockTreeRefType::STORE);
            if (isNullBlockIdx(target_block)) {
                RTP_LLM_LOG_WARNING(
                    "store aborted: %s pool exhausted for group_set[%zu]", tierName(task.target_tier), group_set_id);
                return false;
            }
            group_set->referenceBlocks({group_set_id, Tier::DEVICE, {{nullptr, source.device_blocks}}},
                                       BlockTreeRefType::STORE);
            TransferDescriptor descriptor =
                task.target_tier == Tier::HOST ?
                    TransferDescriptor::deviceToHost(group_set_id, source.device_blocks, target_block) :
                    TransferDescriptor::deviceToDisk(group_set_id, source.device_blocks, target_block);
            descriptor.path_index = key_index;
            task.transfer_task.addDescriptor(std::move(descriptor));
        }
    }
    return true;
}

void StoreTaskRunner::runTransfer(TaskPtr                        task,
                                  const BlockTransferDispatcher& transfer_dispatcher,
                                  BlockTreeCacheMetricsReporter& metrics_reporter,
                                  TransferDoneCallback           callback) {
    try {
        task->phase = Task::Phase::TRANSFERRING;
        const int64_t transfer_begin =
            metrics_reporter.reportTransferStarted(CacheTransferOperation::STORE, Tier::DEVICE, task->target_tier);

        auto stage_state = std::make_shared<TransferStageState>(
            [this, task, &metrics_reporter, transfer_begin, callback](TransferResult result) mutable {
                task->recordMockMemoryTransferOutcome(metrics_reporter, result.ok());
                try {
                    if (pressure_tree_) {
                        pressure_tree_->recordPressureCopy(CacheTransferOperation::STORE, task->descriptors(), result);
                    }
                    static const std::vector<TransferDescriptor> empty_descriptors;
                    metrics_reporter.reportTransferFinished(CacheTransferOperation::STORE,
                                                            Tier::DEVICE,
                                                            task->target_tier,
                                                            task->descriptors().size(),
                                                            transfer_begin,
                                                            result.ok(),
                                                            result.ok() ? task->descriptors() : empty_descriptors,
                                                            group_sets_);
                    metrics_reporter.maybeReportCopyError(Tier::DEVICE, task->target_tier, result);
                    task->phase = Task::Phase::FINISHED;
                    callback(std::move(result.status));
                } catch (const std::exception& exception) {
                    task->phase = Task::Phase::FINISHED;
                    callback(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, exception.what()));
                } catch (...) {
                    task->phase = Task::Phase::FINISHED;
                    callback(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown store completion exception"));
                }
            },
            task->descriptors().size());

        const auto submit_batch = [&](const std::vector<TransferDescriptor>& descriptors, std::vector<size_t> indices) {
            stage_state->addBatch();
            transfer_dispatcher.runTransfer(task->transfer_task.subtask(descriptors),
                                            [stage_state, indices = std::move(indices)](TransferResult result) {
                                                stage_state->completeBatch(std::move(result), indices);
                                            });
        };
        if (task->target_tier == Tier::DISK) {
            for (size_t i = 0; i < task->descriptors().size(); ++i) {
                submit_batch({task->descriptors()[i]}, {i});
            }
        } else if (!task->descriptors().empty()) {
            std::vector<size_t> indices;
            for (size_t i = 0; i < task->descriptors().size(); ++i) {
                indices.push_back(i);
            }
            submit_batch(task->descriptors(), std::move(indices));
        }
        stage_state->finishSubmitting();
    } catch (const std::exception& error) {
        task->phase = Task::Phase::FINISHED;
        callback(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what()));
    } catch (...) {
        task->phase = Task::Phase::FINISHED;
        callback(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown store submission exception"));
    }
}

void StoreTaskRunner::releaseTaskResources(const Task& task) {
    for (const TransferDescriptor& descriptor : task.descriptors()) {
        const GroupSetPtr& group_set = group_sets_[descriptor.group_set_id];
        group_set->releaseSingleBlock(
            task.target_tier, descriptor.singleBlockAt(task.target_tier), BlockTreeRefType::STORE);
        group_set->unreferenceBlocks(
            MultiNodeResource{descriptor.group_set_id, Tier::DEVICE, {{nullptr, descriptor.source_blocks}}},
            BlockTreeRefType::STORE);
    }
}

}  // namespace rtp_llm
