#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/observability/ExecutionRecorder.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include <fstream>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace rtp_llm {
void NormalEngine::runReplay(const std::string& path) {
    const auto    output_path = path + ".rank" + std::to_string(parallelism_config.world_rank) + ".result.jsonl";
    std::ofstream output;
    try {
        output.exceptions(std::ios::badbit | std::ios::failbit);
        output.open(output_path);
        auto*       executor = dynamic_cast<NormalExecutor*>(executor_.get());
        const auto& config   = resource_context_.cache_manager->cacheConfig();
        output << "{\"event\":\"runtime_config\",\"world_rank\":" << parallelism_config.world_rank
               << ",\"tp_size\":" << parallelism_config.tp_size << ",\"dp_size\":" << parallelism_config.dp_size
               << ",\"model_config\":" << ExecutionRecorder::quote(model_config_.to_string())
               << ",\"cache_config\":" << ExecutionRecorder::quote(config.debugString()) << "}\n";
        if (!executor || isMTPEagle() || model_config_.mm_model_config.is_multimodal || config.groupNums() != 1
            || config.linear_group_num || config.swa_group_num || config.use_opaque_kv_cache_store
            || config.kv_scale_stride_bytes
            || (config.dtype != DataType::TYPE_FP16 && config.dtype != DataType::TYPE_BF16)
            || pd_sep_config.role_type != RoleType::PDFUSION || parallelism_config.dp_size != 1
            || ffn_disaggregate_config.enable_ffn_disaggregate) {
            throw std::runtime_error(
                "unsupported replay: requires DP=1, ordinary fused text model and one unquantized full KV group");
        }
        std::ifstream plan(path);
        std::string   magic;
        int           warmup = 0, repeat = 0, count = 0;
        if (!(plan >> magic >> warmup >> repeat >> count) || magic != "RTP_BATCH_REPLAY_V1" || warmup < 0 || repeat <= 0
            || count <= 0 || count > 10000 || repeat > 10000 || warmup > 10000) {
            throw std::runtime_error("invalid replay plan header");
        }
        auto context                = resource_context_;
        context.reuse_cache         = false;
        context.enable_memory_cache = false;
        context.enable_remote_cache = false;
        context.system_prompt.reset();
        for (int b = 0; b < count; ++b) {
            int64_t execution;
            int     batch;
            if (!(plan >> execution >> batch) || execution <= 0 || batch <= 0 || batch > 65536)
                throw std::runtime_error("invalid replay batch header");
            std::list<GenerateStreamPtr> streams;
            bool                         prefill_seen = false;
            for (int i = 0; i < batch; ++i) {
                int     phase;
                int64_t q, kv, prompt;
                if (!(plan >> phase >> q >> kv >> prompt) || (phase != 0 && phase != 1) || q <= 0 || kv < 0
                    || prompt <= 0 || prompt > model_config_.max_seq_len || q > model_config_.max_seq_len
                    || kv >= model_config_.max_seq_len - q || (phase == 0 && (q != 1 || prefill_seen)))
                    throw std::runtime_error("invalid or unsupported replay request");
                prefill_seen      = prefill_seen || phase == 1;
                auto input        = std::make_shared<GenerateInput>();
                input->request_id = i + 1;
                // Stable legal token IDs; this reproduces shape, not MoE routes.
                input->input_ids       = torch::arange(kv + q, torch::kInt32).remainder(model_config_.vocab_size);
                input->generate_config = std::make_shared<GenerateConfig>();
                input->generate_config->max_new_tokens = 1;
                auto stream =
                    std::make_shared<NormalGenerateStream>(input, model_config_, runtime_config, context, nullptr);
                const auto status = stream->initKVBlock();
                if (!status.ok())
                    throw std::runtime_error(status.ToString());
                stream->setReuseLength(phase == 1 ? kv : 0);
                stream->setIsContextStream(phase == 1);
                // CompleteTokenIds owns the q+KV sequence; decode input_lengths
                // separately carries the original prompt length, as in serving.
                input->input_ids = torch::zeros({prompt}, torch::kInt32);
                streams.push_back(stream);
            }
            auto reset_cache = [&] {
                for (const auto& stream : streams) {
                    for (const auto block : stream->kvCache().blocks(0)) {
                        for (int layer = 0; layer < model_config_.num_layers; ++layer) {
                            for (const auto& buffer : context.cache_manager->convertIndexToBuffer(block, layer)) {
                                if (!buffer.addr || !buffer.size_bytes)
                                    continue;
                                if (!buffer.is_cuda)
                                    throw std::runtime_error("CPU cache replay unsupported");
                                auto bytes =
                                    torch::from_blob(buffer.addr,
                                                     {static_cast<int64_t>(buffer.size_bytes)},
                                                     torch::TensorOptions(torch::kUInt8)
                                                         .device(torch::Device(torch::kCUDA, buffer.device_index)));
                                bytes.zero_();
                            }
                        }
                    }
                }
                cudaSyncAndCheck();
            };
            for (int r = 0; r < warmup + repeat; ++r) {
                reset_cache();
                const auto start = ExecutionRecorder::monotonicNs();
                executor->replayBatch(streams, execution);
                const auto elapsed = ExecutionRecorder::monotonicNs() - start;
                if (r >= warmup) {
                    output
                        << "{\"execution_id\":" << execution << ",\"world_rank\":" << parallelism_config.world_rank
                        << ",\"repeat\":" << r - warmup << ",\"elapsed_ns\":" << elapsed
                        << ",\"measurement\":\"synchronized_forward_with_input_preparation\",\"route_controlled\":false}\n";
                    output.flush();
                }
            }
            const auto   dir = std::filesystem::path(path).parent_path().string();
            TorchProfile profile("replay_" + std::to_string(execution) + "_wr"
                                     + std::to_string(parallelism_config.world_rank) + "_",
                                 dir);
            reset_cache();
            profile.start();
            executor->replayBatch(streams, execution);
            profile.stop();
        }
        output << "{\"status\":\"complete\"}\n";
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("batch replay failed: %s", error.what());
        try {
            output << "{\"status\":\"error\",\"reason\":" << ExecutionRecorder::quote(error.what()) << "}\n";
        } catch (...) {
            RTP_LLM_LOG_ERROR("could not write replay failure to %s", output_path.c_str());
        }
    }
}
}  // namespace rtp_llm
