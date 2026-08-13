// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-FileCopyrightText: Copyright 2025 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <shared_mutex>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <ankerl/unordered_dense.h>

#include "common/common_types.h"
#include "common/thread_worker.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/shader_cache.h"
#include "video_core/spirv_cache.h"

namespace Core {
class System;
}

namespace Shader::IR {
struct Program;
}

namespace VideoCore {
class ShaderNotify;
}

namespace Vulkan {

struct ComputePipelineCacheKey {
    u64 unique_hash;
    u32 shared_memory_size;
    std::array<u32, 3> workgroup_size;

    size_t Hash() const noexcept;

    bool operator==(const ComputePipelineCacheKey& rhs) const noexcept;

    bool operator!=(const ComputePipelineCacheKey& rhs) const noexcept {
        return !operator==(rhs);
    }
};
static_assert(std::has_unique_object_representations_v<ComputePipelineCacheKey>);
static_assert(std::is_trivially_copyable_v<ComputePipelineCacheKey>);
static_assert(std::is_trivially_constructible_v<ComputePipelineCacheKey>);

} // namespace Vulkan

namespace std {

template <>
struct hash<Vulkan::ComputePipelineCacheKey> {
    size_t operator()(const Vulkan::ComputePipelineCacheKey& k) const noexcept {
        return k.Hash();
    }
};

} // namespace std

namespace Vulkan {

class ComputePipeline;
class DescriptorPool;
class Device;
class PipelineStatistics;
class RenderPassCache;
class Scheduler;

using VideoCommon::ShaderInfo;

struct ShaderPools {
    void ReleaseContents() {
        flow_block.ReleaseContents();
        block.ReleaseContents();
        inst.ReleaseContents();
    }

    Shader::ObjectPool<Shader::IR::Inst> inst{8192};
    Shader::ObjectPool<Shader::IR::Block> block{32};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_block{32};
};

class PipelineCache : public VideoCommon::ShaderCache {
public:
    explicit PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory_, const Device& device,
                           Scheduler& scheduler, DescriptorPool& descriptor_pool,
                           GuestDescriptorQueue& guest_descriptor_queue,
                           RenderPassCache& render_pass_cache, BufferCache& buffer_cache,
                           TextureCache& texture_cache, VideoCore::ShaderNotify& shader_notify_);
    ~PipelineCache();

    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipeline();

    [[nodiscard]] ComputePipeline* CurrentComputePipeline();

    void LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback);

private:
    void SubmitSpeculativeShader(u64 unique_hash, std::vector<u64> maxwell_code,
                               Shader::Stage stage, u32 local_memory_size,
                               u32 shared_memory_size, std::array<u32, 3> workgroup_size,
                               u32 start_address, u32 texture_bound,
                               Shader::ProgramHeader sph);
    void OnNewShaderSeen(VideoCommon::GenericEnvironment& env, u64 unique_hash) override;
    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipelineSlowPath();

    [[nodiscard]] GraphicsPipeline* BuiltPipeline(GraphicsPipeline* pipeline) const noexcept;

    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline();

    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline(
        ShaderPools& pools, const GraphicsPipelineCacheKey& key,
        std::span<Shader::Environment* const> envs, PipelineStatistics* statistics,
        bool build_in_parallel);

    std::unique_ptr<ComputePipeline> CreateComputePipeline(const ComputePipelineCacheKey& key,
                                                           const ShaderInfo* shader);

    std::unique_ptr<ComputePipeline> CreateComputePipeline(ShaderPools& pools,
                                                           const ComputePipelineCacheKey& key,
                                                           Shader::Environment& env,
                                                           PipelineStatistics* statistics,
                                                           bool build_in_parallel);

    void SerializeVulkanPipelineCache(const std::filesystem::path& filename,
                                      const vk::PipelineCache& pipeline_cache, u32 cache_version);

    vk::PipelineCache LoadVulkanPipelineCache(const std::filesystem::path& filename,
                                              u32 expected_cache_version);

    /// Evicts old unused pipelines to free memory when under pressure
    void EvictOldPipelines();

public:
    /// Public interface to evict old pipelines (for memory pressure handling)
    void TriggerPipelineEviction() {
        EvictOldPipelines();
    }

    const Device& device;
    Scheduler& scheduler;
    DescriptorPool& descriptor_pool;
    GuestDescriptorQueue& guest_descriptor_queue;
    RenderPassCache& render_pass_cache;
    BufferCache& buffer_cache;
    TextureCache& texture_cache;
    VideoCore::ShaderNotify& shader_notify;

    VideoCommon::SpirvCache spirv_cache;
    std::filesystem::path spirv_cache_filename;
    Common::ThreadWorker speculative_worker;
    Common::ThreadWorker serialization_thread;

    // Shader recompiler pools reused across speculative translations on the
    // speculative_worker thread.  Reusing rather than reallocating per shader
    // eliminates repeated large VirtualAlloc/VirtualFree calls that fragment the
    // address space and push Dynarmic JIT allocations outside the ±2 GB range
    // required for 32-bit RIP-relative addressing.
    // IMPORTANT: must only ever be accessed from speculative_worker's thread.
    ShaderPools spec_pools;
    bool use_asynchronous_shaders{};
    bool use_vulkan_pipeline_cache{};

    // Phase 4 narrow prototype's actual fix for the runaway-pipeline-creation freeze a real
    // build surfaced: ResolvePhase4PrototypeSpecValue() must NOT blindly read cbuf 2 offset
    // 192 for every fragment-shaded draw regardless of relevance -- for shaders that don't
    // actually have the marked descriptor, that memory is ordinary application data (often
    // changing every draw by design), and reinterpreting it as a texture handle produces an
    // unstable graphics_key that defeats pipeline caching entirely. Populated once per
    // fragment shader, in CreateGraphicsPipeline (where real Shader::Info is available, see
    // that function's doc comment at the population site), keyed by that shader's own
    // unique_hash (graphics_key.unique_hashes[5] -- ShaderType::Pixel's raw index, NOT 4;
    // see ResolvePhase4PrototypeSpecValue's doc comment for why those two numbers are both
    // real and both needed for different arrays). true only for the 12 real shaders this
    // prototype targets; false (not absent) for every other fragment shader once seen once,
    // so ResolvePhase4PrototypeSpecValue can skip the read entirely for known-irrelevant
    // shaders without re-deciding it every draw.
    //
    // ankerl::unordered_dense::map, not std::unordered_map: this is read once per draw call
    // for every fragment-shaded draw (via ResolvePhase4PrototypeSpecValue, called from
    // CurrentGraphicsPipeline, on the hot path this whole fix exists to keep cheap and stable)
    // -- exactly the read-heavy, write-rare, u64-keyed pattern SpirvCache's own entries_/
    // unique_hashes_/keys_by_hash_ (spirv_cache.h) already use unordered_dense for, in this
    // same codebase. u64 keys need no custom hasher, same as those.
    //
    // phase4_prototype_fragment_shader_table_mutex: a REAL, separate bug the very next real
    // test found -- CreateGraphicsPipeline (the write site) is called from workers.QueueWork()
    // for the boot-time bulk pipeline-loading path (confirmed by reading that call site
    // directly, not assumed), meaning multiple worker threads can write this table
    // concurrently during exactly the "compiling shaders" bulk-load phase, while
    // CurrentGraphicsPipelineSlowPath's synchronous path (the live draw-time miss case) and
    // ResolvePhase4PrototypeSpecValue's reads happen on the main/render thread at the same
    // time -- an unsynchronized concurrent read/write on a container with no built-in thread
    // safety, which is exactly the kind of bug that can corrupt internal hash-table state and
    // hang rather than crash cleanly. std::shared_mutex, not a plain mutex, mirroring
    // SpirvCache's own mutex_ (spirv_cache.h) exactly -- same shape of problem, hot-path reads
    // outnumbering rare writes by a huge margin, so std::shared_lock for reads and
    // std::unique_lock for writes (matching SpirvCache's own real usage, confirmed by reading
    // it, not assumed) is the right tool, not just a correct one.
    mutable std::shared_mutex phase4_prototype_fragment_shader_table_mutex;
    mutable ankerl::unordered_dense::map<u64, bool> phase4_prototype_fragment_shader_table;

    // Phase 4 narrow prototype's graphics_cache lookup-timing fix. Called from
    // CurrentGraphicsPipeline(), after RefreshStages()/state.Refresh() but before the
    // graphics_cache/Next() lookups that graphics_key feeds -- see this method's definition in
    // vk_pipeline_cache.cpp for the full reasoning and its real, flagged limitations (assumes
    // no secondary cbuf combine for this one hardcoded slot; gated by
    // phase4_prototype_fragment_shader_table above rather than running unconditionally, after
    // an earlier version of this method did exactly that and froze real gameplay).
    bool ResolvePhase4PrototypeSpecValue() const;

    GraphicsPipelineCacheKey graphics_key{};
    GraphicsPipeline* current_pipeline{};

    std::unordered_map<ComputePipelineCacheKey, std::unique_ptr<ComputePipeline>> compute_cache;
    std::unordered_map<GraphicsPipelineCacheKey, std::unique_ptr<GraphicsPipeline>> graphics_cache;

    std::unordered_map<const GraphicsPipeline*, u64> graphics_pipeline_last_use;
    std::unordered_map<const ComputePipeline*, u64> compute_pipeline_last_use;

    u64 last_memory_pressure_frame{0};
    static constexpr u64 MEMORY_PRESSURE_COOLDOWN = 300;

    ShaderPools main_pools;

    Shader::Profile profile;
    Shader::HostTranslateInfo host_info;

    std::filesystem::path pipeline_cache_filename;

    std::filesystem::path vulkan_pipeline_cache_filename;
    vk::PipelineCache vulkan_pipeline_cache;

    Common::ThreadWorker workers;
    DynamicFeatures dynamic_features;

};

} // namespace Vulkan
