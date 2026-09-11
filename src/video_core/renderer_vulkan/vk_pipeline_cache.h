// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-FileCopyrightText: Copyright 2025 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
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
#include "shader_recompiler/runtime_info.h"
#include "shader_recompiler/varying_state.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/precache_compiler_target.h"
#include "video_core/precache_cfg_artifact.h"
#include "video_core/precache_frontend_artifact.h"
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

    // Complete pending cache serialization and persist exact SPIR-V entries.
    // Safe to call while rendering is still alive; SpirvCache snapshots under
    // its own lock and writes outside it.
    void FlushSpirvCache();

private:
    friend class GraphicsPipeline;
    friend class ComputePipeline;

    // First template-cache layer: immutable, cbuf-independent CFG snapshots.
    // start_address is part of identity because Flow::Location is guest-address
    // based; relocating identical code must rebuild/rebase rather than reuse a
    // graph with stale branch locations.
    struct CfgTemplateKey {
        u64 unique_hash{};
        u32 start_address{};
        bool exits_to_dispatcher{};
        bool operator==(const CfgTemplateKey&) const = default;
    };
    struct CfgTemplateKeyHash {
        size_t operator()(const CfgTemplateKey& key) const noexcept {
            size_t hash{static_cast<size_t>(key.unique_hash)};
            hash ^= static_cast<size_t>(key.start_address) + 0x9e3779b9U + (hash << 6) +
                    (hash >> 2);
            return hash ^ static_cast<size_t>(key.exits_to_dispatcher);
        }
    };
    mutable std::shared_mutex cfg_templates_mutex;
    ankerl::unordered_dense::map<CfgTemplateKey, Shader::Maxwell::Flow::CFG::Template,
                                 CfgTemplateKeyHash>
        cfg_templates;
    mutable std::atomic<u64> cfg_template_hits{};
    mutable std::atomic<u64> cfg_template_inserts{};
    mutable std::atomic<u64> cfg_template_shadow_verified{};
    mutable std::atomic<u64> cfg_template_shadow_rejected{};
    mutable std::atomic<u64> cfg_template_cached_verified{};
    mutable std::atomic<u64> cfg_template_cached_rejected{};
    // Per-session claims keep expensive module shadow work bounded to one
    // fresh finalization per immutable CFG identity. This set is deliberately
    // not persisted: each executable revision must re-prove its own lowering.
    std::unordered_set<CfgTemplateKey, CfgTemplateKeyHash> cfg_template_module_claims;
    // A graph can round-trip structurally yet fail the full lowering/module
    // comparison. Quarantine that identity for this session so a later draw
    // cannot reinsert the same unproven persisted form after its one bounded
    // module-validation claim has already been consumed.
    std::unordered_set<CfgTemplateKey, CfgTemplateKeyHash> cfg_template_module_rejected_keys;
    mutable std::atomic<u64> cfg_template_module_verified{};
    mutable std::atomic<u64> cfg_template_module_rejected{};
    mutable std::atomic<u64> cfg_template_module_us{};
    mutable std::atomic<u64> cfg_template_module_graphics_verified{};
    mutable std::atomic<u64> cfg_template_module_compute_verified{};
    mutable std::atomic<u64> cfg_template_module_merged_vertex_verified{};

    [[nodiscard]] std::optional<Shader::Maxwell::Flow::CFG::Template> LookupCfgTemplate(
        CfgTemplateKey key, bool count_as_reuse = true) const;
    void InsertCfgTemplate(CfgTemplateKey key, Shader::Maxwell::Flow::CFG::Template source);
    void EraseCfgTemplate(CfgTemplateKey key);
    void RejectCfgTemplateModule(CfgTemplateKey key);
    void ValidateAndPersistCfgTemplate(CfgTemplateKey key, Shader::Environment& env,
                                       const Shader::Maxwell::Flow::CFG& cfg);
    [[nodiscard]] bool ClaimCfgTemplateModuleValidation(
        CfgTemplateKey key, const VideoCommon::PrecacheCfgArtifactKey* scanner_artifact);
    void ValidateCfgTemplateGraphicsModule(
        CfgTemplateKey key, Shader::Environment& env,
        const Shader::Maxwell::Flow::CFG::Template& source,
        u32 cfg_request_address,
        const Shader::RuntimeInfo& runtime_info, const Shader::IR::Program& expected_program,
        const std::vector<u32>& expected_spirv,
        const Shader::Backend::Bindings& starting_binding,
        const Shader::Backend::Bindings& expected_end_binding,
        const VideoCommon::PrecacheCfgArtifactKey* scanner_artifact);
    void ValidateCfgTemplateComputeModule(CfgTemplateKey key, Shader::Environment& env,
                                          const Shader::Maxwell::Flow::CFG::Template& source,
                                          u32 cfg_request_address,
                                          const Shader::IR::Program& expected_program,
                                          const std::vector<u32>& expected_spirv,
                                          const VideoCommon::PrecacheCfgArtifactKey* scanner_artifact);
    void ValidateCfgTemplateMergedVertexModule(
        CfgTemplateKey key, Shader::Environment& vertex_a_env,
        const Shader::Maxwell::Flow::CFG::Template& vertex_a_source,
        Shader::Environment& vertex_b_env,
        const Shader::Maxwell::Flow::CFG::Template& vertex_b_source,
        const Shader::RuntimeInfo& runtime_info, const Shader::IR::Program& expected_program,
        const std::vector<u32>& expected_spirv,
        const Shader::Backend::Bindings& starting_binding,
        const Shader::Backend::Bindings& expected_end_binding,
        const VideoCommon::PrecacheCfgArtifactKey* vertex_a_artifact,
        const VideoCommon::PrecacheCfgArtifactKey* vertex_b_artifact);
    void LoadCfgTemplates(const std::filesystem::path& path);
    void SaveCfgTemplates(const std::filesystem::path& path) const;

    struct PrecacheCfgReplay {
        Shader::Maxwell::Flow::CFG::Template cfg;
        std::vector<Shader::Maxwell::PredecodedInstruction> decoded_instructions;
        VideoCommon::PrecacheCfgArtifactKey key;
    };

    void LoadPrecacheCfgArtifactCache(const std::filesystem::path& path);
    void ShadowValidatePrecacheCfgArtifact(
        u64 program_identity, Shader::Stage stage, bool exits_to_dispatcher,
        const Shader::Maxwell::Flow::CFG& fresh_cfg) const;
    [[nodiscard]] std::optional<PrecacheCfgReplay>
    LookupPrecacheCfgArtifact(u64 program_identity, Shader::Stage stage, u32 cfg_request_address,
                              bool exits_to_dispatcher) const;
    void QuarantinePrecacheCfgArtifact(const VideoCommon::PrecacheCfgArtifactKey& key) const;
    mutable std::shared_mutex precache_cfg_artifacts_mutex;
    std::unordered_map<VideoCommon::PrecacheCfgArtifactKey,
                       VideoCommon::PrecacheCfgArtifact,
        VideoCommon::PrecacheCfgArtifactKeyHash>
        precache_cfg_artifacts;
    mutable std::unordered_set<VideoCommon::PrecacheCfgArtifactKey,
                               VideoCommon::PrecacheCfgArtifactKeyHash>
        precache_cfg_artifact_quarantine;
    std::unordered_set<VideoCommon::PrecacheCfgArtifactKey,
                       VideoCommon::PrecacheCfgArtifactKeyHash>
        precache_cfg_artifact_module_claimed_keys;
    mutable std::atomic<u64> precache_cfg_artifact_loaded{};
    mutable std::atomic<u64> precache_cfg_artifact_probes{};
    mutable std::array<std::atomic<u64>, 7> precache_cfg_artifact_loaded_stages{};
    mutable std::array<std::atomic<u64>, 7> precache_cfg_artifact_probe_stages{};
    mutable std::atomic<u64> precache_cfg_artifact_lookups{};
    mutable std::atomic<u64> precache_cfg_artifact_verified{};
    mutable std::atomic<u64> precache_cfg_artifact_rejected{};
    mutable std::atomic<u64> precache_cfg_artifact_module_sources{};
    mutable std::atomic<u64> precache_cfg_artifact_module_claims{};
    mutable std::atomic<u64> precache_cfg_artifact_module_verified{};
    mutable std::atomic<u64> precache_cfg_artifact_module_rejected{};
    mutable std::atomic<u64> precache_cfg_artifact_replay_attempts{};
    mutable std::atomic<u64> precache_cfg_artifact_replay_hits{};
    mutable std::atomic<u64> precache_cfg_artifact_replay_fallbacks{};
    mutable std::atomic<u64> precache_decoded_artifact_hits{};
    mutable std::atomic<u64> precache_decoded_artifact_instructions{};
    mutable std::atomic<u64> precache_cfg_artifact_replay_cfg_us{};
    mutable std::atomic<u64> precache_cfg_artifact_fresh_cfg_us{};
    mutable std::atomic<u64> precache_cfg_artifact_validation_cfg_us{};
    mutable std::atomic<u64> precache_cfg_artifact_quarantined{};

    void LoadPrecacheFrontendArtifactCache(const std::filesystem::path& path);
    [[nodiscard]] std::optional<VideoCommon::PrecacheFrontendArtifact>
    LookupPrecacheFrontendArtifact(const VideoCommon::PrecacheFrontendArtifactKey& key) const;
    void QuarantinePrecacheFrontendArtifact(
        const VideoCommon::PrecacheFrontendArtifactKey& key) const;
    mutable std::shared_mutex precache_frontend_artifacts_mutex;
    std::unordered_map<VideoCommon::PrecacheFrontendArtifactKey,
                       VideoCommon::PrecacheFrontendArtifact,
                       VideoCommon::PrecacheFrontendArtifactKeyHash>
        precache_frontend_artifacts;
    mutable std::unordered_set<VideoCommon::PrecacheFrontendArtifactKey,
                               VideoCommon::PrecacheFrontendArtifactKeyHash>
        precache_frontend_artifact_quarantine;
    mutable std::unordered_set<VideoCommon::PrecacheFrontendArtifactKey,
                               VideoCommon::PrecacheFrontendArtifactKeyHash>
        precache_frontend_artifact_shadow_validated;
    mutable std::atomic<u64> precache_frontend_artifact_loaded{};
    mutable std::atomic<u64> precache_frontend_artifact_lookups{};
    mutable std::atomic<u64> precache_frontend_artifact_hits{};
    mutable std::atomic<u64> precache_frontend_artifact_restored{};
    mutable std::atomic<u64> precache_frontend_artifact_fallbacks{};
    mutable std::atomic<u64> precache_frontend_artifact_quarantined{};
    mutable std::atomic<u64> precache_frontend_pipeline_candidates{};
    mutable std::atomic<u64> precache_frontend_nonlive_skipped{};
    mutable std::atomic<u64> precache_frontend_pipeline_full_hits{};
    mutable std::atomic<u64> precache_frontend_pipeline_restore_failures{};
    mutable std::atomic<u64> precache_frontend_artifact_restore_us{};
    mutable std::atomic<u64> precache_frontend_artifact_shadow_verified{};
    mutable std::atomic<u64> precache_frontend_artifact_shadow_rejected{};
    mutable std::atomic<u64> precache_frontend_artifact_shadow_us{};

    // Value-only state from the latest real translation for each shader.
    struct RealStageStoresSnapshot {
        Shader::VaryingState stores{};
        std::map<Shader::IR::Attribute, Shader::IR::Attribute> legacy_stores_mapping{};
        Shader::VaryingState passthrough{};
        bool is_geometry_passthrough{};
        Shader::Backend::Bindings end_binding{};
    };
    mutable std::shared_mutex real_stage_stores_mutex;
    mutable ankerl::unordered_dense::map<u64, RealStageStoresSnapshot> real_stage_stores_by_hash;
    mutable std::atomic<u64> shader_cfg_us{};
    mutable std::atomic<u64> shader_template_us{};
    mutable std::atomic<u64> shader_finalize_us{};
    mutable std::atomic<u64> shader_emit_us{};
    mutable std::atomic<u64> shader_stage_count{};
    mutable std::atomic<u64> shader_exact_module_hits{};
    mutable std::atomic<u64> compute_prefrontend_disk_hits{};
    mutable std::atomic<u64> compute_prefrontend_live_candidates{};
    mutable std::atomic<u64> compute_prefrontend_live_hits{};
    mutable std::atomic<u64> graphics_pipeline_create_count{};
    mutable std::atomic<u64> graphics_pipeline_create_us{};
    mutable std::atomic<u64> graphics_pipeline_create_max_us{};
    mutable std::atomic<u64> compute_pipeline_create_count{};
    mutable std::atomic<u64> compute_pipeline_create_us{};
    mutable std::atomic<u64> compute_pipeline_create_max_us{};
    // Constructor time only measures enqueueing. These clocks cover actual worker start,
    // Vulkan driver pipeline creation, and scheduler-thread blocking waits.
    mutable std::atomic<u64> graphics_pipeline_queue_us{};
    mutable std::atomic<u64> graphics_pipeline_queue_max_us{};
    mutable std::atomic<u64> graphics_pipeline_driver_count{};
    mutable std::atomic<u64> graphics_pipeline_driver_us{};
    mutable std::atomic<u64> graphics_pipeline_driver_max_us{};
    mutable std::atomic<u64> graphics_pipeline_driver_boot_count{};
    mutable std::atomic<u64> graphics_pipeline_driver_boot_us{};
    mutable std::atomic<u64> graphics_pipeline_driver_boot_max_us{};
    mutable std::atomic<u64> graphics_pipeline_driver_live_count{};
    mutable std::atomic<u64> graphics_pipeline_driver_live_us{};
    mutable std::atomic<u64> graphics_pipeline_driver_live_max_us{};
    mutable std::atomic<u64> graphics_pipeline_wait_count{};
    mutable std::atomic<u64> graphics_pipeline_wait_us{};
    mutable std::atomic<u64> graphics_pipeline_wait_max_us{};
    mutable std::atomic<u64> graphics_pipeline_wait_boot_count{};
    mutable std::atomic<u64> graphics_pipeline_wait_boot_us{};
    mutable std::atomic<u64> graphics_pipeline_wait_boot_max_us{};
    mutable std::atomic<u64> graphics_pipeline_wait_live_count{};
    mutable std::atomic<u64> graphics_pipeline_wait_live_us{};
    mutable std::atomic<u64> graphics_pipeline_wait_live_max_us{};
    mutable std::atomic<u64> compute_pipeline_queue_us{};
    mutable std::atomic<u64> compute_pipeline_queue_max_us{};
    mutable std::atomic<u64> compute_pipeline_driver_count{};
    mutable std::atomic<u64> compute_pipeline_driver_us{};
    mutable std::atomic<u64> compute_pipeline_driver_max_us{};
    mutable std::atomic<u64> compute_pipeline_driver_boot_count{};
    mutable std::atomic<u64> compute_pipeline_driver_boot_us{};
    mutable std::atomic<u64> compute_pipeline_driver_boot_max_us{};
    mutable std::atomic<u64> compute_pipeline_driver_live_count{};
    mutable std::atomic<u64> compute_pipeline_driver_live_us{};
    mutable std::atomic<u64> compute_pipeline_driver_live_max_us{};
    mutable std::atomic<u64> compute_pipeline_wait_count{};
    mutable std::atomic<u64> compute_pipeline_wait_us{};
    mutable std::atomic<u64> compute_pipeline_wait_max_us{};
    mutable std::atomic<u64> compute_pipeline_wait_boot_count{};
    mutable std::atomic<u64> compute_pipeline_wait_boot_us{};
    mutable std::atomic<u64> compute_pipeline_wait_boot_max_us{};
    mutable std::atomic<u64> compute_pipeline_wait_live_count{};
    mutable std::atomic<u64> compute_pipeline_wait_live_us{};
    mutable std::atomic<u64> compute_pipeline_wait_live_max_us{};
    mutable std::mutex shader_compiler_stats_mutex;
    mutable std::chrono::steady_clock::time_point shader_compiler_stats_last_log{};

    // Returns a copied real predecessor snapshot, if present.
    std::optional<RealStageStoresSnapshot> ResolveRealStageStoresSnapshot(
        u64 previous_stage_unique_hash) const;

    // Separate frontend work from SPIR-V emission in timing reports.
    void RecordShaderCompilerWork(std::chrono::microseconds cfg_time,
                                  std::chrono::microseconds template_time,
                                  std::chrono::microseconds finalize_time,
                                  std::chrono::microseconds emit_time,
                                  bool exact_module_hit) const;
    void RecordPipelineCreateWork(bool compute, std::chrono::microseconds create_time) const;
    void RecordPipelineBuildQueueDelay(bool compute, std::chrono::microseconds delay) const;
    void RecordPipelineDriverCreate(bool compute, bool boot_preload,
                                    std::chrono::microseconds duration) const;
    void RecordPipelineBuildWait(bool compute, bool boot_preload,
                                 std::chrono::microseconds duration) const;

    void SubmitSpeculativeShader(u64 unique_hash, std::vector<u64> maxwell_code,
                               Shader::Stage stage, u32 local_memory_size,
                               u32 shared_memory_size, std::array<u32, 3> workgroup_size,
                               u32 start_address, u32 texture_bound,
                               Shader::ProgramHeader sph,
                               std::optional<RealStageStoresSnapshot> previous_stage_snapshot);
    void OnNewShaderSeen(VideoCommon::GenericEnvironment& env, u64 unique_hash,
                        u64 previous_stage_unique_hash) override;
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
    std::filesystem::path cfg_templates_filename;
    std::filesystem::path precache_cfg_artifacts_filename;
    std::filesystem::path precache_frontend_artifacts_filename;
    // Cross-session adaptive texture-slot learning.
    std::filesystem::path phase4_prototype_slots_filename;
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

    // Per-fragment-shader polymorphic-slot mask. Reads are hot; writes are rare.
    mutable std::shared_mutex phase4_prototype_fragment_shader_table_mutex;
    mutable ankerl::unordered_dense::map<u64, u32> phase4_prototype_fragment_shader_table;

    // Resolve known polymorphic texture slots before graphics-cache lookup.
    u64 ResolvePhase4PrototypeSpecValue() const;

    // ---- Runtime-variant diagnostic (nothing below reads these back or
    // changes caching/guessing behavior; see RecordPhase3RuntimeVariantDiagnostic()'s
    // definition in vk_pipeline_cache.cpp for the full rationale) ----
    //
    // Tracks, per graphics unique_hash, the distinct "core RuntimeInfo" values
    // (diag_base_runtime_hash — see its declaration in CreateGraphicsPipeline(), which
    // deliberately excludes the binding-offset fold, the same split
    // RuntimeCoreComponentNeverMatchedCount()/RuntimeBindingComponentNeverMatchedCount()
    // already rely on in spirv_cache.h) observed among REAL, non-speculative graphics
    // inserts whose narrowed cbuf_key == 0 — the population cbuf narrowing grew (a
    // stable 33-34% baseline to 41-46%, see SpirvCache::real_cbuf_zero_count_) and
    // exactly the population a speculative entry could ever hope to match, since
    // InsertSpeculative() always hardcodes cbuf_key=0. This answers the
    // actual open question empirically instead of by argument: low cardinality per hash
    // would mean a small scan-time multi-variant guess could plausibly enumerate real
    // states now that cbuf isn't blocking them; hitting the cap on most hashes would
    // mean the opposite — the same kind of structural ceiling the removed second
    // viewport-transform-state guess already hit once (see PreCacheShaders' comment in
    // citron/main.cpp for that experiment and why it was removed rather than refined),
    // just not yet re-measured with cbuf's blocking narrowed out of the way.
    //
    // Deliberately NOT folded into SpirvCache::Insert() (spirv_cache.h/.cpp) even
    // though that already has an established, very similar capped-per-hash pattern
    // (keys_by_hash_) — that function is shared with CreateComputePipeline(), which
    // passes its own workgroup_key through the exact same diag_base_runtime_hash
    // parameter slot (see that function's own declaration of the name); tracking
    // indiscriminately there would silently mix two unrelated quantities into one
    // histogram. Living here instead, populated only from CreateGraphicsPipeline(),
    // keeps it unambiguously graphics-only.
    //
    // Own dedicated shared_mutex rather than reusing phase4_prototype_fragment_shader_table_mutex
    // above: unrelated data, and this is written from exactly the same worker-thread
    // context that table's own doc comment already explains (CreateGraphicsPipeline runs
    // under workers.QueueWork() during boot-time bulk pipeline loading, concurrently with
    // the live draw-time path on the main/render thread), so it needs the same kind of
    // real synchronization, not a borrowed lock that would make this diagnostic's writes
    // block that table's unrelated reads or vice versa.
    mutable std::shared_mutex phase3_diag_runtime_variants_mutex;
    mutable ankerl::unordered_dense::map<u64, std::vector<u64>> phase3_diag_cbuf_zero_runtime_variants_by_hash;
    mutable std::chrono::steady_clock::time_point phase3_diag_last_log_time{};

    // Records one observation for the diagnostic above (capped at 8 distinct values per
    // hash — hitting the cap is itself the useful signal, not a measurement failure; see
    // the field's own doc comment) and, at most every 30 seconds, logs a cardinality
    // histogram across every hash tracked so far. Called only from
    // CreateGraphicsPipeline(), only where the insert is real (not speculative) and
    // cbuf_key == 0 — see that call site for why those two gates are what make the data
    // meaningful. Safe to call from any worker thread.
    void RecordPhase3RuntimeVariantDiagnostic(u64 unique_hash, u64 diag_base_runtime_hash) const;

    // ---- Runtime-state diagnostic (diagnostic only; does not change behavior) ----
    //
    // Tracks, per graphics unique_hash, the distinct generic_input_types values observed
    // among REAL, non-speculative graphics inserts -- generic_input_types_hash is
    // CityHash64 over the raw 32-entry AttributeType array, the exact same hash
    // SpirvRelevantHash folds in unconditionally for every stage (runtime_info.h), so a
    // value tracked here is directly comparable to what actually distinguishes cache
    // entries. Answers handoff_13's own open question for this field empirically instead
    // of by argument: low cardinality per hash would mean this attribute-format state is
    // realistically worth chasing the way y_negate was (a small, enumerable set of real
    // values a spec constant or scan-time guess could plausibly cover); high cardinality
    // would mean it isn't.
    //
    // Deliberately no cbuf_key==0 gate at the call site, unlike RecordPhase3RuntimeVariantDiagnostic
    // above: that gate exists there because InsertSpeculative() hardcodes cbuf_key=0, so
    // cbuf_key!=0 real inserts are structurally unreachable by any speculative entry
    // regardless of cbuf narrowing. generic_input_types' speculative-matching
    // potential isn't tied to cbuf narrowing at all, so restricting to that same subset
    // here would just throw away real, relevant data for no reason -- the full real
    // population is the right one for this specific question.
    //
    // Own dedicated mutex, same reasoning as phase3_diag_runtime_variants_mutex above:
    // unrelated data, written from the same worker-thread context, shouldn't share a lock
    // with something unrelated.
    mutable std::shared_mutex phase5_diag_generic_input_types_mutex;
    mutable ankerl::unordered_dense::map<u64, std::vector<u64>> phase5_diag_generic_input_types_variants_by_hash;
    mutable std::chrono::steady_clock::time_point phase5_diag_generic_input_types_last_log_time{};

    // Same shape as RecordPhase3RuntimeVariantDiagnostic (capped at 8 distinct values per
    // hash, throttled to at most once per 30 seconds) -- see that method's own doc comment
    // just above for why both those choices are made the way they are; identical reasoning
    // applies here. Safe to call from any worker thread.
    void RecordGenericInputTypesCardinalityDiagnostic(u64 unique_hash,
                                                        u64 generic_input_types_hash) const;

    // ---- Runtime-state diagnostic: confirming (or correcting) two speculative-default
    // guesses runtime_info.h's ApplySpeculativeDefaults flags as REASONED rather than
    // MEASURED -- convert_depth_mode (argued from DepthMode::MinusOneToOne's HW enum
    // value of 0, not from data) and tess_primitive/spacing/clockwise (argued from
    // nothing stronger than "match each enum's own value-0 entry for consistency",
    // explicitly called the lowest-confidence guesses in that pass). Same diagnostic-only
    // contract as everything else in this section: nothing below reads these back. ----
    //
    // Simple true/false frequency, not a cardinality-by-hash table like the two diagnostics
    // above: the open question here isn't "does this vary per shader" (it doesn't --
    // convert_depth_mode is one pipeline-wide GPU register, not a per-draw guess target the
    // way generic_input_types is), it's "which value is actually common", so a plain global
    // count answers it directly and more cheaply.
    mutable std::mutex phase5_diag_convert_depth_mode_mutex;
    mutable u64 phase5_diag_convert_depth_mode_true_count{0};
    mutable u64 phase5_diag_convert_depth_mode_total_count{0};
    mutable std::chrono::steady_clock::time_point phase5_diag_convert_depth_mode_last_log_time{};
    // Called from both real MakeRuntimeInfo assignment sites (VertexB and Geometry) --
    // see runtime_info.h's ApplySpeculativeDefaults for why this guess is the same for
    // both. Throttled the same 30s way as the diagnostics above.
    void RecordConvertDepthModeDiagnostic(bool convert_depth_mode) const;

    // Real distribution across a game's actual TessellationEval-stage draws, not per-hash
    // (tessellation is opt-in per shader, not a "does this shader see multiple states"
    // question the way generic_input_types was) -- three small frequency maps, one per
    // field, logged together. Expect little to no data most sessions: TessellationEval is
    // relatively rare across the real game library, which is exactly why this trio had
    // nothing stronger than an internal-consistency argument to lean on in the first place.
    mutable std::mutex phase5_diag_tess_state_mutex;
    mutable ankerl::unordered_dense::map<Shader::TessPrimitive, u64> phase5_diag_tess_primitive_counts;
    mutable ankerl::unordered_dense::map<Shader::TessSpacing, u64> phase5_diag_tess_spacing_counts;
    mutable u64 phase5_diag_tess_clockwise_true_count{0};
    mutable u64 phase5_diag_tess_total_count{0};
    mutable std::chrono::steady_clock::time_point phase5_diag_tess_state_last_log_time{};
    void RecordTessellationStateDiagnostic(Shader::TessPrimitive primitive,
                                            Shader::TessSpacing spacing, bool clockwise) const;

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
    VideoCommon::PrecacheCompilerTarget precache_compiler_target;

    std::filesystem::path pipeline_cache_filename;

    std::filesystem::path vulkan_pipeline_cache_filename;
    vk::PipelineCache vulkan_pipeline_cache;

    Common::ThreadWorker workers;
    DynamicFeatures dynamic_features;

};

} // namespace Vulkan
