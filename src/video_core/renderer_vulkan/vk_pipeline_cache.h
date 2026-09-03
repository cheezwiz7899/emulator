// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-FileCopyrightText: Copyright 2025 Citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
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
#include "shader_recompiler/runtime_info.h"
#include "shader_recompiler/varying_state.h"
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
    // Real, non-speculative per-stage translated-shader data, keyed by unique_hash,
    // captured from CreateGraphicsPipeline()'s own real translations (see the capture
    // site inside that function's per-stage loop). Same four fields MakeRuntimeInfo()
    // reads from a real previous_program, for the same reason the scanner's
    // PreviousStageStoresSnapshot (citron/main.cpp) does — plain value types, no
    // dependency on the ObjectPools the owning IR::Program's block/instruction graph
    // actually lives in, so safe to keep around past that Program's own lifetime.
    // Distinct type from the scanner's, despite matching shape: this one is populated
    // continuously across an entire live session from real draws (overwritten with the
    // most recent real observation each time — per Phase 3's own cardinality data, most
    // shaders only ever show one real state anyway, so "most recent" is usually also
    // "the only one"), where the scanner's is scoped to one BNSH shader program's five
    // sibling stages during one boot-time scan pass.
    struct RealStageStoresSnapshot {
        Shader::VaryingState stores{};
        std::map<Shader::IR::Attribute, Shader::IR::Attribute> legacy_stores_mapping{};
        Shader::VaryingState passthrough{};
        bool is_geometry_passthrough{};
    };
    mutable std::shared_mutex real_stage_stores_mutex;
    mutable ankerl::unordered_dense::map<u64, RealStageStoresSnapshot> real_stage_stores_by_hash;

    // Looks up real_stage_stores_by_hash for previous_stage_unique_hash (0, or a hash
    // with no recorded real snapshot yet, both correctly yield std::nullopt — the
    // caller already handles "no real data" via the same sentinel fallback it used
    // before this existed). Cheap (single shared-lock map lookup + a small value-type
    // copy) and safe to call from OnNewShaderSeen()'s caller thread — deliberately
    // resolved here, synchronously, rather than deferred into the speculative_worker
    // background thread, so that thread never needs to touch real_stage_stores_mutex
    // at all.
    std::optional<RealStageStoresSnapshot> ResolveRealStageStoresSnapshot(
        u64 previous_stage_unique_hash) const;

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
    // Cross-session Phase 4 adaptive slot learning -- see Shader::ActivePhase4PrototypeSlots's
    // doc comment (environment.h) for the full design. Loaded in LoadDiskResources (published
    // via Shader::SetActivePhase4PrototypeSlots before any shader translation starts -- empty
    // for a fresh profile, no hardcoded defaults); saved in the destructor from whatever
    // GenericEnvironment::RecordResolvedTextureType recorded as candidates this session
    // (VideoCommon::TakePhase4PrototypeCandidates, shader_environment.cpp).
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
    // real and both needed for different arrays).
    //
    // Value is a bitmask over Shader::ActivePhase4PrototypeSlots() (environment.h; bit i set means
    // this shader has a descriptor for slot i), not a plain bool -- was a single bool when
    // exactly one slot could ever be marked polymorphic, widened so multiple known slots can
    // be tracked per shader without one colliding into another's entry. 0 (not absent) once a
    // shader has been seen and has none of the known slots, so ResolvePhase4PrototypeSpecValue
    // can skip straight to "nothing to do" for known-irrelevant shaders without re-deciding it
    // every draw; distinguished from "not yet seen" the same way as before, via .contains().
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
    mutable ankerl::unordered_dense::map<u64, u32> phase4_prototype_fragment_shader_table;

    // Phase 4 narrow prototype's graphics_cache lookup-timing fix. Called from
    // CurrentGraphicsPipeline(), after RefreshStages()/state.Refresh() but before the
    // graphics_cache/Next() lookups that graphics_key feeds -- see this method's definition in
    // vk_pipeline_cache.cpp for the full reasoning and its real, flagged limitations (assumes
    // no secondary cbuf combine for any known slot; gated by
    // phase4_prototype_fragment_shader_table above rather than running unconditionally, after
    // an earlier version of this method did exactly that and froze real gameplay). Returns a
    // bitmask (bit i = slot i resolved to its array variant on this draw), directly assignable
    // to GraphicsPipelineCacheKey::phase4_prototype_needs_array_variant -- widened from a
    // single bool alongside the table above, for the same reason.
    u64 ResolvePhase4PrototypeSpecValue() const;

    // ---- Phase 3 groundwork (diagnostic only — nothing below reads these back or
    // changes caching/guessing behavior; see RecordPhase3RuntimeVariantDiagnostic()'s
    // definition in vk_pipeline_cache.cpp for the full rationale) ----
    //
    // Tracks, per graphics unique_hash, the distinct "core RuntimeInfo" values
    // (diag_base_runtime_hash — see its declaration in CreateGraphicsPipeline(), which
    // deliberately excludes the binding-offset fold, the same split
    // RuntimeCoreComponentNeverMatchedCount()/RuntimeBindingComponentNeverMatchedCount()
    // already rely on in spirv_cache.h) observed among REAL, non-speculative graphics
    // inserts whose narrowed cbuf_key == 0 — exactly the population Phase 1 grew (a
    // stable 33-34% baseline to 41-46%, see SpirvCache::real_cbuf_zero_count_) and
    // exactly the population a speculative entry could ever hope to match, since
    // InsertSpeculative() always hardcodes cbuf_key=0. This is what answers Phase 3's
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

    // ---- Phase 5 groundwork (diagnostic only — same "nothing below reads these back or
    // changes caching/guessing behavior" as the Phase 3 block above) ----
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
    // regardless of Phase 3's own outcome. generic_input_types' speculative-matching
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

    // ---- Phase 5 groundwork: confirming (or correcting) the two speculative-default
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

    std::filesystem::path pipeline_cache_filename;

    std::filesystem::path vulkan_pipeline_cache_filename;
    vk::PipelineCache vulkan_pipeline_cache;

    Common::ThreadWorker workers;
    DynamicFeatures dynamic_features;

};

} // namespace Vulkan
