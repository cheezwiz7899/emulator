// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/program_header.h"
#include "shader_recompiler/shader_info.h"
#include "shader_recompiler/stage.h"

namespace Shader {

// Forward declaration so Environment can expose a no-RTTI downcast helper.
// Defined in video_core/shader_environment.h.
} // namespace Shader
namespace VideoCommon { class GenericEnvironment; }
namespace VideoCommon { class FileEnvironment; }
namespace Shader {

class Environment {
public:
    virtual ~Environment() = default;

    /// Returns a GenericEnvironment* if this object derives from GenericEnvironment,
    /// nullptr otherwise (e.g. FileEnvironment). Used in place of dynamic_cast
    /// when the build disables RTTI (-fno-rtti).
    [[nodiscard]] virtual VideoCommon::GenericEnvironment* AsGenericEnvironment() noexcept {
        return nullptr;
    }
    [[nodiscard]] virtual const VideoCommon::GenericEnvironment*
        AsGenericEnvironment() const noexcept {
        return nullptr;
    }

    /// Returns a FileEnvironment* if this object IS a FileEnvironment, nullptr
    /// otherwise. Same no-RTTI downcast pattern as AsGenericEnvironment() above,
    /// for the one class that deliberately does NOT derive from GenericEnvironment
    /// but still deserializes real (not guessed) cbuf/texture capture data from
    /// disk — see FileEnvironment::CapturedCbufValues() and friends.
    [[nodiscard]] virtual VideoCommon::FileEnvironment* AsFileEnvironment() noexcept {
        return nullptr;
    }
    [[nodiscard]] virtual const VideoCommon::FileEnvironment*
        AsFileEnvironment() const noexcept {
        return nullptr;
    }

    [[nodiscard]] virtual u64 ReadInstruction(u32 address) = 0;

    [[nodiscard]] virtual u32 ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) = 0;

    // Same read as ReadCbufValue, called ONLY from GetTextureHandle() (texture_pass.cpp)
    // for the two reads that resolve a bindless texture handle — never from anywhere
    // else (constant-propagation's FoldDriverConstBuffer and the indirect-branch-table
    // walk in control_flow.cpp both call plain ReadCbufValue(), deliberately not this).
    // That makes this call site itself the signal: whichever (index, offset) pairs
    // arrive here are — as far as every Environment::ReadCbufValue caller in the
    // codebase is concerned — used for NOTHING except resolving a handle that
    // ReadTextureType()/ReadTexturePixelFormat() then resolve independently and that
    // spirv_cache's texture_key already captures on its own. Default implementation
    // just forwards to ReadCbufValue() (identical behavior to before this existed) —
    // GenericEnvironment is the only override, and it ALSO records the (index, offset)
    // pair for VideoCommon::GraphicsPipelineCacheKey-adjacent diagnostics (see
    // CapturedTextureHandleCbufKeys() in shader_environment.h). Every other Environment
    // (FileEnvironment, the scanner's SpeculativeShaderEnvironment) gets the safe
    // default: no tagging, same as if this method didn't exist.
    [[nodiscard]] virtual u32 ReadCbufValueForTextureHandle(u32 cbuf_index, u32 cbuf_offset) {
        return ReadCbufValue(cbuf_index, cbuf_offset);
    }

    /// Returns the byte size of the const buffer at cbuf_index, or 0 if not bound /
    /// not knowable from this environment. Callers treat 0 as "unknown" and fall
    /// back to the original conservative default (e.g. disk-cache replay path).
    [[nodiscard]] virtual u32 ReadCbufSize([[maybe_unused]] u32 cbuf_index) {
        return 0;
    }

    [[nodiscard]] virtual TextureType ReadTextureType(u32 raw_handle) = 0;

    [[nodiscard]] virtual TexturePixelFormat ReadTexturePixelFormat(u32 raw_handle) = 0;

    // Phase 4 (specialization-constant texture-type resolution) feasibility instrumentation —
    // see handoff_04_specialization_constants_investigation.md, item 2: "how many distinct
    // (type, format) pairs does a single logical texture slot actually take on in real
    // content". Called immediately after a texture handle resolves, from the SAME call sites
    // in texture_pass.cpp that call ReadTextureType()/ReadTexturePixelFormat() for handle
    // resolution (mirrors ReadCbufValueForTextureHandle's pattern above, including the
    // reasoning for why a default no-op is safe here). Two independent hooks, not one
    // combined (type, format) callback: the two reads are NOT reliably called together at
    // the same site (e.g. the ImageQueryDimensions case only ever calls the type read; the
    // texel-fetch SNORM-workaround case only ever calls the format read), so a combined
    // signature would force a meaningless placeholder value on whichever half wasn't
    // actually resolved at that call site.
    //
    // "Slot" identity is (this shader's unique_hash, cbuf_index, cbuf_offset) — the same
    // (index, offset) pair ReadCbufValueForTextureHandle tags, plus the shader identity,
    // since the same cbuf coordinates mean different logical textures in different shaders.
    // Default no-op — only GenericEnvironment overrides these, so only real, live-gameplay
    // translations contribute (never FileEnvironment disk-replay or the scanner's
    // SpeculativeShaderEnvironment, which returns a fixed guess unconditionally and would
    // only pollute the distinct-value count with a constant that was never really observed).
    //
    // NOT YET WIRED TO ANY REPORTING CADENCE — see GenericEnvironment::
    // LogTextureSlotVarianceReportThrottled()'s doc comment in shader_environment.h for where
    // this data actually gets surfaced, and its own doc comment for why this whole mechanism
    // is unverified: written without the ability to build or run this codebase, needs a real
    // compile and a real play session before anyone trusts its output.
    virtual void RecordResolvedTextureType([[maybe_unused]] u32 cbuf_index,
                                           [[maybe_unused]] u32 cbuf_offset,
                                           [[maybe_unused]] u32 handle,
                                           [[maybe_unused]] TextureType type) {}
    virtual void RecordResolvedTexturePixelFormat([[maybe_unused]] u32 cbuf_index,
                                                  [[maybe_unused]] u32 cbuf_offset,
                                                  [[maybe_unused]] TexturePixelFormat format) {}

    [[nodiscard]] virtual bool IsTexturePixelFormatInteger(u32 raw_handle) = 0;

    // Third axis of the same instrumentation, added after the first two (TextureType,
    // TexturePixelFormat) shipped and were checked against 5 real sessions -- neither showed
    // enough real-world benefit to justify further Phase 4 work on its own (TextureType:
    // 16/8671 observed (shader, slot) pairs ever varied; TexturePixelFormat: 0/8671, dead end
    // -- see handoff_09/handoff_10). Rather than conclude no other axis could exist, this adds
    // the one other genuinely plausible, currently-untracked, SPIR-V-shape-relevant candidate
    // this codebase already resolves per-handle: IsTexturePixelFormatInteger, called just
    // above. Component type (integer vs float) is baked into OpTypeImage's Sampled Type
    // operand (spirv_emit_context.cpp's ImageType) for BOTH sampled and storage images, unlike
    // PixelFormat which the caveat above restricts to storage only -- if this axis shows real
    // variance anywhere, it would apply more broadly than PixelFormat ever could have. Same
    // call-site pattern as the other two: hook in from texture_pass.cpp right where
    // IsTexturePixelFormatInteger(env, cbuf) already gets called, default no-op so only real
    // GenericEnvironment translations contribute.
    virtual void RecordResolvedIsTexturePixelFormatInteger(
        [[maybe_unused]] u32 cbuf_index, [[maybe_unused]] u32 cbuf_offset,
        [[maybe_unused]] bool is_integer) {}

    [[nodiscard]] virtual u32 ReadViewportTransformState() = 0;

    [[nodiscard]] virtual u32 TextureBoundBuffer() const = 0;

    [[nodiscard]] virtual u32 LocalMemorySize() const = 0;

    [[nodiscard]] virtual u32 SharedMemorySize() const = 0;

    [[nodiscard]] virtual std::array<u32, 3> WorkgroupSize() const = 0;

    [[nodiscard]] virtual bool HasHLEMacroState() const = 0;

    [[nodiscard]] virtual std::optional<ReplaceConstant> GetReplaceConstBuffer(u32 bank,
                                                                               u32 offset) = 0;

    virtual void Dump(u64 pipeline_hash, u64 shader_hash) = 0;

    [[nodiscard]] const ProgramHeader& SPH() const noexcept {
        return sph;
    }

    [[nodiscard]] const std::array<u32, 8>& GpPassthroughMask() const noexcept {
        return gp_passthrough_mask;
    }

    [[nodiscard]] Stage ShaderStage() const noexcept {
        return stage;
    }

    [[nodiscard]] u32 StartAddress() const noexcept {
        return start_address;
    }

    [[nodiscard]] bool IsProprietaryDriver() const noexcept {
        return is_proprietary_driver;
    }

protected:
    ProgramHeader sph{};
    std::array<u32, 8> gp_passthrough_mask{};
    Stage stage{};
    u32 start_address{};
    bool is_proprietary_driver{};
};

// Phase 4 (specialization-constant texture-type resolution). Each entry is one
// (cbuf_index, cbuf_offset) coordinate confirmed to have real, per-title-varying texture-type
// slots (Color2D vs ColorArray2D).
//
// FULLY ADAPTIVE, both live and cross-session -- no hardcoded coordinate list anywhere,
// including the two TotK confirmed clean (12+3 real shaders, (2,192)/(2,280)). Removed
// deliberately once the adaptive mechanism below was confirmed to cover them functionally:
// keeping them as permanent hardcoded defaults on top of a mechanism whose whole point is not
// needing that would have been redundant, not extra safety. See handoff_09/handoff_10 for the
// full design and both revisions.
//
// The ACTIVE table for a session (ActivePhase4PrototypeSlots() below) starts, at boot, as
// whatever VideoCommon::LoadPhase4PrototypeSlots (video_core/phase4_prototype_slots_file.h)
// finds in THIS game's own phase4_prototype_slots.bin -- empty for a game/profile that's never
// hit this path before, same as every coordinate was before Phase 4 existed at all. A
// coordinate discovered THIS session (via GenericEnvironment::RecordResolvedTextureType's
// existing variance instrumentation, shader_environment.cpp, hitting a genuine second distinct
// value at a not-yet-active coordinate) grows the active table immediately, live -- any shader
// translated from that point on in the SAME session gets the polymorphic treatment for it --
// and is persisted (PipelineCache::~PipelineCache, vk_pipeline_cache.cpp) so every session
// after this one for this game starts with it already active.
//
// Real consequence worth being explicit about, not just implicit in "adaptive": a fresh
// profile, a fresh install, or a wiped phase4_prototype_slots.bin costs ONE session's worth of
// rediscovery for coordinates that were previously always-active defaults -- the first
// shader(s) touching them that session translate ordinarily (no polymorphic treatment) until
// the live trigger fires partway through, exactly as if this were the very first time this
// mechanism had ever seen this game. Every session after that starts warm again. This also
// means an EXISTING spirv_cache.bin from a session that ran under the old hardcoded-defaults
// scheme may see a one-time round of extra misses early in the next session for shaders
// translated before re-discovery fires (their texture_key differs: computed without exclusion
// this time, until the coordinate is relearned) -- not a correctness issue (see the
// no-version-bump-needed reasoning below, which already covers exactly this category of
// change), just a one-time re-warming cost, same shape as any other cache-key change already
// accepted as "silently unreachable, not wrong."
//
// What's still deliberately NOT done: re-translating a shader that's ALREADY been translated
// this session under a smaller table. That needs real cache-entry invalidation while other
// threads may be mid-read of the entry being invalidated -- machinery that doesn't exist
// anywhere in this codebase today, in exactly the class of code (pipeline/SPIR-V cache
// consistency) that has already produced one freeze, a second freeze, and a real driver crash
// once in this investigation. Growing the table live is safe without that machinery
// specifically because it only ever changes what NOT-YET-translated shaders do; an
// already-cached shader is simply unaffected either way, correct, just not optimized for a
// coordinate it didn't know about at its own translation time.
//
// A slot's index into the ACTIVE table doubles as its SpecId (spirv_emit_context.cpp) and its
// bit position in GraphicsPipelineCacheKey::phase4_prototype_needs_array_variant
// (vk_graphics_pipeline.h) for the rest of that session. Stability, not staticness, is the
// real invariant (see SetActivePhase4PrototypeSlots's doc comment): an already-published
// index never moves, because every real call site only ever grows the table by appending, so
// any two shaders (or worker threads) that reference the same active slot always agree on its
// SpecId without needing any allocation/coordination beyond that.
//
// No extra cache-version bump needed when the active table's contents differ from what a
// PRIOR session (or a different profile/install) had, including the "used to have hardcoded
// defaults, now starts empty" transition this specific patch introduces: texture_key already
// depends on IsPhase4PrototypeSlot (via ComputeTextureKeyExcludingHandles's handle-exclusion
// set, shader_environment.cpp/spirv_cache.cpp), so a shader whose slot's active-or-not status
// differs from a previous run naturally computes a DIFFERENT texture_key -- old entries just
// don't match on lookup (silently unreachable, same acceptable category v7 through v9 already
// established), entirely from the key changing, with no separate version signal required. This
// only covers the active table's CONTENTS varying (session to session, install to install,
// before/after this exact patch); a change to the MECHANISM itself (new instruction coverage,
// N-way variants, etc.) would still need an explicit SPIRV_CACHE_VERSION bump the same way
// v7-v9 did, since that changes what EVERY game's cache means, not just what one game's active
// table happens to contain right now.
//
// Shared between texture_pass.cpp (which uses IsPhase4PrototypeSlot to canonicalize
// flags.type and Phase4PrototypeSlotId to fill in TextureDescriptor::phase4_prototype_slot_id)
// and shader_environment.cpp (which uses IsPhase4PrototypeSlot to know which resolved handles
// to exclude from texture_key, so every real variant of an active slot hashes to the same key
// instead of fragmenting the cache) -- one definition so all three can't independently drift
// out of sync.
struct Phase4PrototypeSlot {
    u32 cbuf_index;
    u32 cbuf_offset;
};

// Upper bound on how many coordinates can be active in one session. Real data so far: TotK
// needs 2, MK8D needs 0 -- nothing tested has come close to this. A coordinate discovered
// beyond this cap is simply never learned (see MergePhase4PrototypeSlots below) and that slot
// keeps behaving exactly as it did before this mechanism existed -- fall-back-safe by
// construction, not a special case. u64 bitmask in GraphicsPipelineCacheKey
// (vk_graphics_pipeline.h) hard-caps this at 64 regardless; 8 is nowhere near that ceiling and
// leaves real headroom to raise later without touching that struct's layout.
inline constexpr size_t kMaxPhase4PrototypeSlots = 8;
static_assert(kMaxPhase4PrototypeSlots <= 64,
              "GraphicsPipelineCacheKey::phase4_prototype_needs_array_variant is a u64 bitmask");

namespace detail {
inline std::atomic<const std::vector<Phase4PrototypeSlot>*> g_active_phase4_prototype_slots{
    nullptr};
} // namespace detail

// Dedupes `extra` on (cbuf_index, cbuf_offset) against itself, preserving input order, and
// truncates to kMaxPhase4PrototypeSlots. Pure/deterministic given the same `extra`. Two real
// callers, both building `extra` from "whatever's already active" plus something new, never by
// reordering or dropping an existing entry -- see SetActivePhase4PrototypeSlots's doc comment
// for why that specific property is what makes repeated calls safe: PipelineCache::
// LoadDiskResources (`extra` = VideoCommon::LoadPhase4PrototypeSlots's on-disk result, at
// boot) and GenericEnvironment::RecordResolvedTextureType (`extra` = the current active table
// plus one newly-discovered coordinate, live, whenever variance is first proven for a
// not-yet-active slot).
[[nodiscard]] inline std::vector<Phase4PrototypeSlot>
MergePhase4PrototypeSlots(std::span<const Phase4PrototypeSlot> extra) {
    std::vector<Phase4PrototypeSlot> merged;
    for (const Phase4PrototypeSlot& slot : extra) {
        if (merged.size() >= kMaxPhase4PrototypeSlots) {
            break;
        }
        const bool already_present{std::any_of(
            merged.begin(), merged.end(), [&slot](const Phase4PrototypeSlot& existing) {
                return existing.cbuf_index == slot.cbuf_index &&
                       existing.cbuf_offset == slot.cbuf_offset;
            })};
        if (!already_present) {
            merged.push_back(slot);
        }
    }
    return merged;
}

// Called once per game boot (PipelineCache::LoadDiskResources, vk_pipeline_cache.cpp) --
// AND, since handoff_09's revision, again during a session whenever
// GenericEnvironment::RecordResolvedTextureType (shader_environment.cpp) discovers a new
// variance-proven coordinate, so shaders translated later in the SAME session (not just next
// session) benefit too. `slot_list` should already be the output of MergePhase4PrototypeSlots
// (deduplicated, capped) -- this function trusts its caller rather than re-validating, since
// both real callers already do.
//
// The actual safety property this whole mechanism depends on is STABILITY, not staticness:
// once a coordinate has a SpecId/bit-position (its index in whatever `slot_list` was last
// published), that index must never change for the rest of the session -- growing the table
// is fine, REPLACING an existing entry's position is not. Both real callers only ever build
// `slot_list` by taking the CURRENT table and appending, so this invariant holds by construction
// rather than by convention -- there is no call site anywhere that constructs `slot_list` by
// reordering or dropping an already-published entry.
//
// Deliberately leaked (`new`, never `delete`d) on every call, not just the first: may be read
// from worker threads doing bulk shader translation at any point after any prior publish --
// safely reclaiming an OLD pointer while some other thread could still be mid-read of it
// would need a real reclamation scheme (hazard pointers or similar); leaking a handful of
// small structs a handful of times per session (real data: variance events are rare, order of
// 1-10 per session) is a better trade than adding that machinery for this. Release/acquire
// (not relaxed) specifically so "this call happened, and the vector it published is fully
// constructed" is visible to every thread that later reads it, not just this one.
inline void SetActivePhase4PrototypeSlots(std::vector<Phase4PrototypeSlot> slot_list) {
    detail::g_active_phase4_prototype_slots.store(
        new std::vector<Phase4PrototypeSlot>(std::move(slot_list)), std::memory_order_release);
}

// The table every Phase4Prototype* function below actually reads. Falls back to an EMPTY
// table (not a hardcoded default -- see the removal note above) whenever
// SetActivePhase4PrototypeSlots hasn't run yet for this process -- covers title_id==0
// (LoadDiskResources' own early-return case) and any shader translated before boot-time
// loading completes. Safe by construction: an empty table means IsPhase4PrototypeSlot is
// false for everything, i.e. ordinary pre-Phase-4 behavior, exactly the same fallback shape
// as a corrupt/unreadable phase4_prototype_slots.bin already degrades to.
[[nodiscard]] inline std::span<const Phase4PrototypeSlot> ActivePhase4PrototypeSlots() noexcept {
    const auto* const active{
        detail::g_active_phase4_prototype_slots.load(std::memory_order_acquire)};
    if (active) {
        return *active;
    }
    return {};
}

// Returns the slot's index into ActivePhase4PrototypeSlots() (also its SpecId) if
// (cbuf_index, cbuf_offset) names a currently-active Phase 4 prototype slot, or std::nullopt
// otherwise. Not constexpr -- the active table is runtime data -- but still a tiny linear scan
// over at most kMaxPhase4PrototypeSlots elements, called only at shader TRANSLATION time, not
// per-draw, so the cost of not being constexpr here is not measurable.
[[nodiscard]] inline std::optional<u32> Phase4PrototypeSlotId(u32 cbuf_index,
                                                               u32 cbuf_offset) noexcept {
    const std::span<const Phase4PrototypeSlot> slot_list{ActivePhase4PrototypeSlots()};
    for (size_t i = 0; i < slot_list.size(); ++i) {
        if (slot_list[i].cbuf_index == cbuf_index && slot_list[i].cbuf_offset == cbuf_offset) {
            return static_cast<u32>(i);
        }
    }
    return std::nullopt;
}

[[nodiscard]] inline bool IsPhase4PrototypeSlot(u32 cbuf_index, u32 cbuf_offset) noexcept {
    return Phase4PrototypeSlotId(cbuf_index, cbuf_offset).has_value();
}

} // namespace Shader
