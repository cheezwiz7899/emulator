// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/stage.h"

namespace VideoCommon {

// This is deliberately an owned description of a scanner source. It is not a
// graphics-pipeline recipe: neither BNSH nor a standalone program provides the
// live runtime, layout, render-pass, or fixed-state inputs required for one.
struct PrecacheSourceStage {
    Shader::Stage stage{};
    u64 source_offset{};
    std::vector<u8> bytes;
    std::optional<u64> program_identity;

    [[nodiscard]] bool HasExactProgramIdentity() const noexcept {
        return program_identity.has_value();
    }
};

enum class PrecacheSourceOrigin : u8 {
    BNSHDeclaredProgram,
    StandaloneProgram,
};

struct PrecacheSourceProgramGroup {
    PrecacheSourceOrigin origin{PrecacheSourceOrigin::BNSHDeclaredProgram};
    u64 source_container_identity{};
    u64 container_offset{};
    u64 variation_index{};
    u64 program_offset{};
    std::vector<PrecacheSourceStage> stages;
};

// A source catalog survives the scanner's process-local lifetime. It contains
// only bounded, owned source program groups and provenance; it is explicitly
// not a graphics request, Shader::Info snapshot, SPIR-V artifact, or Vulkan
// object cache. A later consumer must still supply and validate every missing
// runtime, layout, render-pass, fixed-state and policy dependency.
struct PrecacheSourceCatalog {
    u64 title_id{};
    u64 compiler_target_fingerprint{};
    std::vector<PrecacheSourceProgramGroup> groups;
};

enum class PrecacheSourceCatalogRejection : u8 {
    None,
    Malformed,
    UnsupportedVersion,
    Capacity,
    Integrity,
};

struct PrecacheSourceCatalogLoadResult {
    std::optional<PrecacheSourceCatalog> catalog;
    PrecacheSourceCatalogRejection rejection{PrecacheSourceCatalogRejection::None};
};

struct PrecacheBNSHParseLimits {
    size_t max_groups{4096};
    size_t max_variations{16384};
    size_t max_stage_bytes{1024 * 1024};
    size_t max_total_bytes{64 * 1024 * 1024};
};

enum class PrecacheBNSHParseRejection : u8 {
    None,
    Malformed,
    UnsupportedLayout,
    Capacity,
    AmbiguousStage,
};

struct PrecacheBNSHParseResult {
    std::vector<PrecacheSourceProgramGroup> groups;
    PrecacheBNSHParseRejection rejection{PrecacheBNSHParseRejection::None};
    size_t declared_variations{};
    size_t copied_bytes{};
};

struct PrecacheStandaloneParseResult {
    std::optional<PrecacheSourceProgramGroup> group;
    PrecacheBNSHParseRejection rejection{PrecacheBNSHParseRejection::None};
    size_t copied_bytes{};
};

// Parse only the documented BNSH/GRSC layout used by the scanner. All offsets
// are relative to container_offset and all reads are bounded by container_bytes.
// ByteCode and ControlCode are intentionally not interchangeable here: a record
// whose declared byte-code span cannot be represented exactly is rejected rather
// than falling back to a guessed execution section.
[[nodiscard]] PrecacheBNSHParseResult ParsePrecacheBNSHProgramGroups(
    std::span<const u8> container_bytes, size_t container_offset,
    const PrecacheBNSHParseLimits& limits = {});

// Make a one-stage source record from a format that has no declared graphics
// program grouping. This is intentionally a source catalog record, not a
// guessed partner for another standalone stage. `stage` must agree with SPH.
[[nodiscard]] PrecacheStandaloneParseResult ParsePrecacheStandaloneProgram(
    std::span<const u8> program_bytes, Shader::Stage stage, size_t source_offset,
    const PrecacheBNSHParseLimits& limits = {});

// The wire format is explicit little-endian, versioned, length-bounded and
// protected by an integrity checksum. Decode validates identities against the
// owned code bytes before returning any group. These helpers do not create,
// compile, or serve graphics pipelines.
[[nodiscard]] std::optional<std::vector<u8>> SerializePrecacheSourceCatalog(
    const PrecacheSourceCatalog& catalog, const PrecacheBNSHParseLimits& limits = {});
[[nodiscard]] PrecacheSourceCatalogLoadResult DeserializePrecacheSourceCatalog(
    std::span<const u8> bytes, const PrecacheBNSHParseLimits& limits = {});
[[nodiscard]] bool SavePrecacheSourceCatalog(const std::filesystem::path& filename,
                                             const PrecacheSourceCatalog& catalog,
                                             const PrecacheBNSHParseLimits& limits = {});
[[nodiscard]] PrecacheSourceCatalogLoadResult LoadPrecacheSourceCatalog(
    const std::filesystem::path& filename, const PrecacheBNSHParseLimits& limits = {});

enum class PrecacheDependency : u8 {
    ProgramBytes,
    SchedulingAlignment,
    CbufValues,
    CbufSizes,
    TextureShape,
    TextureBinding,
    ViewportTransform,
    ProprietaryDriver,
    InterfaceState,
    HLEMacroState,
    RuntimeInfo,
    PredecessorStores,
    CompilerPolicy,
    DescriptorLayout,
    FixedPipelineState,
    RenderPass,
    Count,
};

enum class PrecacheDependencyProvenance : u8 {
    Unknown,
    ProgramDerived,
    SourceExact,
    CapturedExact,
    RuntimeABI,
};

struct PrecacheDependencyLedger {
    std::array<PrecacheDependencyProvenance,
               static_cast<size_t>(PrecacheDependency::Count)>
        provenance{};

    [[nodiscard]] PrecacheDependencyProvenance Get(PrecacheDependency dependency) const noexcept;
    void Set(PrecacheDependency dependency, PrecacheDependencyProvenance value) noexcept;
};

enum class PrecacheContractRejection : u8 {
    None,
    UnsupportedStageShape,
    MissingProgramIdentity,
    MissingDependency,
};

struct PrecacheContractDecision {
    bool eligible{};
    PrecacheContractRejection primary_rejection{PrecacheContractRejection::None};
    u64 missing_dependencies{};
};

// CapturedExact is a separate source class: it comes from a real live graphics
// request and is already subject to the normal transferable-cache validation.
// It is deliberately broader than the scanner's initial VertexB+Fragment-only
// source class, but does not permit Compute or duplicate graphics stages.
struct PrecacheCapturedGraphicsStage {
    Shader::Stage stage{};
    u64 program_identity{};
};

// The first D contract intentionally accepts only an exact VertexB+Fragment
// source group. It does not claim that this pair is compilable without every
// dependency below; callers must fill the ledger from an audited source.
[[nodiscard]] PrecacheContractDecision ClassifyPrecacheSourceProgramGroup(
    const PrecacheSourceProgramGroup& group, const PrecacheDependencyLedger& ledger) noexcept;

[[nodiscard]] PrecacheContractDecision ClassifyPrecacheCapturedGraphicsPipeline(
    std::span<const PrecacheCapturedGraphicsStage> stages,
    const PrecacheDependencyLedger& ledger) noexcept;

[[nodiscard]] constexpr u64 PrecacheDependencyBit(PrecacheDependency dependency) noexcept {
    return 1ULL << static_cast<u8>(dependency);
}

} // namespace VideoCommon
