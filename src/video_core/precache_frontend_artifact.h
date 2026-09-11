// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/object_pool.h"

namespace VideoCommon {

// Explicit owned representation of scanner frontend IR. This intentionally
// contains IDs and scalar payloads only: no Program, Shader::Info, Value,
// pool, guest, host, or Vulkan object crosses an artifact boundary.
enum class PrecacheFrontendFreezeError : u8 {
    None,
    UnsupportedInfo,
    AssociatedPseudoOperation,
    UnsupportedValue,
    InvalidGraph,
    TooLarge,
};

struct PrecacheFrozenValue {
    Shader::IR::Type type{};
    u64 payload{};

    bool operator==(const PrecacheFrozenValue&) const noexcept = default;
};

struct PrecacheFrozenInstruction {
    Shader::IR::Opcode opcode{};
    u32 flags{};
    std::vector<PrecacheFrozenValue> args;
    // Only populated for Phi. Each entry owns predecessor block ID for same
    // index in args. u32 max is never a valid ID.
    std::vector<u32> phi_predecessors;

    bool operator==(const PrecacheFrozenInstruction&) const noexcept = default;
};

struct PrecacheFrozenBlock {
    std::vector<PrecacheFrozenInstruction> instructions;
    std::vector<u32> successors;
    u32 order{};

    bool operator==(const PrecacheFrozenBlock&) const noexcept = default;
};

struct PrecacheFrozenSyntaxNode {
    Shader::IR::AbstractSyntaxNode::Type type{};
    PrecacheFrozenValue condition{};
    std::array<u32, 3> blocks{};

    bool operator==(const PrecacheFrozenSyntaxNode&) const noexcept = default;
};

// Exact scanner-to-live recipe contract. Compiler target, original SPH bytes,
// local-memory size, stage, scheduler placement and dispatcher behavior are
// independently matched; program hash alone is never enough.
struct PrecacheFrontendArtifactKey {
    u64 program_identity{};
    u64 compiler_target_fingerprint{};
    u64 source_header{};
    u32 local_memory_size{};
    Shader::Stage stage{};
    u8 scheduler_slot{};
    bool exits_to_dispatcher{};

    bool operator==(const PrecacheFrontendArtifactKey&) const noexcept = default;
};

struct PrecacheFrontendArtifactKeyHash {
    [[nodiscard]] size_t operator()(const PrecacheFrontendArtifactKey& key) const noexcept;
};

struct PrecacheFrontendArtifact {
    Shader::Stage stage{};
    std::array<u32, 3> workgroup_size{};
    Shader::OutputTopology output_topology{};
    u32 output_vertices{};
    u32 invocations{};
    u32 local_memory_size{};
    u32 shared_memory_size{};
    bool is_geometry_passthrough{};
    Shader::IR::FrontendDependencyManifest frontend_dependencies{};
    std::vector<PrecacheFrozenBlock> blocks;
    std::vector<u32> post_order_blocks;
    std::vector<PrecacheFrozenSyntaxNode> syntax_list;

    [[nodiscard]] bool IsValid() const noexcept;
    bool operator==(const PrecacheFrontendArtifact&) const noexcept = default;
};

struct PrecacheFrontendArtifactRecord {
    PrecacheFrontendArtifactKey key;
    PrecacheFrontendArtifact artifact;

    [[nodiscard]] bool IsValid() const noexcept;
};

[[nodiscard]] std::optional<PrecacheFrontendArtifact> FreezePrecacheFrontendArtifact(
    const Shader::IR::Program& program, PrecacheFrontendFreezeError& error);

// Caller supplies fresh pools. Any failed validation returns nullopt before
// restoring a usable Program; caller must use normal frontend fallback.
[[nodiscard]] std::optional<Shader::IR::Program> RestorePrecacheFrontendArtifact(
    const PrecacheFrontendArtifact& artifact, Shader::ObjectPool<Shader::IR::Inst>& inst_pool,
    Shader::ObjectPool<Shader::IR::Block>& block_pool);

// Private all-or-nothing sidecar. A malformed, incomplete, duplicate, or
// stale-contract record produces no artifacts and normal frontend fallback.
[[nodiscard]] bool SavePrecacheFrontendArtifacts(
    const std::filesystem::path& filename,
    std::span<const PrecacheFrontendArtifactRecord> artifacts);
[[nodiscard]] std::optional<std::vector<PrecacheFrontendArtifactRecord>>
LoadPrecacheFrontendArtifacts(const std::filesystem::path& filename);

} // namespace VideoCommon
