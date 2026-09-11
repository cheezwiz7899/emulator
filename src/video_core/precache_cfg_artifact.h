// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <ranges>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate/translate.h"

namespace VideoCommon {

// Exact CFG artifact identity.
struct PrecacheCfgArtifactKey {
    u64 program_identity{};
    Shader::Stage stage{};
    u8 scheduler_slot{};
    bool exits_to_dispatcher{};

    bool operator==(const PrecacheCfgArtifactKey&) const noexcept = default;
};

[[nodiscard]] inline bool PredecodedInstructionsMatchTemplate(
    const Shader::Maxwell::Flow::CFG::Template& cfg,
    std::span<const Shader::Maxwell::PredecodedInstruction> decoded_instructions) {
    std::vector<u32> expected_locations;
    for (const auto& block : cfg.blocks) {
        for (auto location = block.begin; location != block.end; ++location) {
            expected_locations.push_back(location.Offset());
        }
    }
    std::ranges::sort(expected_locations);
    if (std::adjacent_find(expected_locations.begin(), expected_locations.end()) !=
            expected_locations.end() ||
        expected_locations.size() != decoded_instructions.size()) {
        return false;
    }
    for (size_t index = 0; index < expected_locations.size(); ++index) {
        if (decoded_instructions[index].location != expected_locations[index]) {
            return false;
        }
    }
    return true;
}

struct PrecacheCfgArtifactKeyHash {
    [[nodiscard]] size_t operator()(const PrecacheCfgArtifactKey& key) const noexcept {
        size_t hash{static_cast<size_t>(key.program_identity)};
        hash ^= static_cast<size_t>(key.stage) + 0x9e3779b9U + (hash << 6) + (hash >> 2);
        hash ^= static_cast<size_t>(key.scheduler_slot) + 0x9e3779b9U + (hash << 6) + (hash >> 2);
        hash ^=
            static_cast<size_t>(key.exits_to_dispatcher) + 0x9e3779b9U + (hash << 6) + (hash >> 2);
        return hash;
    }
};

[[nodiscard]] inline std::optional<PrecacheCfgArtifactKey> MakePrecacheCfgArtifactKey(
    u64 program_identity, Shader::Stage stage, u32 cfg_entry_address,
    bool exits_to_dispatcher = false) noexcept {
    // A CFG entry is an instruction address, never a virtual Location. Keep
    // invalid/empty source out of a persistent artifact namespace.
    if (program_identity == 0 ||
        static_cast<u32>(stage) > static_cast<u32>(Shader::Stage::VertexA) ||
        cfg_entry_address % 8 != 0) {
        return std::nullopt;
    }
    return PrecacheCfgArtifactKey{
        .program_identity = program_identity,
        .stage = stage,
        .scheduler_slot = static_cast<u8>(cfg_entry_address % 32),
        .exits_to_dispatcher = exits_to_dispatcher,
    };
}

// Owned scanner artifact with no environment, pool, guest pointer, or Vulkan state.
struct PrecacheCfgArtifact {
    PrecacheCfgArtifactKey key;
    Shader::Maxwell::Flow::CFG::Template cfg;
    std::vector<Shader::Maxwell::PredecodedInstruction> decoded_instructions;

    [[nodiscard]] bool IsValid() const noexcept {
        if (key.program_identity == 0 ||
            static_cast<u32>(key.stage) > static_cast<u32>(Shader::Stage::VertexA) ||
            key.scheduler_slot >= 32 || key.scheduler_slot % 8 != 0 || cfg.functions.empty() ||
            cfg.functions.front().entrypoint.IsVirtual() ||
            cfg.functions.front().entrypoint.Offset() % 32 != key.scheduler_slot ||
            cfg.exits_to_dispatcher != key.exits_to_dispatcher) {
            return false;
        }
        if (!Shader::Maxwell::Flow::CFG::IsValidTemplate(cfg)) {
            return false;
        }
        u32 previous_location{};
        bool first = true;
        for (const auto& instruction : decoded_instructions) {
            const auto decoded = Shader::Maxwell::TryDecode(instruction.instruction);
            const bool valid_opcode = decoded ? *decoded == instruction.opcode
                                              : instruction.instruction == 0 &&
                                                    instruction.opcode == Shader::Maxwell::Opcode::NOP;
            if (!Shader::Maxwell::Location::IsRawOffset(instruction.location) ||
                (!first && instruction.location <= previous_location) ||
                !valid_opcode) {
                return false;
            }
            previous_location = instruction.location;
            first = false;
        }
        return PredecodedInstructionsMatchTemplate(cfg, decoded_instructions);
    }

    // Reconstruct for matching program, stage, and dispatcher contract.
    [[nodiscard]] std::optional<Shader::Maxwell::Flow::CFG::Template> RebaseFor(
        u64 program_identity, Shader::Stage stage, Shader::Maxwell::Location destination_entry,
        bool exits_to_dispatcher = false) const {
        if (!IsValid()) {
            return std::nullopt;
        }
        const auto destination_key = MakePrecacheCfgArtifactKey(
            program_identity, stage, destination_entry.Offset(), exits_to_dispatcher);
        if (!destination_key || destination_key->program_identity != key.program_identity ||
            destination_key->stage != key.stage ||
            destination_key->scheduler_slot != key.scheduler_slot ||
            destination_key->exits_to_dispatcher != key.exits_to_dispatcher) {
            return std::nullopt;
        }
        return Shader::Maxwell::Flow::CFG::RebaseTemplate(cfg, cfg.functions.front().entrypoint,
                                                          destination_entry, true);
    }
};

// Private, all-or-nothing disk codec for scanner artifacts. A malformed file
// yields no artifacts; callers must fall back to normal CFG construction.
[[nodiscard]] bool SavePrecacheCfgArtifacts(const std::filesystem::path& filename,
                                            std::span<const PrecacheCfgArtifact> artifacts);
[[nodiscard]] std::optional<std::vector<PrecacheCfgArtifact>> LoadPrecacheCfgArtifacts(
    const std::filesystem::path& filename);

} // namespace VideoCommon
