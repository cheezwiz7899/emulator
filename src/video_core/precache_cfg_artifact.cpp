// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "video_core/precache_cfg_artifact.h"

namespace VideoCommon {
namespace {

constexpr std::array<char, 8> kMagic{'c', 'i', 't', 'r', 'c', 'f', 'a', '\0'};
// Version 5 adds scanner-owned decoded instructions. v4 CFG-only records are
// intentionally incompatible: decode replay needs complete payload coverage.
constexpr u32 kVersion = 5;
constexpr u32 kMaxArtifacts = 131072;
constexpr u32 kMaxBlocks = 65536;
constexpr u32 kMaxStackEntries = 1024;
constexpr u32 kMaxIndirectBranches = 65536;
constexpr u32 kMaxDecodedInstructions = 1U << 20;

template <typename T>
void Write(std::ofstream& file, const T& value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <typename T>
void Read(std::ifstream& file, T& value) {
    file.read(reinterpret_cast<char*>(&value), sizeof(value));
}

u32 EncodeBlockId(size_t id) {
    return id == Shader::Maxwell::Flow::CFG::NoTemplateBlock ? std::numeric_limits<u32>::max()
                                                             : static_cast<u32>(id);
}

std::optional<size_t> DecodeBlockId(u32 id, u32 block_count) {
    if (id == std::numeric_limits<u32>::max()) {
        return Shader::Maxwell::Flow::CFG::NoTemplateBlock;
    }
    if (id >= block_count) {
        return std::nullopt;
    }
    return id;
}

bool IsValidStage(u32 stage) {
    return stage <= static_cast<u32>(Shader::Stage::VertexA);
}

void WriteArtifact(std::ofstream& file, const PrecacheCfgArtifact& artifact) {
    const auto& source = artifact.cfg;
    Write(file, artifact.key.program_identity);
    Write(file, static_cast<u32>(artifact.key.stage));
    Write(file, artifact.key.scheduler_slot);
    const u8 exits = artifact.key.exits_to_dispatcher ? 1 : 0;
    Write(file, exits);
    Write(file, static_cast<u32>(source.functions.size()));
    Write(file, static_cast<u32>(source.blocks.size()));
    for (const auto& function : source.functions) {
        Write(file, function.entrypoint.Offset());
        Write(file, static_cast<u32>(function.blocks.size()));
        for (const size_t id : function.blocks) {
            Write(file, static_cast<u32>(id));
        }
    }
    for (const auto& block : source.blocks) {
        Write(file, block.begin.Offset());
        Write(file, block.end.Offset());
        Write(file, static_cast<u8>(block.end_class));
        const auto [pred, negated] = block.cond.GetPred();
        Write(file, static_cast<u16>(block.cond.GetFlowTest()));
        Write(file, static_cast<u8>(pred));
        Write(file, static_cast<u8>(negated));
        Write(file, EncodeBlockId(block.branch_true));
        Write(file, EncodeBlockId(block.branch_false));
        Write(file, static_cast<u32>(block.function_call));
        Write(file, EncodeBlockId(block.return_block));
        Write(file, static_cast<u64>(block.branch_reg));
        Write(file, block.branch_offset);
        const auto& stack = block.stack.Entries();
        Write(file, static_cast<u32>(stack.size()));
        for (const auto& entry : stack) {
            Write(file, static_cast<u8>(entry.token));
            Write(file, entry.target.Offset());
        }
        Write(file, static_cast<u32>(block.indirect_branches.size()));
        for (const auto& [target, address] : block.indirect_branches) {
            Write(file, EncodeBlockId(target));
            Write(file, address);
        }
    }
    Write(file, static_cast<u32>(artifact.decoded_instructions.size()));
    for (const auto& instruction : artifact.decoded_instructions) {
        Write(file, instruction.location);
        Write(file, instruction.instruction);
        Write(file, static_cast<u16>(instruction.opcode));
    }
}

std::optional<PrecacheCfgArtifact> ReadArtifact(std::ifstream& file) {
    PrecacheCfgArtifact artifact;
    u32 stage{};
    u8 exits{};
    u32 function_count{};
    u32 block_count{};
    Read(file, artifact.key.program_identity);
    Read(file, stage);
    Read(file, artifact.key.scheduler_slot);
    Read(file, exits);
    Read(file, function_count);
    Read(file, block_count);
    if (artifact.key.program_identity == 0 || !IsValidStage(stage) ||
        artifact.key.scheduler_slot >= 32 || artifact.key.scheduler_slot % 8 != 0 || exits > 1 ||
        function_count == 0 || block_count == 0 || function_count > block_count ||
        block_count > kMaxBlocks) {
        return std::nullopt;
    }
    artifact.key.stage = static_cast<Shader::Stage>(stage);
    artifact.key.exits_to_dispatcher = exits != 0;
    artifact.cfg.exits_to_dispatcher = artifact.key.exits_to_dispatcher;
    artifact.cfg.functions.resize(function_count);
    artifact.cfg.blocks.resize(block_count);
    std::vector<bool> owned(block_count);
    for (size_t function_id = 0; function_id < artifact.cfg.functions.size(); ++function_id) {
        auto& function = artifact.cfg.functions[function_id];
        u32 entrypoint{};
        u32 count{};
        Read(file, entrypoint);
        Read(file, count);
        if (!Shader::Maxwell::Location::IsRawOffset(entrypoint) || count == 0 ||
            count > block_count) {
            return std::nullopt;
        }
        function.entrypoint = Shader::Maxwell::Location::FromRawOffset(entrypoint);
        function.blocks.resize(count);
        for (size_t& id : function.blocks) {
            u32 value{};
            Read(file, value);
            if (value >= block_count || owned[value]) {
                return std::nullopt;
            }
            owned[value] = true;
            artifact.cfg.blocks[value].owner = function_id;
            id = value;
        }
    }
    for (auto& block : artifact.cfg.blocks) {
        u32 begin{}, end{}, branch_true{}, branch_false{}, function_call{}, return_block{};
        u8 end_class{}, pred{}, negated{};
        u16 flow_test{};
        u64 branch_reg{};
        u32 stack_count{}, indirect_count{};
        Read(file, begin);
        Read(file, end);
        Read(file, end_class);
        Read(file, flow_test);
        Read(file, pred);
        Read(file, negated);
        Read(file, branch_true);
        Read(file, branch_false);
        Read(file, function_call);
        Read(file, return_block);
        Read(file, branch_reg);
        Read(file, block.branch_offset);
        Read(file, stack_count);
        const auto true_id = DecodeBlockId(branch_true, block_count);
        const auto false_id = DecodeBlockId(branch_false, block_count);
        const auto return_id = DecodeBlockId(return_block, block_count);
        if (!Shader::Maxwell::Location::IsRawOffset(begin) ||
            !Shader::Maxwell::Location::IsRawOffset(end) ||
            end_class > static_cast<u8>(Shader::Maxwell::Flow::EndClass::Kill) || negated > 1 ||
            function_call >= function_count || stack_count > kMaxStackEntries || !true_id ||
            !false_id || !return_id) {
            return std::nullopt;
        }
        block.begin = Shader::Maxwell::Location::FromRawOffset(begin);
        block.end = Shader::Maxwell::Location::FromRawOffset(end);
        block.end_class = static_cast<Shader::Maxwell::Flow::EndClass>(end_class);
        block.cond = Shader::IR::Condition{static_cast<Shader::IR::FlowTest>(flow_test),
                                           static_cast<Shader::IR::Pred>(pred), negated != 0};
        block.branch_true = *true_id;
        block.branch_false = *false_id;
        block.function_call = function_call;
        block.return_block = *return_id;
        block.branch_reg = static_cast<Shader::IR::Reg>(branch_reg);
        std::vector<Shader::Maxwell::Flow::StackEntry> stack;
        stack.reserve(stack_count);
        for (u32 index = 0; index < stack_count; ++index) {
            u8 token{};
            u32 target{};
            Read(file, token);
            Read(file, target);
            if (token > static_cast<u8>(Shader::Maxwell::Flow::Token::PLONGJMP) ||
                !Shader::Maxwell::Location::IsRawOffset(target)) {
                return std::nullopt;
            }
            stack.push_back({static_cast<Shader::Maxwell::Flow::Token>(token),
                             Shader::Maxwell::Location::FromRawOffset(target)});
        }
        block.stack.SetEntries(std::move(stack));
        Read(file, indirect_count);
        if (indirect_count > kMaxIndirectBranches) {
            return std::nullopt;
        }
        block.indirect_branches.reserve(indirect_count);
        for (u32 index = 0; index < indirect_count; ++index) {
            u32 target{};
            u32 address{};
            Read(file, target);
            Read(file, address);
            const auto target_id = DecodeBlockId(target, block_count);
            if (!target_id) {
                return std::nullopt;
            }
            block.indirect_branches.push_back({*target_id, address});
        }
    }
    u32 decoded_count{};
    Read(file, decoded_count);
    if (decoded_count > kMaxDecodedInstructions) {
        return std::nullopt;
    }
    artifact.decoded_instructions.resize(decoded_count);
    for (auto& instruction : artifact.decoded_instructions) {
        u16 opcode{};
        Read(file, instruction.location);
        Read(file, instruction.instruction);
        Read(file, opcode);
        instruction.opcode = static_cast<Shader::Maxwell::Opcode>(opcode);
    }
    if (!artifact.IsValid()) {
        return std::nullopt;
    }
    return artifact;
}

} // namespace

bool SavePrecacheCfgArtifacts(const std::filesystem::path& filename,
                              std::span<const PrecacheCfgArtifact> artifacts) {
    if (artifacts.empty() || artifacts.size() > kMaxArtifacts) {
        return false;
    }
    std::vector<PrecacheCfgArtifact> ordered{artifacts.begin(), artifacts.end()};
    std::ranges::sort(ordered, [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.key.program_identity, lhs.key.stage, lhs.key.scheduler_slot,
                        lhs.key.exits_to_dispatcher) <
               std::tie(rhs.key.program_identity, rhs.key.stage, rhs.key.scheduler_slot,
                        rhs.key.exits_to_dispatcher);
    });
    std::unordered_set<PrecacheCfgArtifactKey, PrecacheCfgArtifactKeyHash> keys;
    keys.reserve(ordered.size());
    for (const auto& artifact : ordered) {
        if (!artifact.IsValid() || artifact.cfg.blocks.size() > kMaxBlocks ||
            artifact.cfg.functions.size() > artifact.cfg.blocks.size() ||
            !keys.insert(artifact.key).second) {
            return false;
        }
        for (const auto& block : artifact.cfg.blocks) {
            if (block.stack.Entries().size() > kMaxStackEntries ||
                block.indirect_branches.size() > kMaxIndirectBranches) {
                return false;
            }
        }
        if (artifact.decoded_instructions.size() > kMaxDecodedInstructions) {
            return false;
        }
    }
    const auto temporary = filename.parent_path() / (filename.filename().string() + ".tmp");
    try {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            return false;
        }
        file.exceptions(std::ofstream::failbit);
        file.write(kMagic.data(), kMagic.size());
        Write(file, kVersion);
        Write(file, static_cast<u32>(ordered.size()));
        for (const auto& artifact : ordered) {
            WriteArtifact(file, artifact);
        }
        file.close();
    } catch (...) {
        std::error_code ec;
        std::filesystem::remove(temporary, ec);
        return false;
    }
    // Windows does not replace an existing file with rename(). Preserve the
    // last complete artifact file until the new one is safely installed.
    const auto backup = filename.parent_path() / (filename.filename().string() + ".bak");
    std::error_code ec;
    std::filesystem::remove(backup, ec);
    const bool had_previous = std::filesystem::exists(filename, ec);
    ec.clear();
    if (had_previous) {
        std::filesystem::rename(filename, backup, ec);
        if (ec) {
            std::filesystem::remove(temporary, ec);
            return false;
        }
    }
    std::filesystem::rename(temporary, filename, ec);
    if (ec) {
        std::filesystem::remove(temporary, ec);
        if (had_previous) {
            ec.clear();
            std::filesystem::rename(backup, filename, ec);
        }
        return false;
    }
    if (had_previous) {
        std::filesystem::remove(backup, ec);
    }
    return true;
}

std::optional<std::vector<PrecacheCfgArtifact>> LoadPrecacheCfgArtifacts(
    const std::filesystem::path& filename) {
    try {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            // Save() retains this only during the Windows replacement gap.
            // Recover the last complete artifact set if termination happened
            // after moving the primary but before installing the new file.
            const auto backup = filename.parent_path() / (filename.filename().string() + ".bak");
            file.clear();
            file.open(backup, std::ios::binary);
            if (!file.is_open()) {
                return std::nullopt;
            }
        }
        file.exceptions(std::ifstream::failbit);
        std::array<char, 8> magic{};
        u32 version{};
        u32 count{};
        file.read(magic.data(), magic.size());
        Read(file, version);
        Read(file, count);
        if (magic != kMagic || version != kVersion || count == 0 || count > kMaxArtifacts) {
            return std::nullopt;
        }
        std::vector<PrecacheCfgArtifact> artifacts;
        artifacts.reserve(count);
        std::unordered_set<PrecacheCfgArtifactKey, PrecacheCfgArtifactKeyHash> keys;
        keys.reserve(count);
        for (u32 index = 0; index < count; ++index) {
            auto artifact = ReadArtifact(file);
            if (!artifact || !keys.insert(artifact->key).second) {
                return std::nullopt;
            }
            artifacts.push_back(std::move(*artifact));
        }
        if (file.peek() != std::char_traits<char>::eof()) {
            return std::nullopt;
        }
        return artifacts;
    } catch (...) {
        return std::nullopt;
    }
}

} // namespace VideoCommon
