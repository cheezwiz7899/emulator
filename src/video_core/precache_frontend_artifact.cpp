// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/precache_frontend_artifact.h"

#include <algorithm>
#include <array>
#include <bit>
#include <fstream>
#include <limits>
#include <system_error>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace VideoCommon {
namespace {

constexpr u32 kInvalidId = std::numeric_limits<u32>::max();
constexpr size_t kMaxBlocks = 65536;
constexpr size_t kMaxInstructions = 1U << 20;
constexpr size_t kMaxSyntaxNodes = 1U << 20;
constexpr u32 kMaxArtifacts = 131072;
constexpr std::array<char, 8> kMagic{'c', 'i', 't', 'r', 'f', 'i', 'r', '\0'};
constexpr u32 kVersion = 1;

template <typename T>
void Write(std::ofstream& file, const T& value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <typename T>
void Read(std::ifstream& file, T& value) {
    file.read(reinterpret_cast<char*>(&value), sizeof(value));
}

[[nodiscard]] bool IsValidBlockId(u32 id, size_t count) noexcept {
    return id != kInvalidId && id < count;
}

[[nodiscard]] bool IsValidStage(Shader::Stage stage) noexcept {
    return static_cast<u32>(stage) <= static_cast<u32>(Shader::Stage::VertexA);
}

[[nodiscard]] bool IsValidKey(const PrecacheFrontendArtifactKey& key) noexcept {
    return key.program_identity != 0 && IsValidStage(key.stage) && key.scheduler_slot < 32 &&
           key.scheduler_slot % 8 == 0;
}

[[nodiscard]] bool IsValidOpcode(Shader::IR::Opcode opcode) noexcept {
    return static_cast<size_t>(opcode) <
           sizeof(Shader::IR::Detail::META_TABLE) / sizeof(Shader::IR::Detail::META_TABLE[0]);
}

[[nodiscard]] bool IsValidOutputTopology(Shader::OutputTopology topology) noexcept {
    return topology == Shader::OutputTopology{} || topology == Shader::OutputTopology::PointList ||
           topology == Shader::OutputTopology::LineStrip ||
           topology == Shader::OutputTopology::TriangleStrip;
}

[[nodiscard]] bool IsSupportedValueType(Shader::IR::Type type) noexcept {
    using Shader::IR::Type;
    switch (type) {
    case Type::Void:
    case Type::Opaque:
    case Type::Reg:
    case Type::Pred:
    case Type::Attribute:
    case Type::Patch:
    case Type::U1:
    case Type::U8:
    case Type::U16:
    case Type::U32:
    case Type::U64:
    case Type::F32:
    case Type::F64:
        return true;
    default:
        return false;
    }
}

[[nodiscard]] bool IsValidFrozenValue(const PrecacheFrozenValue& value,
                                      size_t instruction_count) noexcept {
    using Shader::IR::Type;
    if (!IsSupportedValueType(value.type)) {
        return false;
    }
    switch (value.type) {
    case Type::Void:
        return value.payload == 0;
    case Type::Opaque:
        return value.payload < instruction_count;
    case Type::Reg:
        return value.payload <= static_cast<u64>(Shader::IR::Reg::RZ);
    case Type::Pred:
        return value.payload <= static_cast<u64>(Shader::IR::Pred::PT);
    case Type::Attribute:
        return value.payload <= static_cast<u64>(Shader::IR::Attribute::DrawID);
    case Type::Patch:
        return value.payload <= static_cast<u64>(Shader::IR::Patch::Component119);
    case Type::U1:
        return value.payload <= 1;
    case Type::U8:
        return value.payload <= std::numeric_limits<u8>::max();
    case Type::U16:
        return value.payload <= std::numeric_limits<u16>::max();
    case Type::U32:
    case Type::F32:
        return value.payload <= std::numeric_limits<u32>::max();
    case Type::U64:
    case Type::F64:
        return true;
    default:
        return false;
    }
}

[[nodiscard]] std::optional<PrecacheFrozenValue> FreezeValue(
    const Shader::IR::Value& value,
    const std::unordered_map<const Shader::IR::Inst*, u32>& instruction_ids) {
    // Value::Type resolves Identity and Phi. Preserve non-immediate values
    // as explicit instruction IDs instead; otherwise a restored identity
    // would silently become its result value and alter use/SSA semantics.
    if (!value.IsImmediate()) {
        const auto it{instruction_ids.find(value.Inst())};
        if (it == instruction_ids.end()) {
            return std::nullopt;
        }
        return PrecacheFrozenValue{.type = Shader::IR::Type::Opaque, .payload = it->second};
    }
    const Shader::IR::Type type{value.Type()};
    if (!IsSupportedValueType(type)) {
        return std::nullopt;
    }
    PrecacheFrozenValue frozen{.type = type};
    switch (type) {
    case Shader::IR::Type::Void:
        return frozen;
    case Shader::IR::Type::Opaque:
        return std::nullopt;
    case Shader::IR::Type::Reg:
        frozen.payload = static_cast<u64>(value.Reg());
        return frozen;
    case Shader::IR::Type::Pred:
        frozen.payload = static_cast<u64>(value.Pred());
        return frozen;
    case Shader::IR::Type::Attribute:
        frozen.payload = static_cast<u64>(value.Attribute());
        return frozen;
    case Shader::IR::Type::Patch:
        frozen.payload = static_cast<u64>(value.Patch());
        return frozen;
    case Shader::IR::Type::U1:
        frozen.payload = value.U1() ? 1 : 0;
        return frozen;
    case Shader::IR::Type::U8:
        frozen.payload = value.U8();
        return frozen;
    case Shader::IR::Type::U16:
        frozen.payload = value.U16();
        return frozen;
    case Shader::IR::Type::U32:
        frozen.payload = value.U32();
        return frozen;
    case Shader::IR::Type::U64:
        frozen.payload = value.U64();
        return frozen;
    case Shader::IR::Type::F32:
        frozen.payload = std::bit_cast<u32>(value.F32());
        return frozen;
    case Shader::IR::Type::F64:
        frozen.payload = std::bit_cast<u64>(value.F64());
        return frozen;
    default:
        return std::nullopt;
    }
}

[[nodiscard]] std::optional<Shader::IR::Value> RestoreValue(
    const PrecacheFrozenValue& frozen, std::span<Shader::IR::Inst* const> instructions) {
    using Shader::IR::Type;
    if (!IsValidFrozenValue(frozen, instructions.size())) {
        return std::nullopt;
    }
    switch (frozen.type) {
    case Type::Void:
        return Shader::IR::Value{};
    case Type::Opaque:
        if (frozen.payload >= instructions.size()) {
            return std::nullopt;
        }
        return Shader::IR::Value{instructions[frozen.payload]};
    case Type::Reg:
        if (frozen.payload > static_cast<u64>(Shader::IR::Reg::RZ)) {
            return std::nullopt;
        }
        return Shader::IR::Value{static_cast<Shader::IR::Reg>(frozen.payload)};
    case Type::Pred:
        if (frozen.payload > static_cast<u64>(Shader::IR::Pred::PT)) {
            return std::nullopt;
        }
        return Shader::IR::Value{static_cast<Shader::IR::Pred>(frozen.payload)};
    case Type::Attribute:
        if (frozen.payload > static_cast<u64>(Shader::IR::Attribute::DrawID)) {
            return std::nullopt;
        }
        return Shader::IR::Value{static_cast<Shader::IR::Attribute>(frozen.payload)};
    case Type::Patch:
        if (frozen.payload > static_cast<u64>(Shader::IR::Patch::Component119)) {
            return std::nullopt;
        }
        return Shader::IR::Value{static_cast<Shader::IR::Patch>(frozen.payload)};
    case Type::U1:
        if (frozen.payload > 1) {
            return std::nullopt;
        }
        return Shader::IR::Value{frozen.payload != 0};
    case Type::U8:
        if (frozen.payload > std::numeric_limits<u8>::max()) {
            return std::nullopt;
        }
        return Shader::IR::Value{static_cast<u8>(frozen.payload)};
    case Type::U16:
        if (frozen.payload > std::numeric_limits<u16>::max()) {
            return std::nullopt;
        }
        return Shader::IR::Value{static_cast<u16>(frozen.payload)};
    case Type::U32:
        if (frozen.payload > std::numeric_limits<u32>::max()) {
            return std::nullopt;
        }
        return Shader::IR::Value{static_cast<u32>(frozen.payload)};
    case Type::U64:
        return Shader::IR::Value{frozen.payload};
    case Type::F32:
        if (frozen.payload > std::numeric_limits<u32>::max()) {
            return std::nullopt;
        }
        return Shader::IR::Value{std::bit_cast<f32>(static_cast<u32>(frozen.payload))};
    case Type::F64:
        return Shader::IR::Value{std::bit_cast<f64>(frozen.payload)};
    default:
        return std::nullopt;
    }
}

[[nodiscard]] bool IsValidSyntaxType(Shader::IR::AbstractSyntaxNode::Type type) noexcept {
    return static_cast<u32>(type) <=
           static_cast<u32>(Shader::IR::AbstractSyntaxNode::Type::Unreachable);
}

void WriteValue(std::ofstream& file, const PrecacheFrozenValue& value) {
    Write(file, static_cast<u32>(value.type));
    Write(file, value.payload);
}

[[nodiscard]] std::optional<PrecacheFrozenValue> ReadValue(std::ifstream& file) {
    u32 type{};
    PrecacheFrozenValue value;
    Read(file, type);
    Read(file, value.payload);
    value.type = static_cast<Shader::IR::Type>(type);
    return IsSupportedValueType(value.type) ? std::optional{value} : std::nullopt;
}

void WriteArtifact(std::ofstream& file, const PrecacheFrontendArtifactRecord& record) {
    const auto& key{record.key};
    const auto& artifact{record.artifact};
    Write(file, key.program_identity);
    Write(file, key.compiler_target_fingerprint);
    Write(file, key.source_header);
    Write(file, key.local_memory_size);
    Write(file, static_cast<u32>(key.stage));
    Write(file, key.scheduler_slot);
    Write(file, static_cast<u8>(key.exits_to_dispatcher));
    Write(file, artifact.workgroup_size[0]);
    Write(file, artifact.workgroup_size[1]);
    Write(file, artifact.workgroup_size[2]);
    Write(file, static_cast<u32>(artifact.output_topology));
    Write(file, artifact.output_vertices);
    Write(file, artifact.invocations);
    Write(file, artifact.shared_memory_size);
    Write(file, static_cast<u8>(artifact.is_geometry_passthrough));
    Write(file, artifact.frontend_dependencies.flags);
    Write(file, static_cast<u32>(artifact.blocks.size()));
    for (const auto& block : artifact.blocks) {
        Write(file, block.order);
        Write(file, static_cast<u32>(block.successors.size()));
        for (u32 successor : block.successors) Write(file, successor);
        Write(file, static_cast<u32>(block.instructions.size()));
        for (const auto& instruction : block.instructions) {
            Write(file, static_cast<u32>(instruction.opcode));
            Write(file, instruction.flags);
            Write(file, static_cast<u32>(instruction.args.size()));
            for (const auto& value : instruction.args) WriteValue(file, value);
            for (u32 predecessor : instruction.phi_predecessors) Write(file, predecessor);
        }
    }
    Write(file, static_cast<u32>(artifact.post_order_blocks.size()));
    for (u32 block : artifact.post_order_blocks) Write(file, block);
    Write(file, static_cast<u32>(artifact.syntax_list.size()));
    for (const auto& node : artifact.syntax_list) {
        Write(file, static_cast<u32>(node.type));
        WriteValue(file, node.condition);
        for (u32 block : node.blocks) Write(file, block);
    }
}

[[nodiscard]] std::optional<PrecacheFrontendArtifactRecord> ReadArtifact(std::ifstream& file) {
    PrecacheFrontendArtifactRecord record;
    u32 stage{}, topology{}, block_count{};
    u8 exits{}, geometry{};
    Read(file, record.key.program_identity);
    Read(file, record.key.compiler_target_fingerprint);
    Read(file, record.key.source_header);
    Read(file, record.key.local_memory_size);
    Read(file, stage);
    Read(file, record.key.scheduler_slot);
    Read(file, exits);
    if (exits > 1) return std::nullopt;
    record.key.stage = static_cast<Shader::Stage>(stage);
    record.key.exits_to_dispatcher = exits != 0;
    auto& artifact{record.artifact};
    artifact.stage = record.key.stage;
    artifact.local_memory_size = record.key.local_memory_size;
    Read(file, artifact.workgroup_size[0]);
    Read(file, artifact.workgroup_size[1]);
    Read(file, artifact.workgroup_size[2]);
    Read(file, topology);
    Read(file, artifact.output_vertices);
    Read(file, artifact.invocations);
    Read(file, artifact.shared_memory_size);
    Read(file, geometry);
    Read(file, artifact.frontend_dependencies.flags);
    Read(file, block_count);
    if (!IsValidKey(record.key) || geometry > 1 || block_count == 0 || block_count > kMaxBlocks) {
        return std::nullopt;
    }
    artifact.output_topology = static_cast<Shader::OutputTopology>(topology);
    artifact.is_geometry_passthrough = geometry != 0;
    artifact.blocks.resize(block_count);
    size_t instruction_count{};
    for (auto& block : artifact.blocks) {
        u32 successor_count{}, instruction_count_in_block{};
        Read(file, block.order);
        Read(file, successor_count);
        if (successor_count > block_count) return std::nullopt;
        block.successors.resize(successor_count);
        for (u32& successor : block.successors) Read(file, successor);
        Read(file, instruction_count_in_block);
        if (instruction_count_in_block > kMaxInstructions - instruction_count) return std::nullopt;
        instruction_count += instruction_count_in_block;
        block.instructions.resize(instruction_count_in_block);
        for (auto& instruction : block.instructions) {
            u32 opcode{}, arg_count{};
            Read(file, opcode);
            Read(file, instruction.flags);
            Read(file, arg_count);
            instruction.opcode = static_cast<Shader::IR::Opcode>(opcode);
            if (instruction.opcode == Shader::IR::Opcode::Phi ? arg_count > block_count
                                                               : arg_count > 5) {
                return std::nullopt;
            }
            instruction.args.reserve(arg_count);
            for (u32 index = 0; index < arg_count; ++index) {
                const auto value{ReadValue(file)};
                if (!value) return std::nullopt;
                instruction.args.push_back(*value);
            }
            if (instruction.opcode == Shader::IR::Opcode::Phi) {
                instruction.phi_predecessors.resize(arg_count);
                for (u32& predecessor : instruction.phi_predecessors) Read(file, predecessor);
            }
        }
    }
    u32 post_order_count{}, syntax_count{};
    Read(file, post_order_count);
    if (post_order_count > block_count) return std::nullopt;
    artifact.post_order_blocks.resize(post_order_count);
    for (u32& block : artifact.post_order_blocks) Read(file, block);
    Read(file, syntax_count);
    if (syntax_count > kMaxSyntaxNodes) return std::nullopt;
    artifact.syntax_list.resize(syntax_count);
    for (auto& node : artifact.syntax_list) {
        u32 type{};
        Read(file, type);
        const auto condition{ReadValue(file)};
        if (!condition) return std::nullopt;
        node.type = static_cast<Shader::IR::AbstractSyntaxNode::Type>(type);
        node.condition = *condition;
        for (u32& block : node.blocks) Read(file, block);
    }
    return record.IsValid() ? std::optional{std::move(record)} : std::nullopt;
}

} // namespace

size_t PrecacheFrontendArtifactKeyHash::operator()(const PrecacheFrontendArtifactKey& key) const
    noexcept {
    size_t hash{static_cast<size_t>(key.program_identity)};
    const auto combine = [&hash](size_t value) {
        hash ^= value + 0x9e3779b9U + (hash << 6) + (hash >> 2);
    };
    combine(static_cast<size_t>(key.compiler_target_fingerprint));
    combine(static_cast<size_t>(key.source_header));
    combine(static_cast<size_t>(key.local_memory_size));
    combine(static_cast<size_t>(key.stage));
    combine(static_cast<size_t>(key.scheduler_slot));
    combine(static_cast<size_t>(key.exits_to_dispatcher));
    return hash;
}

bool PrecacheFrontendArtifact::IsValid() const noexcept {
    if (!IsValidStage(stage) || !IsValidOutputTopology(output_topology) || blocks.empty() ||
        blocks.size() > kMaxBlocks ||
        syntax_list.size() > kMaxSyntaxNodes) {
        return false;
    }
    size_t instruction_count{};
    for (const auto& block : blocks) {
        instruction_count += block.instructions.size();
        if (instruction_count > kMaxInstructions) {
            return false;
        }
    }
    for (const auto& block : blocks) {
        for (u32 successor : block.successors) {
            if (!IsValidBlockId(successor, blocks.size())) {
                return false;
            }
        }
        for (const auto& instruction : block.instructions) {
            if (!IsValidOpcode(instruction.opcode) ||
                (instruction.opcode == Shader::IR::Opcode::Phi
                     ? instruction.args.size() != instruction.phi_predecessors.size()
                     : instruction.args.size() != Shader::IR::NumArgsOf(instruction.opcode)) ||
                (instruction.opcode == Shader::IR::Opcode::Phi
                     ? instruction.args.size() > blocks.size()
                     : instruction.args.size() > 5) ||
                (instruction.opcode == Shader::IR::Opcode::Phi &&
                 !IsSupportedValueType(static_cast<Shader::IR::Type>(instruction.flags)))) {
                return false;
            }
            for (const auto& value : instruction.args) {
                if (!IsValidFrozenValue(value, instruction_count)) {
                    return false;
                }
            }
            for (u32 predecessor : instruction.phi_predecessors) {
                if (!IsValidBlockId(predecessor, blocks.size())) {
                    return false;
                }
            }
        }
    }
    for (u32 block : post_order_blocks) {
        if (!IsValidBlockId(block, blocks.size())) {
            return false;
        }
    }
    for (const auto& node : syntax_list) {
        if (!IsValidSyntaxType(node.type) ||
            !IsValidFrozenValue(node.condition, instruction_count)) {
            return false;
        }
        for (u32 block : node.blocks) {
            if (block != kInvalidId && !IsValidBlockId(block, blocks.size())) {
                return false;
            }
        }
        switch (node.type) {
        case Shader::IR::AbstractSyntaxNode::Type::If:
        case Shader::IR::AbstractSyntaxNode::Type::Repeat:
        case Shader::IR::AbstractSyntaxNode::Type::Break:
            if (node.condition.type != Shader::IR::Type::U1 &&
                node.condition.type != Shader::IR::Type::Opaque) {
                return false;
            }
            break;
        default:
            if (node.condition.type != Shader::IR::Type::Void) {
                return false;
            }
            break;
        }
    }
    return true;
}

bool PrecacheFrontendArtifactRecord::IsValid() const noexcept {
    return IsValidKey(key) && artifact.IsValid() && artifact.stage == key.stage &&
           artifact.local_memory_size == key.local_memory_size;
}

std::optional<PrecacheFrontendArtifact> FreezePrecacheFrontendArtifact(
    const Shader::IR::Program& program, PrecacheFrontendFreezeError& error) {
    error = PrecacheFrontendFreezeError::None;
    if (program.info != Shader::Info{} || program.blocks.empty() ||
        program.blocks.size() > kMaxBlocks) {
        error = PrecacheFrontendFreezeError::UnsupportedInfo;
        return std::nullopt;
    }
    std::unordered_map<const Shader::IR::Block*, u32> block_ids;
    block_ids.reserve(program.blocks.size());
    std::unordered_map<const Shader::IR::Inst*, u32> instruction_ids;
    size_t instruction_count{};
    for (u32 block_id = 0; block_id < program.blocks.size(); ++block_id) {
        const auto [_, inserted]{block_ids.emplace(program.blocks[block_id], block_id)};
        if (!inserted || program.blocks[block_id] == nullptr) {
            error = PrecacheFrontendFreezeError::InvalidGraph;
            return std::nullopt;
        }
        for (const Shader::IR::Inst& inst : *program.blocks[block_id]) {
            if (++instruction_count > kMaxInstructions ||
                !instruction_ids.emplace(&inst, static_cast<u32>(instruction_ids.size())).second) {
                error = PrecacheFrontendFreezeError::TooLarge;
                return std::nullopt;
            }
            if (inst.HasAssociatedPseudoOperation()) {
                error = PrecacheFrontendFreezeError::AssociatedPseudoOperation;
                return std::nullopt;
            }
        }
    }
    PrecacheFrontendArtifact artifact{
        .stage = program.stage,
        .workgroup_size = program.workgroup_size,
        .output_topology = program.output_topology,
        .output_vertices = program.output_vertices,
        .invocations = program.invocations,
        .local_memory_size = program.local_memory_size,
        .shared_memory_size = program.shared_memory_size,
        .is_geometry_passthrough = program.is_geometry_passthrough,
        .frontend_dependencies = program.frontend_dependencies,
    };
    artifact.blocks.resize(program.blocks.size());
    for (u32 block_id = 0; block_id < program.blocks.size(); ++block_id) {
        const Shader::IR::Block& source{*program.blocks[block_id]};
        auto& destination{artifact.blocks[block_id]};
        destination.order = source.GetOrder();
        for (Shader::IR::Block* successor : source.ImmSuccessors()) {
            const auto it{block_ids.find(successor)};
            if (it == block_ids.end()) {
                error = PrecacheFrontendFreezeError::InvalidGraph;
                return std::nullopt;
            }
            destination.successors.push_back(it->second);
        }
        for (const Shader::IR::Inst& inst : source) {
            auto& frozen{destination.instructions.emplace_back()};
            frozen.opcode = inst.GetOpcode();
            frozen.flags = inst.Flags<u32>();
            for (size_t index = 0; index < inst.NumArgs(); ++index) {
                const auto value{FreezeValue(inst.Arg(index), instruction_ids)};
                if (!value) {
                    error = PrecacheFrontendFreezeError::UnsupportedValue;
                    return std::nullopt;
                }
                frozen.args.push_back(*value);
                if (inst.GetOpcode() == Shader::IR::Opcode::Phi) {
                    const auto it{block_ids.find(inst.PhiBlock(index))};
                    if (it == block_ids.end()) {
                        error = PrecacheFrontendFreezeError::InvalidGraph;
                        return std::nullopt;
                    }
                    frozen.phi_predecessors.push_back(it->second);
                }
            }
        }
    }
    for (Shader::IR::Block* block : program.post_order_blocks) {
        const auto it{block_ids.find(block)};
        if (it == block_ids.end()) {
            error = PrecacheFrontendFreezeError::InvalidGraph;
            return std::nullopt;
        }
        artifact.post_order_blocks.push_back(it->second);
    }
    for (const auto& source : program.syntax_list) {
        PrecacheFrozenSyntaxNode frozen{.type = source.type,
                                        .blocks = {kInvalidId, kInvalidId, kInvalidId}};
        const auto add_block = [&](size_t index, Shader::IR::Block* block) -> bool {
            const auto it{block_ids.find(block)};
            if (it == block_ids.end()) {
                return false;
            }
            frozen.blocks[index] = it->second;
            return true;
        };
        switch (source.type) {
        case Shader::IR::AbstractSyntaxNode::Type::Block:
            if (!add_block(0, source.data.block)) {
                error = PrecacheFrontendFreezeError::InvalidGraph;
                return std::nullopt;
            }
            break;
        case Shader::IR::AbstractSyntaxNode::Type::If:
            if (!add_block(0, source.data.if_node.body) || !add_block(1, source.data.if_node.merge)) {
                error = PrecacheFrontendFreezeError::InvalidGraph;
                return std::nullopt;
            }
            if (const auto value{FreezeValue(source.data.if_node.cond, instruction_ids)}) {
                frozen.condition = *value;
            } else {
                error = PrecacheFrontendFreezeError::UnsupportedValue;
                return std::nullopt;
            }
            break;
        case Shader::IR::AbstractSyntaxNode::Type::EndIf:
            if (!add_block(0, source.data.end_if.merge)) {
                error = PrecacheFrontendFreezeError::InvalidGraph;
                return std::nullopt;
            }
            break;
        case Shader::IR::AbstractSyntaxNode::Type::Loop:
            if (!add_block(0, source.data.loop.body) || !add_block(1, source.data.loop.continue_block) ||
                !add_block(2, source.data.loop.merge)) {
                error = PrecacheFrontendFreezeError::InvalidGraph;
                return std::nullopt;
            }
            break;
        case Shader::IR::AbstractSyntaxNode::Type::Repeat:
            if (!add_block(0, source.data.repeat.loop_header) || !add_block(1, source.data.repeat.merge)) {
                error = PrecacheFrontendFreezeError::InvalidGraph;
                return std::nullopt;
            }
            if (const auto value{FreezeValue(source.data.repeat.cond, instruction_ids)}) {
                frozen.condition = *value;
            } else {
                error = PrecacheFrontendFreezeError::UnsupportedValue;
                return std::nullopt;
            }
            break;
        case Shader::IR::AbstractSyntaxNode::Type::Break:
            if (!add_block(0, source.data.break_node.merge) || !add_block(1, source.data.break_node.skip)) {
                error = PrecacheFrontendFreezeError::InvalidGraph;
                return std::nullopt;
            }
            if (const auto value{FreezeValue(source.data.break_node.cond, instruction_ids)}) {
                frozen.condition = *value;
            } else {
                error = PrecacheFrontendFreezeError::UnsupportedValue;
                return std::nullopt;
            }
            break;
        case Shader::IR::AbstractSyntaxNode::Type::Return:
        case Shader::IR::AbstractSyntaxNode::Type::Unreachable:
            break;
        }
        artifact.syntax_list.push_back(std::move(frozen));
    }
    if (!artifact.IsValid()) {
        error = PrecacheFrontendFreezeError::InvalidGraph;
        return std::nullopt;
    }
    return artifact;
}

std::optional<Shader::IR::Program> RestorePrecacheFrontendArtifact(
    const PrecacheFrontendArtifact& artifact, Shader::ObjectPool<Shader::IR::Inst>& inst_pool,
    Shader::ObjectPool<Shader::IR::Block>& block_pool) {
    if (!artifact.IsValid()) {
        return std::nullopt;
    }
    Shader::IR::Program program{
        .stage = artifact.stage,
        .workgroup_size = artifact.workgroup_size,
        .output_topology = artifact.output_topology,
        .output_vertices = artifact.output_vertices,
        .invocations = artifact.invocations,
        .local_memory_size = artifact.local_memory_size,
        .shared_memory_size = artifact.shared_memory_size,
        .is_geometry_passthrough = artifact.is_geometry_passthrough,
        .frontend_dependencies = artifact.frontend_dependencies,
    };
    program.blocks.reserve(artifact.blocks.size());
    std::vector<Shader::IR::Inst*> instructions;
    for (const auto& frozen_block : artifact.blocks) {
        Shader::IR::Block* block{block_pool.Create(inst_pool)};
        block->SetOrder(frozen_block.order);
        program.blocks.push_back(block);
        for (const auto& frozen_inst : frozen_block.instructions) {
            instructions.push_back(&block->AppendNewInstUnbound(frozen_inst.opcode, frozen_inst.flags));
        }
    }
    size_t instruction_id{};
    for (const auto& frozen_block : artifact.blocks) {
        for (const auto& frozen_inst : frozen_block.instructions) {
            Shader::IR::Inst& inst{*instructions[instruction_id++]};
            for (size_t index = 0; index < frozen_inst.args.size(); ++index) {
                const auto value{RestoreValue(frozen_inst.args[index], instructions)};
                if (!value) {
                    return std::nullopt;
                }
                if (frozen_inst.opcode == Shader::IR::Opcode::Phi) {
                    inst.AddPhiOperand(program.blocks[frozen_inst.phi_predecessors[index]], *value);
                } else {
                    inst.SetArg(index, *value);
                }
            }
        }
    }
    for (size_t block_id = 0; block_id < artifact.blocks.size(); ++block_id) {
        for (u32 successor : artifact.blocks[block_id].successors) {
            program.blocks[block_id]->AddBranch(program.blocks[successor]);
        }
    }
    for (u32 block : artifact.post_order_blocks) {
        program.post_order_blocks.push_back(program.blocks[block]);
    }
    for (const auto& frozen : artifact.syntax_list) {
        Shader::IR::AbstractSyntaxNode node{.type = frozen.type};
        const auto block = [&](size_t index) { return program.blocks[frozen.blocks[index]]; };
        switch (frozen.type) {
        case Shader::IR::AbstractSyntaxNode::Type::Block:
            node.data.block = block(0);
            break;
        case Shader::IR::AbstractSyntaxNode::Type::If: {
            const auto condition{RestoreValue(frozen.condition, instructions)};
            if (!condition) return std::nullopt;
            node.data.if_node = {.cond = Shader::IR::U1{*condition}, .body = block(0), .merge = block(1)};
            break;
        }
        case Shader::IR::AbstractSyntaxNode::Type::EndIf:
            node.data.end_if = {.merge = block(0)};
            break;
        case Shader::IR::AbstractSyntaxNode::Type::Loop:
            node.data.loop = {.body = block(0), .continue_block = block(1), .merge = block(2)};
            break;
        case Shader::IR::AbstractSyntaxNode::Type::Repeat: {
            const auto condition{RestoreValue(frozen.condition, instructions)};
            if (!condition) return std::nullopt;
            node.data.repeat = {.cond = Shader::IR::U1{*condition}, .loop_header = block(0), .merge = block(1)};
            break;
        }
        case Shader::IR::AbstractSyntaxNode::Type::Break: {
            const auto condition{RestoreValue(frozen.condition, instructions)};
            if (!condition) return std::nullopt;
            node.data.break_node = {.cond = Shader::IR::U1{*condition}, .merge = block(0), .skip = block(1)};
            break;
        }
        case Shader::IR::AbstractSyntaxNode::Type::Return:
        case Shader::IR::AbstractSyntaxNode::Type::Unreachable:
            break;
        }
        program.syntax_list.push_back(node);
    }
    return program;
}

bool SavePrecacheFrontendArtifacts(const std::filesystem::path& filename,
                                   std::span<const PrecacheFrontendArtifactRecord> artifacts) {
    if (artifacts.empty() || artifacts.size() > kMaxArtifacts) return false;
    std::vector<PrecacheFrontendArtifactRecord> ordered{artifacts.begin(), artifacts.end()};
    std::ranges::sort(ordered, [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.key.program_identity, lhs.key.compiler_target_fingerprint,
                        lhs.key.source_header, lhs.key.local_memory_size, lhs.key.stage,
                        lhs.key.scheduler_slot, lhs.key.exits_to_dispatcher) <
               std::tie(rhs.key.program_identity, rhs.key.compiler_target_fingerprint,
                        rhs.key.source_header, rhs.key.local_memory_size, rhs.key.stage,
                        rhs.key.scheduler_slot, rhs.key.exits_to_dispatcher);
    });
    std::unordered_set<PrecacheFrontendArtifactKey, PrecacheFrontendArtifactKeyHash> keys;
    keys.reserve(ordered.size());
    for (const auto& record : ordered) {
        if (!record.IsValid() || !keys.insert(record.key).second) return false;
    }
    const auto temporary = filename.parent_path() / (filename.filename().string() + ".tmp");
    try {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) return false;
        file.exceptions(std::ofstream::failbit);
        file.write(kMagic.data(), kMagic.size());
        Write(file, kVersion);
        Write(file, static_cast<u32>(ordered.size()));
        for (const auto& record : ordered) WriteArtifact(file, record);
        file.close();
    } catch (...) {
        std::error_code ec;
        std::filesystem::remove(temporary, ec);
        return false;
    }
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
    if (had_previous) std::filesystem::remove(backup, ec);
    return true;
}

std::optional<std::vector<PrecacheFrontendArtifactRecord>> LoadPrecacheFrontendArtifacts(
    const std::filesystem::path& filename) {
    try {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            const auto backup = filename.parent_path() / (filename.filename().string() + ".bak");
            file.clear();
            file.open(backup, std::ios::binary);
            if (!file.is_open()) return std::nullopt;
        }
        file.exceptions(std::ifstream::failbit);
        std::array<char, 8> magic{};
        u32 version{}, count{};
        file.read(magic.data(), magic.size());
        Read(file, version);
        Read(file, count);
        if (magic != kMagic || version != kVersion || count == 0 || count > kMaxArtifacts) {
            return std::nullopt;
        }
        std::vector<PrecacheFrontendArtifactRecord> artifacts;
        artifacts.reserve(count);
        std::unordered_set<PrecacheFrontendArtifactKey, PrecacheFrontendArtifactKeyHash> keys;
        keys.reserve(count);
        for (u32 index = 0; index < count; ++index) {
            auto record{ReadArtifact(file)};
            if (!record || !keys.insert(record->key).second) return std::nullopt;
            artifacts.push_back(std::move(*record));
        }
        if (file.peek() != std::char_traits<char>::eof()) return std::nullopt;
        return artifacts;
    } catch (...) {
        return std::nullopt;
    }
}

} // namespace VideoCommon
