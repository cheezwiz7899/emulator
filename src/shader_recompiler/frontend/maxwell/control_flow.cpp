// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include <fmt/format.h>

#include <ranges>
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/decode.h"
#include "shader_recompiler/frontend/maxwell/indirect_branch_table_track.h"
#include "shader_recompiler/frontend/maxwell/location.h"

namespace Shader::Maxwell::Flow {
namespace {
struct Compare {
    bool operator()(const Block& lhs, Location rhs) const noexcept {
        return lhs.begin < rhs;
    }

    bool operator()(Location lhs, const Block& rhs) const noexcept {
        return lhs < rhs.begin;
    }

    bool operator()(const Block& lhs, const Block& rhs) const noexcept {
        return lhs.begin < rhs.begin;
    }
};

u32 BranchOffset(Location pc, Instruction inst) {
    return pc.Offset() + static_cast<u32>(inst.branch.Offset()) + 8u;
}

void Split(Block* old_block, Block* new_block, Location pc) {
    if (pc <= old_block->begin || pc >= old_block->end) {
        throw InvalidArgument("Invalid address to split={}", pc);
    }
    *new_block = Block{};
    new_block->begin = pc;
    new_block->end = old_block->end;
    new_block->end_class = old_block->end_class;
    new_block->cond = old_block->cond;
    new_block->stack = old_block->stack;
    new_block->branch_true = old_block->branch_true;
    new_block->branch_false = old_block->branch_false;
    new_block->function_call = old_block->function_call;
    new_block->return_block = old_block->return_block;
    new_block->branch_reg = old_block->branch_reg;
    new_block->branch_offset = old_block->branch_offset;
    new_block->indirect_branches = std::move(old_block->indirect_branches);

    const Location old_begin{old_block->begin};
    Stack old_stack{std::move(old_block->stack)};
    *old_block = Block{};
    old_block->begin = old_begin;
    old_block->end = pc;
    old_block->end_class = EndClass::Branch;
    old_block->cond = IR::Condition(true);
    old_block->stack = old_stack;
    old_block->branch_true = new_block;
    old_block->branch_false = nullptr;
}

Token OpcodeToken(Opcode opcode) {
    switch (opcode) {
    case Opcode::PBK:
    case Opcode::BRK:
        return Token::PBK;
    case Opcode::PCNT:
    case Opcode::CONT:
        return Token::PCNT;
    case Opcode::PEXIT:
    case Opcode::EXIT:
        return Token::PEXIT;
    case Opcode::PLONGJMP:
    case Opcode::LONGJMP:
        return Token::PLONGJMP;
    case Opcode::PRET:
    case Opcode::RET:
    case Opcode::CAL:
        return Token::PRET;
    case Opcode::SSY:
    case Opcode::SYNC:
        return Token::SSY;
    default:
        throw InvalidArgument("{}", opcode);
    }
}

bool IsAbsoluteJump(Opcode opcode) {
    switch (opcode) {
    case Opcode::JCAL:
    case Opcode::JMP:
    case Opcode::JMX:
        return true;
    default:
        return false;
    }
}

bool HasFlowTest(Opcode opcode) {
    switch (opcode) {
    case Opcode::BRA:
    case Opcode::BRX:
    case Opcode::EXIT:
    case Opcode::JMP:
    case Opcode::JMX:
    case Opcode::KIL:
    case Opcode::BRK:
    case Opcode::CONT:
    case Opcode::LONGJMP:
    case Opcode::RET:
    case Opcode::SYNC:
        return true;
    case Opcode::CAL:
    case Opcode::JCAL:
        return false;
    default:
        throw InvalidArgument("Invalid branch {}", opcode);
    }
}

std::string NameOf(const Block& block) {
    if (block.begin.IsVirtual()) {
        return fmt::format("\"Virtual {}\"", block.begin);
    } else {
        return fmt::format("\"{}\"", block.begin);
    }
}
} // Anonymous namespace

void Stack::Push(Token token, Location target) {
    entries.push_back({
        .token = token,
        .target{target},
    });
}

std::pair<Location, Stack> Stack::Pop(Token token) const {
    const std::optional<Location> pc{Peek(token)};
    if (!pc) {
        throw LogicError("Token could not be found");
    }
    return {*pc, Remove(token)};
}

std::optional<Location> Stack::Peek(Token token) const {
    const auto it{std::find_if(entries.rbegin(), entries.rend(),
                               [token](const auto& entry) { return entry.token == token; })};
    if (it == entries.rend()) {
        return std::nullopt;
    }
    return it->target;
}

Stack Stack::Remove(Token token) const {
    const auto it{std::find_if(entries.rbegin(), entries.rend(),
                               [token](const auto& entry) { return entry.token == token; })};
    const auto pos{std::distance(entries.rbegin(), it)};
    Stack result;
    result.entries.insert(result.entries.end(), entries.begin(), entries.end() - pos - 1);
    return result;
}

bool Block::Contains(Location pc) const noexcept {
    return pc >= begin && pc < end;
}

Function::Function(ObjectPool<Block>& block_pool, Location start_address)
    : entrypoint{start_address} {
    Label& label{labels.emplace_back()};
    label.address = start_address;
    label.block = block_pool.Create(Block{});
    label.block->begin = start_address;
    label.block->end = start_address;
    label.block->end_class = EndClass::Branch;
    label.block->cond = IR::Condition(true);
    label.block->branch_true = nullptr;
    label.block->branch_false = nullptr;
}

CFG::CFG(Environment& env_, ObjectPool<Block>& block_pool_, Location start_address,
         bool exits_to_dispatcher_)
    : env{env_}, block_pool{block_pool_}, program_start{start_address}, exits_to_dispatcher{
                                                                            exits_to_dispatcher_} {
    if (exits_to_dispatcher) {
        dispatch_block = block_pool.Create(Block{});
        dispatch_block->begin = {};
        dispatch_block->end = {};
        dispatch_block->end_class = EndClass::Exit;
        dispatch_block->cond = IR::Condition(true);
        dispatch_block->stack = {};
        dispatch_block->branch_true = nullptr;
        dispatch_block->branch_false = nullptr;
    }
    functions.emplace_back(block_pool, start_address);
    for (FunctionId function_id = 0; function_id < functions.size(); ++function_id) {
        while (!functions[function_id].labels.empty()) {
            Function& function{functions[function_id]};
            Label label{function.labels.back()};
            function.labels.pop_back();
            AnalyzeLabel(function_id, label);
        }
    }
    if (exits_to_dispatcher) {
        const auto last_block{functions[0].blocks.rbegin()};
        dispatch_block->begin = last_block->end + 1;
        dispatch_block->end = last_block->end + 1;
        functions[0].blocks.insert(*dispatch_block);
    }
}

bool CFG::IsValidTemplate(const Template& source) {
    if (source.functions.empty() || source.blocks.empty()) {
        return false;
    }
    std::vector<bool> seen_blocks(source.blocks.size());
    const auto valid_block = [&source](size_t id) {
        return id == NoTemplateBlock || id < source.blocks.size();
    };
    for (FunctionId function_id = 0; function_id < source.functions.size(); ++function_id) {
        const TemplateFunction& function{source.functions[function_id]};
        if (!Location::IsRawOffset(function.entrypoint.Offset()) || function.blocks.empty()) {
            return false;
        }
        bool owns_entrypoint{};
        for (const size_t block_id : function.blocks) {
            if (block_id >= source.blocks.size() || seen_blocks[block_id] ||
                source.blocks[block_id].owner != function_id) {
                return false;
            }
            seen_blocks[block_id] = true;
            owns_entrypoint |= source.blocks[block_id].begin == function.entrypoint;
        }
        // Function construction begins with a label at its entrypoint. A
        // persisted form whose function has no corresponding entry block can
        // reconstruct an intrusive graph that looks valid but is unreachable
        // or has stale control-flow ownership.
        if (!owns_entrypoint) {
            return false;
        }
    }
    if (std::find(seen_blocks.begin(), seen_blocks.end(), false) != seen_blocks.end()) {
        return false;
    }
    for (const TemplateBlock& block : source.blocks) {
        const auto [pred, pred_negated] = block.cond.GetPred();
        static_cast<void>(pred_negated);
        if (!Location::IsRawOffset(block.begin.Offset()) || !Location::IsRawOffset(block.end.Offset()) ||
            block.end < block.begin || block.owner >= source.functions.size() ||
            block.function_call >= source.functions.size() || !valid_block(block.branch_true) ||
            !valid_block(block.branch_false) || !valid_block(block.return_block) ||
            static_cast<u64>(pred) > static_cast<u64>(IR::Pred::PT) ||
            static_cast<u64>(block.cond.GetFlowTest()) > static_cast<u64>(IR::FlowTest::RGT) ||
            static_cast<u64>(block.branch_reg) > static_cast<u64>(IR::Reg::RZ)) {
            return false;
        }
        for (const StackEntry& entry : block.stack.Entries()) {
            if (!Location::IsRawOffset(entry.target.Offset()) ||
                static_cast<u8>(entry.token) > static_cast<u8>(Token::PLONGJMP)) {
                return false;
            }
        }
        for (const auto& [target, address] : block.indirect_branches) {
            static_cast<void>(address);
            if (!valid_block(target)) {
                return false;
            }
        }
    }
    if (source.exits_to_dispatcher) {
        const size_t dispatch_id{source.functions.front().blocks.back()};
        const TemplateBlock& dispatch{source.blocks[dispatch_id]};
        if (dispatch.end_class != EndClass::Exit || dispatch.branch_true != NoTemplateBlock ||
            dispatch.branch_false != NoTemplateBlock || !dispatch.indirect_branches.empty()) {
            return false;
        }
    }
    return true;
}

CFG::CFG(Environment& env_, ObjectPool<Block>& block_pool_, const Template& source)
    : env{env_}, block_pool{block_pool_},
      program_start{source.functions.empty() ? Location{} : source.functions.front().entrypoint},
      exits_to_dispatcher{source.exits_to_dispatcher} {
    if (!IsValidTemplate(source)) {
        throw InvalidArgument("Invalid CFG template");
    }

    functions.reserve(source.functions.size());
    for (const TemplateFunction& source_function : source.functions) {
        functions.emplace_back();
        functions.back().entrypoint = source_function.entrypoint;
    }

    std::vector<Block*> blocks;
    blocks.reserve(source.blocks.size());
    for (const TemplateBlock& source_block : source.blocks) {
        if (source_block.owner >= functions.size()) {
            throw InvalidArgument("Invalid CFG template block owner");
        }
        Block* const block{block_pool.Create(Block{})};
        block->begin = source_block.begin;
        block->end = source_block.end;
        block->end_class = source_block.end_class;
        block->cond = source_block.cond;
        block->stack = source_block.stack;
        block->function_call = source_block.function_call;
        block->branch_reg = source_block.branch_reg;
        block->branch_offset = source_block.branch_offset;
        blocks.push_back(block);
    }

    const auto resolve_block = [&blocks](size_t index) -> Block* {
        if (index == NoTemplateBlock) {
            return nullptr;
        }
        if (index >= blocks.size()) {
            throw InvalidArgument("Invalid CFG template block reference");
        }
        return blocks[index];
    };
    for (size_t index = 0; index < source.blocks.size(); ++index) {
        const TemplateBlock& source_block{source.blocks[index]};
        Block& block{*blocks[index]};
        if (block.function_call >= functions.size()) {
            throw InvalidArgument("Invalid CFG template function reference");
        }
        block.branch_true = resolve_block(source_block.branch_true);
        block.branch_false = resolve_block(source_block.branch_false);
        block.return_block = resolve_block(source_block.return_block);
        block.indirect_branches.reserve(source_block.indirect_branches.size());
        for (const auto& [target, address] : source_block.indirect_branches) {
            block.indirect_branches.push_back({resolve_block(target), address});
        }
    }

    for (size_t function_id = 0; function_id < source.functions.size(); ++function_id) {
        for (const size_t block_id : source.functions[function_id].blocks) {
            if (block_id >= blocks.size() || source.blocks[block_id].owner != function_id) {
                throw InvalidArgument("Invalid CFG template function block list");
            }
            functions[function_id].blocks.insert(*blocks[block_id]);
        }
    }
    if (exits_to_dispatcher) {
        dispatch_block = blocks[source.functions.front().blocks.back()];
    }
}

bool CFG::RoundTripMatchesTemplate(Environment& env, ObjectPool<Block>& block_pool,
                                   const Template& source) {
    if (!IsValidTemplate(source)) {
        return false;
    }
    CFG replay{env, block_pool, source};
    return replay.MakeTemplate() == source;
}

std::optional<CFG::Template> CFG::RebaseTemplate(const Template& source, Location source_origin,
                                                  Location destination_origin,
                                                  bool require_matching_scheduler_phase) {
    if (!IsValidTemplate(source) || source_origin.IsVirtual() || destination_origin.IsVirtual() ||
        source.functions.front().entrypoint != source_origin ||
        (require_matching_scheduler_phase &&
         source_origin.Offset() % 32 != destination_origin.Offset() % 32)) {
        return std::nullopt;
    }

    // A valid serialized Location alone does not prove it belongs to this
    // shader. Keep corrupt/out-of-range data bounded rather than turning a
    // failed cache record into billions of scheduler steps.
    constexpr u32 kMaxRebaseInstructions = 1U << 20;
    const auto instruction_index = [source_origin](Location location) -> std::optional<u32> {
        const bool is_virtual = location.IsVirtual();
        if (is_virtual) {
            if (location.Offset() > std::numeric_limits<u32>::max() - 4) {
                return std::nullopt;
            }
            location = Location::FromRawOffset(location.Offset() + 4);
        }
        Location cursor{source_origin};
        for (u32 index = 0; index < kMaxRebaseInstructions; ++index, ++cursor) {
            if (cursor == location) {
                return index;
            }
            if (cursor > location) {
                return std::nullopt;
            }
        }
        return std::nullopt;
    };
    const auto rebase_location = [&instruction_index, destination_origin](Location location)
        -> std::optional<Location> {
        const bool is_virtual = location.IsVirtual();
        const auto index = instruction_index(location);
        if (!index) {
            return std::nullopt;
        }
        Location rebased{destination_origin};
        for (u32 step = 0; step < *index; ++step) {
            ++rebased;
        }
        return is_virtual ? rebased.Virtual() : rebased;
    };

    Template result{source};
    for (TemplateFunction& function : result.functions) {
        const auto entrypoint = rebase_location(function.entrypoint);
        if (!entrypoint) {
            return std::nullopt;
        }
        function.entrypoint = *entrypoint;
    }
    for (TemplateBlock& block : result.blocks) {
        // BRX targets are raw guest addresses populated from runtime cbuf
        // data. A relative CFG artifact must model them separately, never
        // guess that an absolute value can be shifted with block Locations.
        if (!block.indirect_branches.empty()) {
            return std::nullopt;
        }
        const auto begin = rebase_location(block.begin);
        const auto end = rebase_location(block.end);
        if (!begin || !end) {
            return std::nullopt;
        }
        block.begin = *begin;
        block.end = *end;
        std::vector<StackEntry> entries{block.stack.Entries()};
        for (StackEntry& entry : entries) {
            const auto target = rebase_location(entry.target);
            if (!target) {
                return std::nullopt;
            }
            entry.target = *target;
        }
        block.stack.SetEntries(std::move(entries));
    }
    if (!IsValidTemplate(result)) {
        return std::nullopt;
    }
    return result;
}

CFG::Template CFG::MakeTemplate() const {
    Template result;
    result.exits_to_dispatcher = exits_to_dispatcher;

    std::unordered_map<const Block*, size_t> block_indices;
    for (FunctionId function_id = 0; function_id < functions.size(); ++function_id) {
        TemplateFunction& result_function{result.functions.emplace_back()};
        result_function.entrypoint = functions[function_id].entrypoint;
        for (const Block& block : functions[function_id].blocks) {
            const size_t block_id{result.blocks.size()};
            block_indices.emplace(&block, block_id);
            result_function.blocks.push_back(block_id);
            result.blocks.emplace_back();
        }
    }

    const auto index_of = [&block_indices](const Block* block) -> size_t {
        if (block == nullptr) {
            return NoTemplateBlock;
        }
        const auto it{block_indices.find(block)};
        if (it == block_indices.end()) {
            throw InvalidArgument("CFG template references block outside CFG");
        }
        return it->second;
    };
    for (FunctionId function_id = 0; function_id < functions.size(); ++function_id) {
        for (const Block& block : functions[function_id].blocks) {
            TemplateBlock& destination{result.blocks.at(block_indices.at(&block))};
            destination.begin = block.begin;
            destination.end = block.end;
            destination.end_class = block.end_class;
            destination.cond = block.cond;
            destination.stack = block.stack;
            destination.branch_true = index_of(block.branch_true);
            destination.branch_false = index_of(block.branch_false);
            destination.function_call = block.function_call;
            destination.return_block = index_of(block.return_block);
            destination.branch_reg = block.branch_reg;
            destination.branch_offset = block.branch_offset;
            destination.owner = function_id;
            destination.indirect_branches.reserve(block.indirect_branches.size());
            for (const IndirectBranch& branch : block.indirect_branches) {
                destination.indirect_branches.push_back({index_of(branch.block), branch.address});
            }
        }
    }
    if (!IsValidTemplate(result)) {
        throw LogicError("Generated invalid CFG template");
    }
    return result;
}

void CFG::AnalyzeLabel(FunctionId function_id, Label& label) {
    if (InspectVisitedBlocks(function_id, label)) {
        // Label address has been visited
        return;
    }
    // Try to find the next block
    Function* const function{&functions[function_id]};
    Location pc{label.address};
    const auto next_it{function->blocks.upper_bound(pc, Compare{})};
    const bool is_last{next_it == function->blocks.end()};
    Block* const next{is_last ? nullptr : &*next_it};
    // Insert before the next block
    Block* const block{label.block};
    // Analyze instructions until it reaches an already visited block or there's a branch
    bool is_branch{false};
    while (!next || pc < next->begin) {
        is_branch = AnalyzeInst(block, function_id, pc) == AnalysisState::Branch;
        if (is_branch) {
            break;
        }
        ++pc;
    }
    if (!is_branch) {
        // If the block finished without a branch,
        // it means that the next instruction is already visited, jump to it
        block->end = pc;
        block->cond = IR::Condition{true};
        block->branch_true = next;
        block->branch_false = nullptr;
    }
    // Function's pointer might be invalid, resolve it again
    // Insert the new block
    functions[function_id].blocks.insert(*block);
}

bool CFG::InspectVisitedBlocks(FunctionId function_id, const Label& label) {
    const Location pc{label.address};
    Function& function{functions[function_id]};
    const auto it{
        std::ranges::find_if(function.blocks, [pc](auto& block) { return block.Contains(pc); })};
    if (it == function.blocks.end()) {
        // Address has not been visited
        return false;
    }
    Block* const visited_block{&*it};
    if (visited_block->begin == pc) {
        throw LogicError("Dangling block");
    }
    Block* const new_block{label.block};
    Split(visited_block, new_block, pc);
    function.blocks.insert(it, *new_block);
    return true;
}

CFG::AnalysisState CFG::AnalyzeInst(Block* block, FunctionId function_id, Location pc) {
    const Instruction inst{env.ReadInstruction(pc.Offset())};
    const Opcode opcode{Decode(inst.raw)};
    switch (opcode) {
    case Opcode::BRA:
    case Opcode::JMP:
    case Opcode::RET:
        if (!AnalyzeBranch(block, function_id, pc, inst, opcode)) {
            return AnalysisState::Continue;
        }
        switch (opcode) {
        case Opcode::BRA:
        case Opcode::JMP:
            AnalyzeBRA(block, function_id, pc, inst, IsAbsoluteJump(opcode));
            break;
        case Opcode::RET:
            block->end_class = EndClass::Return;
            break;
        default:
            break;
        }
        block->end = pc;
        return AnalysisState::Branch;
    case Opcode::BRK:
    case Opcode::CONT:
    case Opcode::LONGJMP:
    case Opcode::SYNC: {
        if (!AnalyzeBranch(block, function_id, pc, inst, opcode)) {
            return AnalysisState::Continue;
        }
        const auto [stack_pc, new_stack]{block->stack.Pop(OpcodeToken(opcode))};
        block->branch_true = AddLabel(block, new_stack, stack_pc, function_id);
        block->end = pc;
        return AnalysisState::Branch;
    }
    case Opcode::KIL: {
        const Predicate pred{inst.Pred()};
        const auto ir_pred{static_cast<IR::Pred>(pred.index)};
        const IR::Condition cond{inst.branch.flow_test, ir_pred, pred.negated};
        AnalyzeCondInst(block, function_id, pc, EndClass::Kill, cond);
        return AnalysisState::Branch;
    }
    case Opcode::PBK:
    case Opcode::PCNT:
    case Opcode::PEXIT:
    case Opcode::PLONGJMP:
    case Opcode::SSY:
        block->stack.Push(OpcodeToken(opcode), BranchOffset(pc, inst));
        return AnalysisState::Continue;
    case Opcode::BRX:
    case Opcode::JMX:
        return AnalyzeBRX(block, pc, inst, IsAbsoluteJump(opcode), function_id);
    case Opcode::EXIT:
        return AnalyzeEXIT(block, function_id, pc, inst);
    case Opcode::PRET:
        throw NotImplementedException("PRET flow analysis");
    case Opcode::CAL:
    case Opcode::JCAL: {
        const bool is_absolute{IsAbsoluteJump(opcode)};
        const Location cal_pc{is_absolute ? inst.branch.Absolute() : BranchOffset(pc, inst)};
        // Technically CAL pushes into PRET, but that's implicit in the function call for us
        // Insert the function into the list if it doesn't exist
        const auto it{std::ranges::find(functions, cal_pc, &Function::entrypoint)};
        const bool exists{it != functions.end()};
        const FunctionId call_id{exists ? static_cast<size_t>(std::distance(functions.begin(), it))
                                        : functions.size()};
        if (!exists) {
            functions.emplace_back(block_pool, cal_pc);
        }
        block->end_class = EndClass::Call;
        block->function_call = call_id;
        block->return_block = AddLabel(block, block->stack, pc + 1, function_id);
        block->end = pc;
        return AnalysisState::Branch;
    }
    default:
        break;
    }
    const Predicate pred{inst.Pred()};
    if (pred == Predicate{true} || pred == Predicate{false}) {
        return AnalysisState::Continue;
    }
    const IR::Condition cond{static_cast<IR::Pred>(pred.index), pred.negated};
    AnalyzeCondInst(block, function_id, pc, EndClass::Branch, cond);
    return AnalysisState::Branch;
}

void CFG::AnalyzeCondInst(Block* block, FunctionId function_id, Location pc,
                          EndClass insn_end_class, IR::Condition cond) {
    if (block->begin != pc) {
        // If the block doesn't start in the conditional instruction
        // mark it as a label to visit it later
        block->end = pc;
        block->cond = IR::Condition{true};
        block->branch_true = AddLabel(block, block->stack, pc, function_id);
        block->branch_false = nullptr;
        return;
    }
    // Create a virtual block and a conditional block
    Block* const conditional_block{block_pool.Create()};
    Block virtual_block{};
    virtual_block.begin = block->begin.Virtual();
    virtual_block.end = block->begin.Virtual();
    virtual_block.end_class = EndClass::Branch;
    virtual_block.stack = block->stack;
    virtual_block.cond = cond;
    virtual_block.branch_true = conditional_block;
    virtual_block.branch_false = nullptr;
    // Save the contents of the visited block in the conditional block
    *conditional_block = std::move(*block);
    // Impersonate the visited block with a virtual block
    *block = std::move(virtual_block);
    // Set the end properties of the conditional instruction
    conditional_block->end = pc + 1;
    conditional_block->end_class = insn_end_class;
    // Add a label to the instruction after the conditional instruction
    Block* const endif_block{AddLabel(conditional_block, block->stack, pc + 1, function_id)};
    // Branch to the next instruction from the virtual block
    block->branch_false = endif_block;
    // And branch to it from the conditional instruction if it is a branch or a kill instruction
    // Kill instructions are considered a branch because they demote to a helper invocation and
    // execution may continue.
    if (insn_end_class == EndClass::Branch || insn_end_class == EndClass::Kill) {
        conditional_block->cond = IR::Condition{true};
        conditional_block->branch_true = endif_block;
        conditional_block->branch_false = nullptr;
    }
    // Finally insert the condition block into the list of blocks
    functions[function_id].blocks.insert(*conditional_block);
}

bool CFG::AnalyzeBranch(Block* block, FunctionId function_id, Location pc, Instruction inst,
                        Opcode opcode) {
    if (inst.branch.is_cbuf) {
        throw NotImplementedException("Branch with constant buffer offset");
    }
    const Predicate pred{inst.Pred()};
    if (pred == Predicate{false}) {
        return false;
    }
    const bool has_flow_test{HasFlowTest(opcode)};
    const IR::FlowTest flow_test{has_flow_test ? inst.branch.flow_test.Value() : IR::FlowTest::T};
    if (pred != Predicate{true} || flow_test != IR::FlowTest::T) {
        block->cond = IR::Condition(flow_test, static_cast<IR::Pred>(pred.index), pred.negated);
        block->branch_false = AddLabel(block, block->stack, pc + 1, function_id);
    } else {
        block->cond = IR::Condition{true};
    }
    return true;
}

void CFG::AnalyzeBRA(Block* block, FunctionId function_id, Location pc, Instruction inst,
                     bool is_absolute) {
    const Location bra_pc{is_absolute ? inst.branch.Absolute() : BranchOffset(pc, inst)};
    block->branch_true = AddLabel(block, block->stack, bra_pc, function_id);
}

CFG::AnalysisState CFG::AnalyzeBRX(Block* block, Location pc, Instruction inst, bool is_absolute,
                                   FunctionId function_id) {
    const std::optional brx_table{TrackIndirectBranchTable(env, pc, program_start)};
    if (!brx_table) {
        TrackIndirectBranchTable(env, pc, program_start);
        throw NotImplementedException("Failed to track indirect branch");
    }
    const IR::FlowTest flow_test{inst.branch.flow_test};
    const Predicate pred{inst.Pred()};
    if (flow_test != IR::FlowTest::T || pred != Predicate{true}) {
        throw NotImplementedException("Conditional indirect branch");
    }
    std::vector<u32> targets;
    targets.reserve(brx_table->num_entries);
    for (u32 i = 0; i < brx_table->num_entries; ++i) {
        u32 target{env.ReadCbufValue(brx_table->cbuf_index, brx_table->cbuf_offset + i * 4)};
        if (!is_absolute) {
            target += pc.Offset();
        }
        target += static_cast<u32>(brx_table->branch_offset);
        target += 8;
        targets.push_back(target);
    }
    std::ranges::sort(targets);
    targets.erase(std::unique(targets.begin(), targets.end()), targets.end());

    block->indirect_branches.reserve(targets.size());
    for (const u32 target : targets) {
        Block* const branch{AddLabel(block, block->stack, target, function_id)};
        block->indirect_branches.push_back({
            .block = branch,
            .address = target,
        });
    }
    block->cond = IR::Condition{true};
    block->end = pc + 1;
    block->end_class = EndClass::IndirectBranch;
    block->branch_reg = brx_table->branch_reg;
    block->branch_offset = brx_table->branch_offset + 8;
    if (!is_absolute) {
        block->branch_offset += pc.Offset();
    }
    return AnalysisState::Branch;
}

CFG::AnalysisState CFG::AnalyzeEXIT(Block* block, FunctionId function_id, Location pc,
                                    Instruction inst) {
    const IR::FlowTest flow_test{inst.branch.flow_test};
    const Predicate pred{inst.Pred()};
    if (pred == Predicate{false} || flow_test == IR::FlowTest::F) {
        // EXIT will never be taken
        return AnalysisState::Continue;
    }
    if (exits_to_dispatcher && function_id != 0) {
        throw NotImplementedException("Dispatch EXIT on external function");
    }
    if (pred != Predicate{true} || flow_test != IR::FlowTest::T) {
        if (block->stack.Peek(Token::PEXIT).has_value()) {
            throw NotImplementedException("Conditional EXIT with PEXIT token");
        }
        const IR::Condition cond{flow_test, static_cast<IR::Pred>(pred.index), pred.negated};
        if (exits_to_dispatcher) {
            block->end = pc;
            block->end_class = EndClass::Branch;
            block->cond = cond;
            block->branch_true = dispatch_block;
            block->branch_false = AddLabel(block, block->stack, pc + 1, function_id);
            return AnalysisState::Branch;
        }
        AnalyzeCondInst(block, function_id, pc, EndClass::Exit, cond);
        return AnalysisState::Branch;
    }
    if (const std::optional<Location> exit_pc{block->stack.Peek(Token::PEXIT)}) {
        const Stack popped_stack{block->stack.Remove(Token::PEXIT)};
        block->cond = IR::Condition{true};
        block->branch_true = AddLabel(block, popped_stack, *exit_pc, function_id);
        block->branch_false = nullptr;
        return AnalysisState::Branch;
    }
    if (exits_to_dispatcher) {
        block->cond = IR::Condition{true};
        block->end = pc;
        block->end_class = EndClass::Branch;
        block->branch_true = dispatch_block;
        block->branch_false = nullptr;
        return AnalysisState::Branch;
    }
    block->end = pc + 1;
    block->end_class = EndClass::Exit;
    return AnalysisState::Branch;
}

Block* CFG::AddLabel(Block* block, Stack stack, Location pc, FunctionId function_id) {
    Function& function{functions[function_id]};
    if (block->begin == pc) {
        // Jumps to itself
        return block;
    }
    if (const auto it{function.blocks.find(pc, Compare{})}; it != function.blocks.end()) {
        // Block already exists and it has been visited
        if (function.blocks.begin() != it) {
            // Check if the previous node is the virtual variant of the label
            // This won't exist if a virtual node is not needed or it hasn't been visited
            // If it hasn't been visited and a virtual node is needed, this will still behave as
            // expected because the node impersonated with its virtual node.
            const auto prev{std::prev(it)};
            if (it->begin.Virtual() == prev->begin) {
                return &*prev;
            }
        }
        return &*it;
    }
    // Make sure we don't insert the same layer twice
    const auto label_it{std::ranges::find(function.labels, pc, &Label::address)};
    if (label_it != function.labels.end()) {
        return label_it->block;
    }
    Block* const new_block{block_pool.Create()};
    new_block->begin = pc;
    new_block->end = pc;
    new_block->end_class = EndClass::Branch;
    new_block->cond = IR::Condition(true);
    new_block->stack = stack;
    new_block->branch_true = nullptr;
    new_block->branch_false = nullptr;
    function.labels.push_back(Label{
        .address{pc},
        .block = new_block,
        .stack{std::move(stack)},
    });
    return new_block;
}

std::string CFG::Dot() const {
    int node_uid{0};

    std::string dot{"digraph shader {\n"};
    for (const Function& function : functions) {
        dot += fmt::format("\tsubgraph cluster_{} {{\n", function.entrypoint);
        dot += fmt::format("\t\tnode [style=filled];\n");
        for (const Block& block : function.blocks) {
            const std::string name{NameOf(block)};
            const auto add_branch = [&](Block* branch, bool add_label) {
                dot += fmt::format("\t\t{}->{}", name, NameOf(*branch));
                if (add_label && block.cond != IR::Condition{true} &&
                    block.cond != IR::Condition{false}) {
                    dot += fmt::format(" [label=\"{}\"]", block.cond);
                }
                dot += '\n';
            };
            dot += fmt::format("\t\t{};\n", name);
            switch (block.end_class) {
            case EndClass::Branch:
                if (block.cond != IR::Condition{false}) {
                    add_branch(block.branch_true, true);
                }
                if (block.cond != IR::Condition{true}) {
                    add_branch(block.branch_false, false);
                }
                break;
            case EndClass::IndirectBranch:
                for (const IndirectBranch& branch : block.indirect_branches) {
                    add_branch(branch.block, false);
                }
                break;
            case EndClass::Call:
                dot += fmt::format("\t\t{}->N{};\n", name, node_uid);
                dot += fmt::format("\t\tN{}->{};\n", node_uid, NameOf(*block.return_block));
                dot += fmt::format("\t\tN{} [label=\"Call {}\"][shape=square][style=stripped];\n",
                                   node_uid, block.function_call);
                dot += '\n';
                ++node_uid;
                break;
            case EndClass::Exit:
                dot += fmt::format("\t\t{}->N{};\n", name, node_uid);
                dot += fmt::format("\t\tN{} [label=\"Exit\"][shape=square][style=stripped];\n",
                                   node_uid);
                ++node_uid;
                break;
            case EndClass::Return:
                dot += fmt::format("\t\t{}->N{};\n", name, node_uid);
                dot += fmt::format("\t\tN{} [label=\"Return\"][shape=square][style=stripped];\n",
                                   node_uid);
                ++node_uid;
                break;
            case EndClass::Kill:
                dot += fmt::format("\t\t{}->N{};\n", name, node_uid);
                dot += fmt::format("\t\tN{} [label=\"Kill\"][shape=square][style=stripped];\n",
                                   node_uid);
                ++node_uid;
                break;
            }
        }
        if (function.entrypoint == 8) {
            dot += fmt::format("\t\tlabel = \"main\";\n");
        } else {
            dot += fmt::format("\t\tlabel = \"Function {}\";\n", function.entrypoint);
        }
        dot += "\t}\n";
    }
    if (!functions.empty()) {
        auto& function{functions.front()};
        if (function.blocks.empty()) {
            dot += "Start;\n";
        } else {
            dot += fmt::format("\tStart -> {};\n", NameOf(*function.blocks.begin()));
        }
        dot += fmt::format("\tStart [shape=diamond];\n");
    }
    dot += "}\n";
    return dot;
}

} // namespace Shader::Maxwell::Flow
