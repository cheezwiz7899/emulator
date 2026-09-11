// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/texture_slot.h"
#include "video_core/precache_cfg_artifact.h"
#include "video_core/precache_frontend_artifact.h"
#include "video_core/shader_program_identity.h"
#include "video_core/speculative_shader_environment.h"
#include "video_core/spirv_cache.h"

namespace {

class TemporaryArtifactFile {
public:
    TemporaryArtifactFile() {
        static std::atomic<u64> next_id{};
        const auto base = std::filesystem::temp_directory_path();
        std::error_code ec;
        for (u64 attempt = 0; attempt < 128; ++attempt) {
            directory = base / ("citron_precache_cfg_artifact_" + std::to_string(++next_id));
            if (std::filesystem::create_directory(directory, ec)) {
                path = directory / "artifact.bin";
                return;
            }
            ec.clear();
        }
        throw std::runtime_error("could not create private CFG artifact directory");
    }

    ~TemporaryArtifactFile() {
        std::error_code ec;
        std::filesystem::remove_all(directory, ec);
    }

    std::filesystem::path path;
    std::filesystem::path directory;
};

TEST_CASE("Maxwell program identity is independent of CFG entry", "[video_core]") {
    constexpr std::array<u64, 7> program{
        0x0000000000001234ULL,
        0x0000000000005678ULL,
        0x1111111111111111ULL,
        0x2222222222222222ULL,
        VideoCommon::MAXWELL_SELF_BRANCH_A,
        0xaaaaaaaaaaaaaaaaULL,
        0xbbbbbbbbbbbbbbbbULL,
    };
    const auto identity = VideoCommon::ComputeMaxwellProgramIdentity(program);
    REQUIRE(identity.has_value());
    const auto bounded_identity =
        VideoCommon::ComputeMaxwellProgramIdentity(program, 4 * sizeof(u64));
    REQUIRE(bounded_identity.has_value());
    REQUIRE(*identity == *bounded_identity);
}

TEST_CASE("Maxwell program identity rejects invalid explicit spans", "[video_core]") {
    constexpr std::array<u64, 3> program{0x1, 0x2, VideoCommon::MAXWELL_SELF_BRANCH_A};
    REQUIRE_FALSE(VideoCommon::ComputeMaxwellProgramIdentity(program, 0).has_value());
    REQUIRE_FALSE(VideoCommon::ComputeMaxwellProgramIdentity(program, 7).has_value());
    REQUIRE_FALSE(VideoCommon::ComputeMaxwellProgramIdentity(program, 4 * sizeof(u64)).has_value());
}

TEST_CASE("Maxwell program identity rejects incomplete or different programs", "[video_core]") {
    constexpr std::array<u64, 3> incomplete{0x1, 0x2, 0x3};
    constexpr std::array<u64, 4> first{0x1, 0x2, VideoCommon::MAXWELL_SELF_BRANCH_B, 0x3};
    constexpr std::array<u64, 4> second{0x1, 0x4, VideoCommon::MAXWELL_SELF_BRANCH_B, 0x3};
    constexpr std::array<u64, 4> different_header{0x9, 0x2, VideoCommon::MAXWELL_SELF_BRANCH_B,
                                                  0x3};
    REQUIRE_FALSE(VideoCommon::ComputeMaxwellProgramIdentity(incomplete).has_value());
    REQUIRE(*VideoCommon::ComputeMaxwellProgramIdentity(first) !=
            *VideoCommon::ComputeMaxwellProgramIdentity(second));
    REQUIRE(*VideoCommon::ComputeMaxwellProgramIdentity(first) !=
            *VideoCommon::ComputeMaxwellProgramIdentity(different_header));
}

TEST_CASE("Maxwell program identity keeps distinct entries in one blob separate", "[video_core]") {
    constexpr std::array<u64, 6> blob{
        0x1111111111111111ULL, 0x2222222222222222ULL, VideoCommon::MAXWELL_SELF_BRANCH_A,
        0x3333333333333333ULL, 0x4444444444444444ULL, VideoCommon::MAXWELL_SELF_BRANCH_B,
    };
    const std::span<const u64> first{blob.data(), 3};
    const std::span<const u64> second{blob.data() + 3, 3};
    const auto first_identity = VideoCommon::ComputeMaxwellProgramIdentity(first);
    const auto second_identity = VideoCommon::ComputeMaxwellProgramIdentity(second);
    REQUIRE(first_identity.has_value());
    REQUIRE(second_identity.has_value());
    REQUIRE(*first_identity != *second_identity);
}

TEST_CASE("SPIR-V cache identity keeps execution stages separate", "[video_core]") {
    constexpr u64 program_hash = 0x123456789abcdef0ULL;
    const VideoCommon::SpirvKey vertex{program_hash, Shader::Stage::VertexB, 7, 0, 1, 2};
    const VideoCommon::SpirvKey fragment{program_hash, Shader::Stage::Fragment, 7, 0, 1, 2};
    REQUIRE(vertex != fragment);
}

TEST_CASE("Pre-cache CFG artifact key separates scheduler slots", "[video_core]") {
    constexpr u64 identity = 0x123456789abcdef0ULL;
    const auto slot_0 =
        VideoCommon::MakePrecacheCfgArtifactKey(identity, Shader::Stage::VertexB, 0x80);
    const auto slot_8 =
        VideoCommon::MakePrecacheCfgArtifactKey(identity, Shader::Stage::VertexB, 0x88);
    const auto slot_16 =
        VideoCommon::MakePrecacheCfgArtifactKey(identity, Shader::Stage::VertexB, 0x90);
    const auto slot_24 =
        VideoCommon::MakePrecacheCfgArtifactKey(identity, Shader::Stage::VertexB, 0x98);
    REQUIRE(slot_0.has_value());
    REQUIRE(slot_8.has_value());
    REQUIRE(slot_16.has_value());
    REQUIRE(slot_24.has_value());
    REQUIRE(slot_0->scheduler_slot == 0);
    REQUIRE(slot_8->scheduler_slot == 8);
    REQUIRE(slot_16->scheduler_slot == 16);
    REQUIRE(slot_24->scheduler_slot == 24);
    REQUIRE(*slot_0 != *slot_8);
    REQUIRE(*slot_8 != *slot_16);
    REQUIRE(*slot_16 != *slot_24);
    REQUIRE(*slot_0 !=
            *VideoCommon::MakePrecacheCfgArtifactKey(identity, Shader::Stage::Fragment, 0x80));
    REQUIRE_FALSE(VideoCommon::MakePrecacheCfgArtifactKey(0, Shader::Stage::VertexB, 0x80));
    REQUIRE_FALSE(
        VideoCommon::MakePrecacheCfgArtifactKey(identity, static_cast<Shader::Stage>(99), 0x80));
    REQUIRE_FALSE(VideoCommon::MakePrecacheCfgArtifactKey(identity, Shader::Stage::VertexB, 0x84));
}

TEST_CASE("Logical texture slots retain complete descriptor provenance", "[video_core]") {
    const Shader::TextureSlot primary{
        .cbuf_index = 2,
        .cbuf_offset = 64,
        .shift_left = 5,
    };
    const Shader::TextureSlot secondary{
        .cbuf_index = 2,
        .cbuf_offset = 64,
        .shift_left = 5,
        .secondary_cbuf_index = 3,
        .secondary_cbuf_offset = 128,
        .secondary_shift_left = 2,
        .has_secondary = true,
    };
    const Shader::TextureSlot array = {
        .cbuf_index = 2,
        .cbuf_offset = 64,
        .shift_left = 5,
        .count = 4,
    };
    Shader::LogicalTextureSlots slots;
    slots.emplace(primary, Shader::TextureSlotShape{});
    slots.emplace(secondary, Shader::TextureSlotShape{});
    slots.emplace(array, Shader::TextureSlotShape{});
    REQUIRE(primary != secondary);
    REQUIRE(primary != array);
    REQUIRE(slots.size() == 3);
}

TEST_CASE("Speculative environment distinguishes unknown dependencies", "[video_core]") {
    VideoCommon::SpeculativeShaderEnvironment env{{0}, Shader::Stage::Fragment, 0, {}};
    REQUIRE_FALSE(env.HasUnknownCbufDependencies());
    REQUIRE_FALSE(env.HasUnknownTextureDependencies());
    REQUIRE_FALSE(env.HasUnknownViewportDependency());
    REQUIRE(env.HasStateIndependentTemplateContract());
    REQUIRE(env.HasCompleteFinalModuleContract());

    static_cast<void>(env.ReadCbufValue(0, 0));
    REQUIRE(env.HasUnknownCbufDependencies());
    REQUIRE_FALSE(env.HasUnknownTextureDependencies());
    REQUIRE_FALSE(env.HasUnknownViewportDependency());
    REQUIRE_FALSE(env.HasStateIndependentTemplateContract());
    REQUIRE_FALSE(env.HasCompleteFinalModuleContract());

    static_cast<void>(env.ReadTextureType(7));
    REQUIRE(env.HasUnknownTextureDependencies());
    REQUIRE_FALSE(env.HasUnknownViewportDependency());

    static_cast<void>(env.ReadViewportTransformState());
    REQUIRE(env.HasUnknownViewportDependency());

    VideoCommon::SpeculativeShaderEnvironment compute{{0}, Shader::Stage::Compute, 0, {}};
    static_cast<void>(compute.WorkgroupSize());
    REQUIRE(compute.HasUnknownComputeLaunchDependency());
    REQUIRE_FALSE(compute.HasStateIndependentTemplateContract());
    REQUIRE_FALSE(compute.HasCompleteFinalModuleContract());

    VideoCommon::SpeculativeShaderEnvironment hle{{0}, Shader::Stage::Fragment, 0, {}};
    REQUIRE_FALSE(hle.HasHLEMacroState());
    REQUIRE(hle.HasUnknownHLEMacroDependency());
    REQUIRE_FALSE(hle.HasStateIndependentTemplateContract());
    REQUIRE_FALSE(hle.HasCompleteFinalModuleContract());

    VideoCommon::SpeculativeShaderEnvironment binding{{0}, Shader::Stage::Fragment, 0, {}};
    static_cast<void>(binding.TextureBoundBuffer());
    REQUIRE(binding.HasUnknownTextureDependencies());
    REQUIRE_FALSE(binding.HasCompleteFinalModuleContract());

    VideoCommon::SpeculativeShaderEnvironment driver{{0}, Shader::Stage::Fragment, 0, {}};
    REQUIRE_FALSE(driver.IsProprietaryDriver());
    REQUIRE(driver.HasUnknownCbufDependencies());
    REQUIRE_FALSE(driver.HasCompleteFinalModuleContract());

    VideoCommon::SpeculativeShaderEnvironment passthrough{{0}, Shader::Stage::Geometry, 0, {}};
    static_cast<void>(passthrough.GpPassthroughMask());
    REQUIRE(passthrough.HasUnknownInterfaceDependency());
    REQUIRE_FALSE(passthrough.HasStateIndependentTemplateContract());
    REQUIRE_FALSE(passthrough.HasCompleteFinalModuleContract());

    VideoCommon::SpeculativeShaderEnvironment alignment{{0}, Shader::Stage::VertexB, 0, {}};
    alignment.MarkSchedulingAlignmentUnknown();
    REQUIRE(alignment.HasUnknownSchedulingAlignmentDependency());
    REQUIRE_FALSE(alignment.HasStateIndependentTemplateContract());
    REQUIRE_FALSE(alignment.HasCompleteFinalModuleContract());
}

TEST_CASE("Speculative environment isolates frontend dependency observations", "[video_core]") {
    VideoCommon::SpeculativeShaderEnvironment env{{0}, Shader::Stage::Fragment, 0, {}};

    // CFG reads and synthetic scheduler placement invalidate a final module,
    // but must not be charged to the later BuildProgramTemplate boundary.
    static_cast<void>(env.ReadCbufValue(0, 0));
    env.MarkSchedulingAlignmentUnknown();
    const auto cfg_dependencies = env.TakeDependencySnapshot();
    REQUIRE_FALSE(env.HasUnknownCbufDependencies());
    REQUIRE_FALSE(env.HasUnknownSchedulingAlignmentDependency());
    REQUIRE(env.HasStateIndependentTemplateContract());

    // A frontend read after the checkpoint remains visible to the frontend
    // gate, independently of the earlier CFG observations.
    static_cast<void>(env.ReadTextureType(7));
    REQUIRE(env.HasUnknownTextureDependencies());
    REQUIRE_FALSE(env.HasStateIndependentTemplateContract());

    env.MergeDependencySnapshot(cfg_dependencies);
    REQUIRE(env.HasUnknownCbufDependencies());
    REQUIRE(env.HasUnknownTextureDependencies());
    REQUIRE(env.HasUnknownSchedulingAlignmentDependency());
    REQUIRE_FALSE(env.HasCompleteFinalModuleContract());
}

TEST_CASE("Pre-cache frontend artifact restores owned IR references", "[video_core]") {
    Shader::ObjectPool<Shader::IR::Inst> source_insts{8};
    Shader::ObjectPool<Shader::IR::Block> source_blocks{2};
    Shader::IR::Block* const source_block{source_blocks.Create(source_insts)};
    source_block->AppendNewInst(Shader::IR::Opcode::GetRegister, {Shader::IR::Value{Shader::IR::Reg::R3}});
    Shader::IR::Inst& loaded{source_block->back()};
    source_block->AppendNewInst(Shader::IR::Opcode::SetRegister,
                                {Shader::IR::Value{Shader::IR::Reg::R4}, Shader::IR::Value{&loaded}});

    Shader::IR::Program source;
    source.stage = Shader::Stage::VertexB;
    source.blocks = {source_block};
    source.post_order_blocks = {source_block};
    Shader::IR::AbstractSyntaxNode node;
    node.type = Shader::IR::AbstractSyntaxNode::Type::Block;
    node.data.block = source_block;
    source.syntax_list.push_back(node);

    VideoCommon::PrecacheFrontendFreezeError error{};
    const auto frozen{VideoCommon::FreezePrecacheFrontendArtifact(source, error)};
    REQUIRE(frozen.has_value());
    REQUIRE(error == VideoCommon::PrecacheFrontendFreezeError::None);
    REQUIRE(frozen->IsValid());

    Shader::ObjectPool<Shader::IR::Inst> restored_insts{8};
    Shader::ObjectPool<Shader::IR::Block> restored_blocks{2};
    const auto restored{VideoCommon::RestorePrecacheFrontendArtifact(*frozen, restored_insts,
                                                                       restored_blocks)};
    REQUIRE(restored.has_value());
    REQUIRE(restored->stage == Shader::Stage::VertexB);
    REQUIRE(restored->blocks.size() == 1);
    REQUIRE(restored->blocks.front()->size() == 2);
    auto it{restored->blocks.front()->begin()};
    Shader::IR::Inst& restored_load{*it++};
    Shader::IR::Inst& restored_store{*it};
    REQUIRE(restored_load.GetOpcode() == Shader::IR::Opcode::GetRegister);
    REQUIRE(restored_store.GetOpcode() == Shader::IR::Opcode::SetRegister);
    REQUIRE(restored_store.Arg(1).Inst() == &restored_load);

    VideoCommon::PrecacheFrontendArtifactRecord record{
        .key = {
            .program_identity = 0x123456789abcdef0ULL,
            .compiler_target_fingerprint = 0x1020304050607080ULL,
            .source_header = 0x8877665544332211ULL,
            .local_memory_size = 0,
            .stage = Shader::Stage::VertexB,
            .scheduler_slot = 0,
            .exits_to_dispatcher = false,
        },
        .artifact = *frozen,
    };
    REQUIRE(record.IsValid());
    TemporaryArtifactFile artifact_file;
    const std::array records{record};
    REQUIRE(VideoCommon::SavePrecacheFrontendArtifacts(artifact_file.path, records));
    const auto loaded_records{VideoCommon::LoadPrecacheFrontendArtifacts(artifact_file.path)};
    REQUIRE(loaded_records.has_value());
    REQUIRE(loaded_records->size() == 1);
    REQUIRE(loaded_records->front().key == record.key);
    REQUIRE(loaded_records->front().artifact.IsValid());
    REQUIRE(loaded_records->front().artifact.blocks.size() == 1);
    REQUIRE(loaded_records->front().artifact.blocks.front().instructions.size() == 2);
}

TEST_CASE("CFG template preserves owners and raw virtual locations", "[video_core]") {
    using Cfg = Shader::Maxwell::Flow::CFG;
    using Location = Shader::Maxwell::Location;

    Cfg::Template source;
    source.functions = {
        {.entrypoint = Location::FromRawOffset(8), .blocks = {0}},
        {.entrypoint = Location::FromRawOffset(12), .blocks = {1}},
    };
    source.blocks.resize(2);
    source.blocks[0].begin = Location::FromRawOffset(8);
    source.blocks[0].end = Location::FromRawOffset(8);
    source.blocks[0].owner = 0;
    source.blocks[0].function_call = 1;
    source.blocks[0].branch_true = 1;
    source.blocks[0].stack.SetEntries(
        {{Shader::Maxwell::Flow::Token::SSY, Location::FromRawOffset(12)}});
    source.blocks[1].begin = Location::FromRawOffset(12);
    source.blocks[1].end = Location::FromRawOffset(12);
    source.blocks[1].owner = 1;
    source.blocks[1].return_block = 0;

    REQUIRE(Cfg::IsValidTemplate(source));
    VideoCommon::SpeculativeShaderEnvironment env{{0}, Shader::Stage::VertexB, 0, {}};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> pool(4);
    REQUIRE(Cfg::RoundTripMatchesTemplate(env, pool, source));

    const auto rebased =
        Cfg::RebaseTemplate(source, Location::FromRawOffset(8), Location::FromRawOffset(40));
    REQUIRE(rebased.has_value());
    REQUIRE(rebased->functions[0].entrypoint == Location::FromRawOffset(40));
    REQUIRE(rebased->functions[1].entrypoint == Location::FromRawOffset(44));
    REQUIRE(rebased->blocks[0].begin == Location::FromRawOffset(40));
    REQUIRE(rebased->blocks[1].begin == Location::FromRawOffset(44));
    REQUIRE(rebased->blocks[0].stack.Entries().size() == 1);
    REQUIRE(rebased->blocks[0].stack.Entries()[0].target == Location::FromRawOffset(44));
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> rebased_pool(4);
    REQUIRE(Cfg::RoundTripMatchesTemplate(env, rebased_pool, *rebased));
    REQUIRE_FALSE(
        Cfg::RebaseTemplate(source, Location::FromRawOffset(8), Location::FromRawOffset(32))
            .has_value());
    const auto alternate_slot_rebased =
        Cfg::RebaseTemplate(source, Location::FromRawOffset(8), Location::FromRawOffset(32), false);
    REQUIRE(alternate_slot_rebased.has_value());
    REQUIRE(alternate_slot_rebased->functions[0].entrypoint == Location::FromRawOffset(32));

    const auto artifact_key =
        VideoCommon::MakePrecacheCfgArtifactKey(0x123456789abcdef0ULL, Shader::Stage::VertexB, 8);
    REQUIRE(artifact_key.has_value());
    VideoCommon::PrecacheCfgArtifact artifact{*artifact_key, source};
    for (const auto& block : source.blocks) {
        for (auto location = block.begin; location != block.end; ++location) {
            artifact.decoded_instructions.push_back(
                {.location = location.Offset(), .instruction = 0,
                 .opcode = Shader::Maxwell::Opcode::NOP});
        }
    }
    std::ranges::sort(artifact.decoded_instructions, {},
                      &Shader::Maxwell::PredecodedInstruction::location);
    REQUIRE(artifact.IsValid());
    auto incomplete_decoded_artifact = artifact;
    incomplete_decoded_artifact.decoded_instructions.pop_back();
    REQUIRE_FALSE(incomplete_decoded_artifact.IsValid());
    auto invalid_decoded_artifact = artifact;
    invalid_decoded_artifact.decoded_instructions.front().opcode =
        Shader::Maxwell::Opcode::BRA;
    REQUIRE_FALSE(invalid_decoded_artifact.IsValid());
    auto invalid_artifact_stage = artifact;
    invalid_artifact_stage.key.stage = static_cast<Shader::Stage>(99);
    REQUIRE_FALSE(invalid_artifact_stage.IsValid());
    auto invalid_artifact_slot = artifact;
    invalid_artifact_slot.key.scheduler_slot = 7;
    REQUIRE_FALSE(invalid_artifact_slot.IsValid());
    const auto artifact_rebased = artifact.RebaseFor(0x123456789abcdef0ULL, Shader::Stage::VertexB,
                                                     Location::FromRawOffset(40));
    REQUIRE(artifact_rebased == rebased);
    auto rebased_decoded = artifact.decoded_instructions;
    for (auto& instruction : rebased_decoded) {
        instruction.location += 32;
    }
    REQUIRE(VideoCommon::PredecodedInstructionsMatchTemplate(*artifact_rebased, rebased_decoded));
    const auto cross_slot_artifact_rebased = artifact.RebaseFor(
        0x123456789abcdef0ULL, Shader::Stage::VertexB, Location::FromRawOffset(32));
    REQUIRE_FALSE(cross_slot_artifact_rebased.has_value());
    REQUIRE_FALSE(
        artifact
            .RebaseFor(0x123456789abcdef1ULL, Shader::Stage::VertexB, Location::FromRawOffset(40))
            .has_value());
    REQUIRE_FALSE(
        artifact
            .RebaseFor(0x123456789abcdef0ULL, Shader::Stage::Fragment, Location::FromRawOffset(40))
            .has_value());
    REQUIRE_FALSE(artifact
                      .RebaseFor(0x123456789abcdef0ULL, Shader::Stage::VertexB,
                                 Location::FromRawOffset(40), true)
                      .has_value());

    TemporaryArtifactFile artifact_file;
    const std::array artifacts{artifact};
    const std::array duplicate_artifacts{artifact, artifact};
    const std::array invalid_stage_artifacts{invalid_artifact_stage};
    REQUIRE_FALSE(VideoCommon::SavePrecacheCfgArtifacts(artifact_file.path, duplicate_artifacts));
    REQUIRE_FALSE(
        VideoCommon::SavePrecacheCfgArtifacts(artifact_file.path, invalid_stage_artifacts));
    auto second_artifact = artifact;
    second_artifact.key.program_identity += 1;
    const std::array ordered_artifacts{artifact, second_artifact};
    const std::array reverse_ordered_artifacts{second_artifact, artifact};
    const auto ordered_path = artifact_file.directory / "ordered.bin";
    const auto reverse_ordered_path = artifact_file.directory / "reverse.bin";
    REQUIRE(VideoCommon::SavePrecacheCfgArtifacts(ordered_path, ordered_artifacts));
    REQUIRE(VideoCommon::SavePrecacheCfgArtifacts(reverse_ordered_path, reverse_ordered_artifacts));
    std::ifstream ordered_file{ordered_path, std::ios::binary};
    std::ifstream reverse_ordered_file{reverse_ordered_path, std::ios::binary};
    const std::vector<char> ordered_bytes{std::istreambuf_iterator<char>{ordered_file},
                                          std::istreambuf_iterator<char>{}};
    const std::vector<char> reverse_ordered_bytes{
        std::istreambuf_iterator<char>{reverse_ordered_file}, std::istreambuf_iterator<char>{}};
    REQUIRE(ordered_bytes == reverse_ordered_bytes);
    REQUIRE(VideoCommon::SavePrecacheCfgArtifacts(artifact_file.path, artifacts));
    const auto loaded_artifacts = VideoCommon::LoadPrecacheCfgArtifacts(artifact_file.path);
    REQUIRE(loaded_artifacts.has_value());
    REQUIRE(loaded_artifacts->size() == 1);
    REQUIRE(loaded_artifacts->front().key == artifact.key);
    REQUIRE(loaded_artifacts->front().cfg == artifact.cfg);
    REQUIRE(loaded_artifacts->front().decoded_instructions == artifact.decoded_instructions);
    auto replacement = artifact;
    replacement.key.program_identity ^= 0x1000ULL;
    const std::array replacement_artifacts{replacement};
    REQUIRE(VideoCommon::SavePrecacheCfgArtifacts(artifact_file.path, replacement_artifacts));
    const auto replaced_artifacts = VideoCommon::LoadPrecacheCfgArtifacts(artifact_file.path);
    REQUIRE(replaced_artifacts.has_value());
    REQUIRE(replaced_artifacts->size() == 1);
    REQUIRE(replaced_artifacts->front().key == replacement.key);
    // Simulate termination in Save()'s short Windows replacement gap.
    const auto backup =
        artifact_file.path.parent_path() / (artifact_file.path.filename().string() + ".bak");
    std::filesystem::copy_file(artifact_file.path, backup,
                               std::filesystem::copy_options::overwrite_existing);
    std::filesystem::remove(artifact_file.path);
    const auto recovered_artifacts = VideoCommon::LoadPrecacheCfgArtifacts(artifact_file.path);
    REQUIRE(recovered_artifacts.has_value());
    REQUIRE(recovered_artifacts->size() == 1);
    REQUIRE(recovered_artifacts->front().key == replacement.key);
    const auto legacy_path = artifact_file.directory / "legacy.bin";
    REQUIRE(VideoCommon::SavePrecacheCfgArtifacts(legacy_path, replacement_artifacts));
    {
        std::fstream legacy_file{legacy_path, std::ios::binary | std::ios::in | std::ios::out};
        REQUIRE(legacy_file.is_open());
        constexpr u32 legacy_version = 3;
        legacy_file.seekp(8);
        legacy_file.write(reinterpret_cast<const char*>(&legacy_version), sizeof(legacy_version));
    }
    REQUIRE_FALSE(VideoCommon::LoadPrecacheCfgArtifacts(legacy_path).has_value());
    REQUIRE(VideoCommon::SavePrecacheCfgArtifacts(artifact_file.path, replacement_artifacts));
    {
        std::ofstream corrupt{artifact_file.path, std::ios::binary | std::ios::app};
        const char trailing{};
        corrupt.write(&trailing, sizeof(trailing));
    }
    REQUIRE_FALSE(VideoCommon::LoadPrecacheCfgArtifacts(artifact_file.path).has_value());

    auto indirect = source;
    indirect.blocks[0].indirect_branches.emplace_back(1, 0x100);
    REQUIRE(Cfg::IsValidTemplate(indirect));
    REQUIRE_FALSE(
        Cfg::RebaseTemplate(indirect, Location::FromRawOffset(8), Location::FromRawOffset(40))
            .has_value());

    auto far_location = source;
    far_location.blocks[0].end = Location::FromRawOffset(0x01000000);
    REQUIRE(Cfg::IsValidTemplate(far_location));
    REQUIRE_FALSE(
        Cfg::RebaseTemplate(far_location, Location::FromRawOffset(8), Location::FromRawOffset(40))
            .has_value());

    auto missing_entrypoint = source;
    missing_entrypoint.blocks[1].begin = Location::FromRawOffset(20);
    missing_entrypoint.blocks[1].end = Location::FromRawOffset(20);
    REQUIRE_FALSE(Cfg::IsValidTemplate(missing_entrypoint));

    auto duplicate = source;
    duplicate.functions[1].blocks = {0};
    REQUIRE_FALSE(Cfg::IsValidTemplate(duplicate));

    auto invalid_virtual = source;
    invalid_virtual.blocks[1].begin = Location::FromRawOffset(13);
    REQUIRE_FALSE(Cfg::IsValidTemplate(invalid_virtual));

    auto invalid_condition = source;
    invalid_condition.blocks[0].cond =
        Shader::IR::Condition{static_cast<Shader::IR::FlowTest>(999), Shader::IR::Pred::P0};
    REQUIRE_FALSE(Cfg::IsValidTemplate(invalid_condition));

    auto invalid_register = source;
    invalid_register.blocks[0].branch_reg = static_cast<Shader::IR::Reg>(999);
    REQUIRE_FALSE(Cfg::IsValidTemplate(invalid_register));

    auto invalid_token = source;
    invalid_token.blocks[0].stack.SetEntries(
        {{static_cast<Shader::Maxwell::Flow::Token>(99), Location::FromRawOffset(12)}});
    REQUIRE_FALSE(Cfg::IsValidTemplate(invalid_token));
}

TEST_CASE("Logical texture key ignores raw handles but preserves slot shapes", "[video_core]") {
    const Shader::TextureSlot first{.cbuf_index = 1, .cbuf_offset = 32, .count = 1};
    const Shader::TextureSlot second{.cbuf_index = 1, .cbuf_offset = 64, .count = 1};
    const Shader::TextureSlotShape color_2d{
        .type = Shader::TextureType::Color2D,
        .pixel_format = Shader::TexturePixelFormat::A8B8G8R8_UNORM,
        .is_integer = false,
    };
    const Shader::TextureSlotShape color_cube{
        .type = Shader::TextureType::ColorCube,
        .pixel_format = Shader::TexturePixelFormat::A8B8G8R8_UNORM,
        .is_integer = false,
    };
    const Shader::LogicalTextureSlots slots{{first, color_2d}, {second, color_cube}};
    const Shader::LogicalTextureSlots swapped{{first, color_cube}, {second, color_2d}};
    Shader::LogicalTextureSlots reverse_order;
    reverse_order.emplace(second, color_cube);
    reverse_order.emplace(first, color_2d);

    REQUIRE(VideoCommon::ComputeLogicalTextureKey(slots) !=
            VideoCommon::ComputeLogicalTextureKey(swapped));

    // Raw descriptor handles are deliberately absent from the logical key.
    // Coverage still requires each observed raw query to have slot provenance.
    const std::unordered_map<u32, Shader::TextureType> first_handles{
        {7, Shader::TextureType::Color2D}, {91, Shader::TextureType::ColorCube}};
    const std::unordered_map<u32, Shader::TextureType> second_handles{
        {1007, Shader::TextureType::Color2D}, {2003, Shader::TextureType::ColorCube}};
    const std::unordered_map<u32, Shader::TexturePixelFormat> first_formats{
        {7, Shader::TexturePixelFormat::A8B8G8R8_UNORM},
        {91, Shader::TexturePixelFormat::A8B8G8R8_UNORM},
    };
    const std::unordered_map<u32, Shader::TexturePixelFormat> second_formats{
        {1007, Shader::TexturePixelFormat::A8B8G8R8_UNORM},
        {2003, Shader::TexturePixelFormat::A8B8G8R8_UNORM},
    };
    REQUIRE(VideoCommon::HasCompleteLogicalTextureCoverage(
        slots, Shader::LogicalTextureHandles{7, 91}, first_handles, first_formats));
    REQUIRE(VideoCommon::HasCompleteLogicalTextureCoverage(
        slots, Shader::LogicalTextureHandles{1007, 2003}, second_handles, second_formats));
    REQUIRE_FALSE(VideoCommon::HasCompleteLogicalTextureCoverage(
        slots, Shader::LogicalTextureHandles{7}, first_handles, first_formats));
    REQUIRE(VideoCommon::ComputeLogicalTextureKey(slots) ==
            VideoCommon::ComputeLogicalTextureKey(reverse_order));

    const Shader::TextureSlot array_slot{.cbuf_index = 1, .cbuf_offset = 32, .count = 4};
    const Shader::TextureSlot indexed_slot{
        .cbuf_index = 1,
        .cbuf_offset = 32,
        .secondary_cbuf_index = 2,
        .secondary_cbuf_offset = 16,
        .secondary_shift_left = 2,
        .count = 1,
        .has_secondary = true,
    };
    const Shader::LogicalTextureSlots array_slots{{array_slot, color_2d}, {second, color_cube}};
    const Shader::LogicalTextureSlots indexed_slots{{indexed_slot, color_2d}, {second, color_cube}};
    REQUIRE(VideoCommon::ComputeLogicalTextureKey(slots) !=
            VideoCommon::ComputeLogicalTextureKey(array_slots));
    REQUIRE(VideoCommon::ComputeLogicalTextureKey(slots) !=
            VideoCommon::ComputeLogicalTextureKey(indexed_slots));

    // Aliased raw handles retain both source slots in the key.
    const Shader::LogicalTextureSlots aliased_slots{{first, color_2d}, {second, color_2d}};
    REQUIRE(VideoCommon::ComputeLogicalTextureKey(aliased_slots) !=
            VideoCommon::ComputeLogicalTextureKey(Shader::LogicalTextureSlots{{first, color_2d}}));
    const std::unordered_map<u32, Shader::TextureType> aliased_types{
        {7, Shader::TextureType::Color2D}};
    const std::unordered_map<u32, Shader::TexturePixelFormat> aliased_formats{
        {7, Shader::TexturePixelFormat::A8B8G8R8_UNORM}};
    REQUIRE(VideoCommon::HasCompleteLogicalTextureCoverage(
        aliased_slots, Shader::LogicalTextureHandles{7}, aliased_types, aliased_formats));
}

} // Anonymous namespace
