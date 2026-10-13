// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/precache_program_contract.h"
#include "video_core/shader_program_identity.h"

namespace {

constexpr size_t kContainerSize = 0x600;
constexpr size_t kGrsc = 0x60;
constexpr size_t kVariations = 0xA0;
constexpr size_t kProgram = 0x120;
constexpr size_t kVertexHeader = 0x200;
constexpr size_t kFragmentHeader = 0x280;
constexpr size_t kVertexCode = 0x300;
constexpr size_t kFragmentCode = 0x400;
constexpr size_t kCodeSize = 48 + 88;

constexpr u32 MakeSphCommon0(u32 shader_type) {
    const u32 sph_type = shader_type == 5 ? 2 : 1;
    return sph_type | (1u << 5) | (shader_type << 10) | (1u << 17);
}

template <typename T>
void Write(std::vector<u8>& bytes, size_t offset, T value) {
    REQUIRE(offset + sizeof(T) <= bytes.size());
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

std::vector<u8> MakeBNSH() {
    std::vector<u8> bytes(kContainerSize);
    Write<u32>(bytes, 0x00, 0x48534E42u);
    Write<u16>(bytes, 0x16, static_cast<u16>(kGrsc));
    Write<u32>(bytes, kGrsc, 0x63737267u);
    Write<u32>(bytes, kGrsc + 0x1C, 1);
    Write<u64>(bytes, kGrsc + 0x20, kVariations);
    Write<u64>(bytes, kVariations + 0x10, kProgram);
    Write<u64>(bytes, kProgram + 0x08, kVertexHeader);
    Write<u64>(bytes, kProgram + 0x28, kFragmentHeader);
    for (const size_t header : {kVertexHeader, kFragmentHeader}) {
        Write<u64>(bytes, header + 0x10, header == kVertexHeader ? kVertexCode : kFragmentCode);
        Write<u32>(bytes, header + 0x18, kCodeSize);
    }
    Write<u32>(bytes, kVertexCode + 48, MakeSphCommon0(1));
    Write<u32>(bytes, kFragmentCode + 48, MakeSphCommon0(5));
    // The parser retains only after the 48-byte preamble. A zero ProgramHeader is
    // sufficient for parser ownership; the terminal branch supplies its identity.
    Write<u64>(bytes, kVertexCode + 48 + 80, VideoCommon::MAXWELL_SELF_BRANCH_A);
    Write<u64>(bytes, kFragmentCode + 48 + 80, VideoCommon::MAXWELL_SELF_BRANCH_B);
    return bytes;
}

VideoCommon::PrecacheDependencyLedger CompleteLedger() {
    VideoCommon::PrecacheDependencyLedger ledger;
    for (u8 index = 0; index < static_cast<u8>(VideoCommon::PrecacheDependency::Count); ++index) {
        ledger.Set(static_cast<VideoCommon::PrecacheDependency>(index),
                   VideoCommon::PrecacheDependencyProvenance::SourceExact);
    }
    ledger.Set(VideoCommon::PrecacheDependency::ProgramBytes,
               VideoCommon::PrecacheDependencyProvenance::ProgramDerived);
    return ledger;
}

} // Anonymous namespace

TEST_CASE("BNSH D1 source parser retains exact declared byte-code groups", "[video_core]") {
    const auto bytes = MakeBNSH();
    const auto parsed = VideoCommon::ParsePrecacheBNSHProgramGroups(bytes, 0);
    REQUIRE(parsed.rejection == VideoCommon::PrecacheBNSHParseRejection::None);
    REQUIRE(parsed.declared_variations == 1);
    REQUIRE(parsed.groups.size() == 1);
    const auto& group = parsed.groups.front();
    REQUIRE(group.container_offset == 0);
    REQUIRE(group.origin == VideoCommon::PrecacheSourceOrigin::BNSHDeclaredProgram);
    REQUIRE(group.variation_index == 0);
    REQUIRE(group.program_offset == kProgram);
    REQUIRE(group.stages.size() == 2);
    REQUIRE(group.stages[0].stage == Shader::Stage::VertexB);
    REQUIRE(group.stages[1].stage == Shader::Stage::Fragment);
    REQUIRE(group.stages[0].bytes.size() == 88);
    REQUIRE(group.stages[0].HasExactProgramIdentity());
    REQUIRE(group.stages[1].HasExactProgramIdentity());

    const auto decision = VideoCommon::ClassifyPrecacheSourceProgramGroup(group, CompleteLedger());
    REQUIRE(decision.eligible);
    REQUIRE(decision.primary_rejection == VideoCommon::PrecacheContractRejection::None);
}

TEST_CASE("BNSH D1 parser rejects unbounded or ambiguous source declarations", "[video_core]") {
    auto bytes = MakeBNSH();
    Write<u64>(bytes, kVertexHeader + 0x10, 0xFFFF'FFFF'FFFF'FFF0ULL);
    REQUIRE(VideoCommon::ParsePrecacheBNSHProgramGroups(bytes, 0).rejection ==
            VideoCommon::PrecacheBNSHParseRejection::Malformed);

    bytes = MakeBNSH();
    Write<u64>(bytes, kProgram + 0x10, kVertexHeader);
    REQUIRE(VideoCommon::ParsePrecacheBNSHProgramGroups(bytes, 0).rejection ==
            VideoCommon::PrecacheBNSHParseRejection::AmbiguousStage);

    bytes = MakeBNSH();
    VideoCommon::PrecacheBNSHParseLimits limits;
    limits.max_stage_bytes = 64;
    REQUIRE(VideoCommon::ParsePrecacheBNSHProgramGroups(bytes, 0, limits).rejection ==
            VideoCommon::PrecacheBNSHParseRejection::Capacity);

    bytes = MakeBNSH();
    bytes.resize(kFragmentCode + 80);
    REQUIRE(VideoCommon::ParsePrecacheBNSHProgramGroups(bytes, 0).rejection ==
            VideoCommon::PrecacheBNSHParseRejection::Malformed);
}

TEST_CASE("D1 contract classifier fails closed for missing state", "[video_core]") {
    const auto parsed = VideoCommon::ParsePrecacheBNSHProgramGroups(MakeBNSH(), 0);
    REQUIRE(parsed.groups.size() == 1);
    auto ledger = CompleteLedger();
    ledger.Set(VideoCommon::PrecacheDependency::FixedPipelineState,
               VideoCommon::PrecacheDependencyProvenance::Unknown);
    const auto missing = VideoCommon::ClassifyPrecacheSourceProgramGroup(parsed.groups.front(), ledger);
    REQUIRE_FALSE(missing.eligible);
    REQUIRE(missing.primary_rejection == VideoCommon::PrecacheContractRejection::MissingDependency);
    REQUIRE((missing.missing_dependencies &
             VideoCommon::PrecacheDependencyBit(VideoCommon::PrecacheDependency::FixedPipelineState)) != 0);

    auto unsupported = parsed.groups.front();
    unsupported.stages.pop_back();
    REQUIRE(VideoCommon::ClassifyPrecacheSourceProgramGroup(unsupported, CompleteLedger())
                .primary_rejection == VideoCommon::PrecacheContractRejection::UnsupportedStageShape);
}

TEST_CASE("D1 standalone sources retain one exact stage but cannot invent a graphics pair",
          "[video_core]") {
    const auto bytes = MakeBNSH();
    const std::span<const u8> vertex_program{bytes.data() + kVertexCode + 48, 88};
    const auto standalone = VideoCommon::ParsePrecacheStandaloneProgram(
        vertex_program, Shader::Stage::VertexB, 0x1234);
    REQUIRE(standalone.rejection == VideoCommon::PrecacheBNSHParseRejection::None);
    REQUIRE(standalone.group.has_value());
    REQUIRE(standalone.group->origin == VideoCommon::PrecacheSourceOrigin::StandaloneProgram);
    REQUIRE(standalone.group->stages.size() == 1);
    REQUIRE(standalone.group->stages.front().HasExactProgramIdentity());
    REQUIRE(VideoCommon::ClassifyPrecacheSourceProgramGroup(*standalone.group, CompleteLedger())
                .primary_rejection == VideoCommon::PrecacheContractRejection::UnsupportedStageShape);
}

TEST_CASE("D1 captured graphics contract accepts only complete exact provenance", "[video_core]") {
    const std::array captured{
        VideoCommon::PrecacheCapturedGraphicsStage{Shader::Stage::VertexB, 0x1234},
        VideoCommon::PrecacheCapturedGraphicsStage{Shader::Stage::Fragment, 0x5678},
    };
    REQUIRE(VideoCommon::ClassifyPrecacheCapturedGraphicsPipeline(captured, CompleteLedger())
                .eligible);

    auto missing = CompleteLedger();
    missing.Set(VideoCommon::PrecacheDependency::RenderPass,
                VideoCommon::PrecacheDependencyProvenance::Unknown);
    const auto rejected = VideoCommon::ClassifyPrecacheCapturedGraphicsPipeline(captured, missing);
    REQUIRE_FALSE(rejected.eligible);
    REQUIRE((rejected.missing_dependencies &
             VideoCommon::PrecacheDependencyBit(VideoCommon::PrecacheDependency::RenderPass)) != 0);

    const std::array duplicate{
        VideoCommon::PrecacheCapturedGraphicsStage{Shader::Stage::VertexB, 1},
        VideoCommon::PrecacheCapturedGraphicsStage{Shader::Stage::VertexB, 2},
    };
    REQUIRE(VideoCommon::ClassifyPrecacheCapturedGraphicsPipeline(duplicate, CompleteLedger())
                .primary_rejection == VideoCommon::PrecacheContractRejection::UnsupportedStageShape);
}

TEST_CASE("D1 source catalog codec preserves only identity-checked owned source", "[video_core]") {
    auto parsed = VideoCommon::ParsePrecacheBNSHProgramGroups(MakeBNSH(), 0);
    REQUIRE(parsed.groups.size() == 1);
    parsed.groups.front().source_container_identity = 0xC0FFEE;
    const VideoCommon::PrecacheSourceCatalog catalog{
        .title_id = 0x0100F2C0115B6000ULL,
        .compiler_target_fingerprint = 0x123456789ABCDEF0ULL,
        .groups = std::move(parsed.groups),
    };
    const auto encoded = VideoCommon::SerializePrecacheSourceCatalog(catalog);
    REQUIRE(encoded.has_value());
    const auto decoded = VideoCommon::DeserializePrecacheSourceCatalog(*encoded);
    REQUIRE(decoded.catalog.has_value());
    REQUIRE(decoded.catalog->title_id == catalog.title_id);
    REQUIRE(decoded.catalog->compiler_target_fingerprint == catalog.compiler_target_fingerprint);
    REQUIRE(decoded.catalog->groups.size() == 1);
    REQUIRE(decoded.catalog->groups.front().source_container_identity == 0xC0FFEE);
    REQUIRE(decoded.catalog->groups.front().stages[0].bytes == catalog.groups.front().stages[0].bytes);

    auto corrupted = *encoded;
    corrupted.back() ^= 1;
    REQUIRE(VideoCommon::DeserializePrecacheSourceCatalog(corrupted).rejection ==
            VideoCommon::PrecacheSourceCatalogRejection::Integrity);

    auto bad_identity = catalog;
    bad_identity.groups.front().stages.front().program_identity = 0;
    REQUIRE_FALSE(VideoCommon::SerializePrecacheSourceCatalog(bad_identity).has_value());
}
