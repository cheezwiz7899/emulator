// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/precache_program_contract.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <ranges>

#include "shader_recompiler/program_header.h"
#include "video_core/shader_program_identity.h"

namespace VideoCommon {
namespace {

constexpr u32 BNSH_MAGIC = 0x48534E42u;
constexpr u32 GRSC_MAGIC = 0x63737267u;
constexpr size_t BINARY_HEADER_SIZE = 32;
constexpr size_t GRSC_HEADER_SIZE = 56;
constexpr size_t VARIATION_HEADER_SIZE = 64;
constexpr size_t PROGRAM_HEADER_SIZE = 176;
constexpr size_t SHADER_CODE_HEADER_SIZE = 64;
constexpr size_t BYTE_CODE_PREAMBLE_SIZE = 48;
constexpr std::array<char, 8> SOURCE_CATALOG_MAGIC{'c', 'i', 't', 'd', '1', 's', 'r', 'c'};
constexpr u32 SOURCE_CATALOG_VERSION = 1;
constexpr u32 SOURCE_CATALOG_ENDIAN = 0x01020304;
constexpr size_t SOURCE_CATALOG_HEADER_SIZE = 56;
constexpr size_t MAX_SOURCE_CATALOG_STAGES = 6;

constexpr std::array<std::pair<size_t, Shader::Stage>, 6> STAGE_FIELDS{{
    {0x08, Shader::Stage::VertexB},
    {0x10, Shader::Stage::TessellationControl},
    {0x18, Shader::Stage::TessellationEval},
    {0x20, Shader::Stage::Geometry},
    {0x28, Shader::Stage::Fragment},
    {0x30, Shader::Stage::Compute},
}};

[[nodiscard]] bool Add(size_t left, size_t right, size_t& result) noexcept {
    if (right > std::numeric_limits<size_t>::max() - left) {
        return false;
    }
    result = left + right;
    return true;
}

[[nodiscard]] bool Multiply(size_t left, size_t right, size_t& result) noexcept {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

[[nodiscard]] bool Range(size_t offset, size_t length, size_t total) noexcept {
    size_t end{};
    return Add(offset, length, end) && end <= total;
}

template <typename T>
[[nodiscard]] bool Read(std::span<const u8> bytes, size_t offset, T& value) noexcept {
    if (!Range(offset, sizeof(T), bytes.size())) {
        return false;
    }
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return true;
}

[[nodiscard]] std::optional<u64> ProgramIdentity(std::span<const u8> bytes) {
    if (bytes.size() < sizeof(Shader::ProgramHeader) ||
        bytes.size() % sizeof(u64) != 0) {
        return std::nullopt;
    }
    std::vector<u64> words(bytes.size() / sizeof(u64));
    std::memcpy(words.data(), bytes.data(), bytes.size());
    return ComputeMaxwellProgramIdentity(words);
}

[[nodiscard]] bool IsDeclaredStageHeader(std::span<const u8> bytes, Shader::Stage stage) noexcept {
    if (bytes.size() < sizeof(Shader::ProgramHeader)) {
        return false;
    }
    Shader::ProgramHeader sph{};
    std::memcpy(&sph, bytes.data(), sizeof(sph));
    const u32 type = sph.common0.shader_type.Value();
    if (sph.common0.version.Value() == 0 || sph.common0.sass_version.Value() == 0 ||
        type < 1 || type > 5 || sph.common0.sph_type.Value() != (type == 5 ? 2u : 1u)) {
        return false;
    }
    switch (stage) {
    case Shader::Stage::VertexB:
        return type == 1;
    case Shader::Stage::TessellationControl:
        return type == 2;
    case Shader::Stage::TessellationEval:
        return type == 3;
    case Shader::Stage::Geometry:
        return type == 4;
    case Shader::Stage::Fragment:
        return type == 5;
    default:
        return false;
    }
}

[[nodiscard]] bool IsExact(PrecacheDependencyProvenance value) noexcept {
    return value == PrecacheDependencyProvenance::ProgramDerived ||
           value == PrecacheDependencyProvenance::SourceExact ||
           value == PrecacheDependencyProvenance::CapturedExact ||
           value == PrecacheDependencyProvenance::RuntimeABI;
}

[[nodiscard]] u64 Checksum(std::span<const u8> bytes) noexcept {
    // Integrity only. Semantic acceptance below recomputes every program
    // identity and validates all domains, so this checksum is never a source
    // of trust for a stage or recipe.
    u64 value{1469598103934665603ULL};
    for (const u8 byte : bytes) {
        value ^= byte;
        value *= 1099511628211ULL;
    }
    return value;
}

void WriteU8(std::vector<u8>& output, u8 value) {
    output.push_back(value);
}

void WriteU32(std::vector<u8>& output, u32 value) {
    for (size_t index = 0; index < sizeof(value); ++index) {
        output.push_back(static_cast<u8>(value >> (index * 8)));
    }
}

void WriteU64(std::vector<u8>& output, u64 value) {
    for (size_t index = 0; index < sizeof(value); ++index) {
        output.push_back(static_cast<u8>(value >> (index * 8)));
    }
}

[[nodiscard]] bool ReadU8(std::span<const u8> bytes, size_t& offset, u8& value) noexcept {
    if (!Range(offset, 1, bytes.size())) {
        return false;
    }
    value = bytes[offset++];
    return true;
}

[[nodiscard]] bool ReadU32(std::span<const u8> bytes, size_t& offset, u32& value) noexcept {
    if (!Range(offset, sizeof(value), bytes.size())) {
        return false;
    }
    value = 0;
    for (size_t index = 0; index < sizeof(value); ++index) {
        value |= static_cast<u32>(bytes[offset++]) << (index * 8);
    }
    return true;
}

[[nodiscard]] bool ReadU64(std::span<const u8> bytes, size_t& offset, u64& value) noexcept {
    if (!Range(offset, sizeof(value), bytes.size())) {
        return false;
    }
    value = 0;
    for (size_t index = 0; index < sizeof(value); ++index) {
        value |= static_cast<u64>(bytes[offset++]) << (index * 8);
    }
    return true;
}

[[nodiscard]] bool ValidSourceStage(Shader::Stage stage) noexcept {
    return stage == Shader::Stage::VertexB || stage == Shader::Stage::TessellationControl ||
           stage == Shader::Stage::TessellationEval || stage == Shader::Stage::Geometry ||
           stage == Shader::Stage::Fragment;
}

[[nodiscard]] bool ValidateCatalogGroup(const PrecacheSourceProgramGroup& group,
                                        const PrecacheBNSHParseLimits& limits,
                                        size_t& total_bytes) noexcept {
    if ((group.origin != PrecacheSourceOrigin::BNSHDeclaredProgram &&
         group.origin != PrecacheSourceOrigin::StandaloneProgram) ||
        group.stages.empty() || group.stages.size() > MAX_SOURCE_CATALOG_STAGES) {
        return false;
    }
    std::array<bool, MAX_SOURCE_CATALOG_STAGES> seen{};
    for (const auto& stage : group.stages) {
        const size_t index = static_cast<size_t>(stage.stage);
        if (!ValidSourceStage(stage.stage) || index >= seen.size() || seen[index] ||
            stage.bytes.size() > limits.max_stage_bytes || !stage.program_identity ||
            ProgramIdentity(stage.bytes) != stage.program_identity ||
            !IsDeclaredStageHeader(stage.bytes, stage.stage) ||
            total_bytes > limits.max_total_bytes - stage.bytes.size()) {
            return false;
        }
        seen[index] = true;
        total_bytes += stage.bytes.size();
    }
    return true;
}

} // Anonymous namespace

PrecacheBNSHParseResult ParsePrecacheBNSHProgramGroups(
    std::span<const u8> container_bytes, size_t container_offset,
    const PrecacheBNSHParseLimits& limits) {
    PrecacheBNSHParseResult result;
    if (container_offset > container_bytes.size() ||
        !Range(container_offset, BINARY_HEADER_SIZE, container_bytes.size())) {
        result.rejection = PrecacheBNSHParseRejection::Malformed;
        return result;
    }

    u32 magic{};
    if (!Read(container_bytes, container_offset, magic) || magic != BNSH_MAGIC) {
        result.rejection = PrecacheBNSHParseRejection::UnsupportedLayout;
        return result;
    }
    u16 block_offset{};
    if (!Read(container_bytes, container_offset + 0x16, block_offset)) {
        result.rejection = PrecacheBNSHParseRejection::Malformed;
        return result;
    }
    size_t grsc{};
    if (!Add(container_offset, block_offset, grsc) ||
        !Range(grsc, GRSC_HEADER_SIZE, container_bytes.size())) {
        result.rejection = PrecacheBNSHParseRejection::Malformed;
        return result;
    }
    if (!Read(container_bytes, grsc, magic) || magic != GRSC_MAGIC) {
        result.rejection = PrecacheBNSHParseRejection::UnsupportedLayout;
        return result;
    }

    u32 variation_count{};
    u64 variation_offset{};
    if (!Read(container_bytes, grsc + 0x1C, variation_count) ||
        !Read(container_bytes, grsc + 0x20, variation_offset) || variation_count == 0) {
        result.rejection = PrecacheBNSHParseRejection::Malformed;
        return result;
    }
    result.declared_variations = variation_count;
    if (variation_count > limits.max_variations) {
        result.rejection = PrecacheBNSHParseRejection::Capacity;
        return result;
    }
    if (variation_offset > std::numeric_limits<size_t>::max()) {
        result.rejection = PrecacheBNSHParseRejection::Malformed;
        return result;
    }
    size_t variation_table{};
    size_t variation_bytes{};
    if (!Add(container_offset, static_cast<size_t>(variation_offset), variation_table) ||
        !Multiply(variation_count, VARIATION_HEADER_SIZE, variation_bytes) ||
        !Range(variation_table, variation_bytes, container_bytes.size())) {
        result.rejection = PrecacheBNSHParseRejection::Malformed;
        return result;
    }

    for (u32 variation = 0; variation < variation_count; ++variation) {
        // Deferral is safe only at a complete group boundary. Never copy a
        // partial group and then report it as retained source provenance.
        if (result.groups.size() == limits.max_groups) {
            result.rejection = PrecacheBNSHParseRejection::Capacity;
            return result;
        }
        size_t relative_header{};
        size_t header{};
        if (!Multiply(variation, VARIATION_HEADER_SIZE, relative_header) ||
            !Add(variation_table, relative_header, header)) {
            result.rejection = PrecacheBNSHParseRejection::Malformed;
            return result;
        }
        u64 program_offset{};
        if (!Read(container_bytes, header + 0x10, program_offset) || program_offset == 0 ||
            program_offset > std::numeric_limits<size_t>::max()) {
            result.rejection = PrecacheBNSHParseRejection::Malformed;
            return result;
        }
        size_t program_header{};
        if (!Add(container_offset, static_cast<size_t>(program_offset), program_header) ||
            !Range(program_header, PROGRAM_HEADER_SIZE, container_bytes.size())) {
            result.rejection = PrecacheBNSHParseRejection::Malformed;
            return result;
        }

        PrecacheSourceProgramGroup group{
            .origin = PrecacheSourceOrigin::BNSHDeclaredProgram,
            .container_offset = container_offset,
            .variation_index = variation,
            .program_offset = program_offset,
        };
        for (const auto [field, stage] : STAGE_FIELDS) {
            u64 stage_offset{};
            if (!Read(container_bytes, program_header + field, stage_offset) || stage_offset == 0) {
                continue;
            }
            if (stage_offset > std::numeric_limits<size_t>::max()) {
                result.rejection = PrecacheBNSHParseRejection::Malformed;
                return result;
            }
            size_t code_header{};
            if (!Add(container_offset, static_cast<size_t>(stage_offset), code_header) ||
                !Range(code_header, SHADER_CODE_HEADER_SIZE, container_bytes.size())) {
                result.rejection = PrecacheBNSHParseRejection::Malformed;
                return result;
            }
            u64 control_offset{};
            u64 byte_offset{};
            u32 byte_size{};
            u32 control_size{};
            if (!Read(container_bytes, code_header + 0x08, control_offset) ||
                !Read(container_bytes, code_header + 0x10, byte_offset) ||
                !Read(container_bytes, code_header + 0x18, byte_size) ||
                !Read(container_bytes, code_header + 0x1C, control_size)) {
                result.rejection = PrecacheBNSHParseRejection::Malformed;
                return result;
            }
            // The D contract only accepts the declared byte-code section. The scanner's
            // ControlCode fallback remains legacy speculative extraction behavior.
            static_cast<void>(control_offset);
            static_cast<void>(control_size);
            if (byte_offset == 0 || byte_size <= BYTE_CODE_PREAMBLE_SIZE ||
                byte_offset > std::numeric_limits<size_t>::max()) {
                result.rejection = PrecacheBNSHParseRejection::UnsupportedLayout;
                return result;
            }
            const size_t source_size = byte_size - BYTE_CODE_PREAMBLE_SIZE;
            if (result.copied_bytes > limits.max_total_bytes ||
                source_size > limits.max_stage_bytes ||
                source_size > limits.max_total_bytes - result.copied_bytes) {
                result.rejection = PrecacheBNSHParseRejection::Capacity;
                return result;
            }
            size_t raw_source{};
            size_t source{};
            if (!Add(container_offset, static_cast<size_t>(byte_offset), raw_source) ||
                !Add(raw_source, BYTE_CODE_PREAMBLE_SIZE, source) ||
                !Range(source, source_size, container_bytes.size())) {
                result.rejection = PrecacheBNSHParseRejection::Malformed;
                return result;
            }
            if (std::ranges::any_of(group.stages, [stage, source](const auto& existing) {
                    return existing.stage == stage || existing.source_offset == source;
                })) {
                result.rejection = PrecacheBNSHParseRejection::AmbiguousStage;
                return result;
            }
            std::vector<u8> source_bytes(source_size);
            std::memcpy(source_bytes.data(), container_bytes.data() + source, source_size);
            if (!IsDeclaredStageHeader(source_bytes, stage)) {
                result.rejection = PrecacheBNSHParseRejection::UnsupportedLayout;
                return result;
            }
            group.stages.push_back({
                .stage = stage,
                .source_offset = source,
                .bytes = std::move(source_bytes),
            });
            // Compute identity from the just-owned bytes, so it cannot observe a mutable
            // archive buffer after the scanner releases its decompression storage.
            group.stages.back().program_identity = ProgramIdentity(group.stages.back().bytes);
            result.copied_bytes += source_size;
        }
        if (group.stages.empty()) {
            result.rejection = PrecacheBNSHParseRejection::UnsupportedLayout;
            return result;
        }
        result.groups.push_back(std::move(group));
    }
    return result;
}

PrecacheStandaloneParseResult ParsePrecacheStandaloneProgram(
    std::span<const u8> program_bytes, Shader::Stage stage, size_t source_offset,
    const PrecacheBNSHParseLimits& limits) {
    PrecacheStandaloneParseResult result;
    if (program_bytes.size() < sizeof(Shader::ProgramHeader)) {
        result.rejection = PrecacheBNSHParseRejection::Malformed;
        return result;
    }
    const size_t bounded_size = std::min(program_bytes.size(), limits.max_stage_bytes);
    const std::span<const u8> bounded{program_bytes.first(bounded_size)};
    if (!IsDeclaredStageHeader(bounded, stage)) {
        result.rejection = PrecacheBNSHParseRejection::UnsupportedLayout;
        return result;
    }
    if (bounded.size() % sizeof(u64) != 0) {
        result.rejection = PrecacheBNSHParseRejection::Malformed;
        return result;
    }
    std::vector<u64> words(bounded.size() / sizeof(u64));
    std::memcpy(words.data(), bounded.data(), bounded.size());
    const auto identity_size = FindMaxwellProgramSize(words);
    if (!identity_size) {
        result.rejection = program_bytes.size() > bounded_size
                               ? PrecacheBNSHParseRejection::Capacity
                               : PrecacheBNSHParseRejection::UnsupportedLayout;
        return result;
    }
    size_t source_size{};
    if (!Add(*identity_size, sizeof(u64), source_size) || source_size > bounded.size()) {
        result.rejection = PrecacheBNSHParseRejection::Malformed;
        return result;
    }
    std::vector<u8> owned(source_size);
    std::memcpy(owned.data(), bounded.data(), source_size);
    const auto identity = ProgramIdentity(owned);
    if (!identity) {
        result.rejection = PrecacheBNSHParseRejection::UnsupportedLayout;
        return result;
    }
    result.copied_bytes = owned.size();
    result.group = PrecacheSourceProgramGroup{
        .origin = PrecacheSourceOrigin::StandaloneProgram,
        .container_offset = source_offset,
        .variation_index = 0,
        .program_offset = 0,
        .stages = {{
            .stage = stage,
            .source_offset = source_offset,
            .bytes = std::move(owned),
            .program_identity = identity,
        }},
    };
    return result;
}

std::optional<std::vector<u8>> SerializePrecacheSourceCatalog(
    const PrecacheSourceCatalog& catalog, const PrecacheBNSHParseLimits& limits) {
    if (catalog.title_id == 0 || catalog.compiler_target_fingerprint == 0 ||
        catalog.groups.empty() || catalog.groups.size() > limits.max_groups) {
        return std::nullopt;
    }
    size_t total_bytes{};
    for (const auto& group : catalog.groups) {
        if (!ValidateCatalogGroup(group, limits, total_bytes)) {
            return std::nullopt;
        }
    }
    std::vector<u8> payload;
    payload.reserve(total_bytes + catalog.groups.size() * 96);
    for (const auto& group : catalog.groups) {
        WriteU8(payload, static_cast<u8>(group.origin));
        WriteU8(payload, static_cast<u8>(group.stages.size()));
        WriteU8(payload, 0);
        WriteU8(payload, 0);
        WriteU64(payload, group.source_container_identity);
        WriteU64(payload, group.container_offset);
        WriteU64(payload, group.variation_index);
        WriteU64(payload, group.program_offset);
        for (const auto& stage : group.stages) {
            WriteU8(payload, static_cast<u8>(stage.stage));
            WriteU8(payload, 1);
            WriteU8(payload, 0);
            WriteU8(payload, 0);
            WriteU64(payload, stage.source_offset);
            WriteU64(payload, *stage.program_identity);
            WriteU32(payload, static_cast<u32>(stage.bytes.size()));
            payload.insert(payload.end(), stage.bytes.begin(), stage.bytes.end());
        }
    }
    std::vector<u8> output;
    output.reserve(SOURCE_CATALOG_HEADER_SIZE + payload.size());
    output.insert(output.end(), SOURCE_CATALOG_MAGIC.begin(), SOURCE_CATALOG_MAGIC.end());
    WriteU32(output, SOURCE_CATALOG_VERSION);
    WriteU32(output, SOURCE_CATALOG_ENDIAN);
    WriteU64(output, catalog.title_id);
    WriteU64(output, catalog.compiler_target_fingerprint);
    WriteU32(output, static_cast<u32>(catalog.groups.size()));
    WriteU32(output, 0);
    WriteU64(output, payload.size());
    WriteU64(output, Checksum(payload));
    output.insert(output.end(), payload.begin(), payload.end());
    return output;
}

PrecacheSourceCatalogLoadResult DeserializePrecacheSourceCatalog(
    std::span<const u8> bytes, const PrecacheBNSHParseLimits& limits) {
    PrecacheSourceCatalogLoadResult result;
    if (bytes.size() < SOURCE_CATALOG_HEADER_SIZE ||
        !std::equal(SOURCE_CATALOG_MAGIC.begin(), SOURCE_CATALOG_MAGIC.end(), bytes.begin())) {
        result.rejection = PrecacheSourceCatalogRejection::Malformed;
        return result;
    }
    size_t offset = SOURCE_CATALOG_MAGIC.size();
    u32 version{};
    u32 endian{};
    u64 payload_size{};
    u64 payload_checksum{};
    u32 group_count{};
    u32 reserved{};
    PrecacheSourceCatalog catalog;
    if (!ReadU32(bytes, offset, version) || !ReadU32(bytes, offset, endian) ||
        !ReadU64(bytes, offset, catalog.title_id) ||
        !ReadU64(bytes, offset, catalog.compiler_target_fingerprint) ||
        !ReadU32(bytes, offset, group_count) || !ReadU32(bytes, offset, reserved) ||
        !ReadU64(bytes, offset, payload_size) || !ReadU64(bytes, offset, payload_checksum)) {
        result.rejection = PrecacheSourceCatalogRejection::Malformed;
        return result;
    }
    if (version != SOURCE_CATALOG_VERSION || endian != SOURCE_CATALOG_ENDIAN) {
        result.rejection = PrecacheSourceCatalogRejection::UnsupportedVersion;
        return result;
    }
    constexpr u64 MAX_CATALOG_METADATA_PER_GROUP = 256;
    const u64 max_payload = static_cast<u64>(limits.max_total_bytes) +
                            static_cast<u64>(limits.max_groups) * MAX_CATALOG_METADATA_PER_GROUP;
    if (reserved != 0 || catalog.title_id == 0 || catalog.compiler_target_fingerprint == 0 ||
        group_count == 0 || group_count > limits.max_groups || payload_size > max_payload ||
        payload_size != bytes.size() - offset) {
        result.rejection = PrecacheSourceCatalogRejection::Capacity;
        return result;
    }
    const auto payload = bytes.subspan(offset);
    if (Checksum(payload) != payload_checksum) {
        result.rejection = PrecacheSourceCatalogRejection::Integrity;
        return result;
    }
    catalog.groups.reserve(group_count);
    size_t payload_offset{};
    size_t total_bytes{};
    for (u32 group_index = 0; group_index < group_count; ++group_index) {
        u8 origin{};
        u8 stage_count{};
        u8 reserved_a{};
        u8 reserved_b{};
        PrecacheSourceProgramGroup group;
        if (!ReadU8(payload, payload_offset, origin) || !ReadU8(payload, payload_offset, stage_count) ||
            !ReadU8(payload, payload_offset, reserved_a) ||
            !ReadU8(payload, payload_offset, reserved_b) ||
            !ReadU64(payload, payload_offset, group.source_container_identity) ||
            !ReadU64(payload, payload_offset, group.container_offset) ||
            !ReadU64(payload, payload_offset, group.variation_index) ||
            !ReadU64(payload, payload_offset, group.program_offset) || reserved_a != 0 ||
            reserved_b != 0 || stage_count == 0 || stage_count > MAX_SOURCE_CATALOG_STAGES ||
            origin > static_cast<u8>(PrecacheSourceOrigin::StandaloneProgram)) {
            result.rejection = PrecacheSourceCatalogRejection::Malformed;
            return result;
        }
        group.origin = static_cast<PrecacheSourceOrigin>(origin);
        group.stages.reserve(stage_count);
        size_t group_bytes{};
        for (u8 stage_index = 0; stage_index < stage_count; ++stage_index) {
            u8 stage{};
            u8 has_identity{};
            u8 stage_reserved_a{};
            u8 stage_reserved_b{};
            u64 source_offset{};
            u64 identity{};
            u32 byte_count{};
            if (!ReadU8(payload, payload_offset, stage) ||
                !ReadU8(payload, payload_offset, has_identity) ||
                !ReadU8(payload, payload_offset, stage_reserved_a) ||
                !ReadU8(payload, payload_offset, stage_reserved_b) ||
                !ReadU64(payload, payload_offset, source_offset) ||
                !ReadU64(payload, payload_offset, identity) ||
                !ReadU32(payload, payload_offset, byte_count) || has_identity != 1 ||
                stage_reserved_a != 0 || stage_reserved_b != 0 ||
                byte_count > limits.max_stage_bytes ||
                group_bytes > limits.max_total_bytes - byte_count ||
                !Range(payload_offset, byte_count, payload.size())) {
                result.rejection = PrecacheSourceCatalogRejection::Malformed;
                return result;
            }
            PrecacheSourceStage decoded_stage{
                .stage = static_cast<Shader::Stage>(stage),
                .source_offset = source_offset,
                .bytes = std::vector<u8>(payload.begin() + payload_offset,
                                         payload.begin() + payload_offset + byte_count),
                .program_identity = identity,
            };
            payload_offset += byte_count;
            group_bytes += byte_count;
            group.stages.push_back(std::move(decoded_stage));
        }
        if (!ValidateCatalogGroup(group, limits, total_bytes)) {
            result.rejection = PrecacheSourceCatalogRejection::Malformed;
            return result;
        }
        catalog.groups.push_back(std::move(group));
    }
    if (payload_offset != payload.size()) {
        result.rejection = PrecacheSourceCatalogRejection::Malformed;
        return result;
    }
    result.catalog = std::move(catalog);
    return result;
}

bool SavePrecacheSourceCatalog(const std::filesystem::path& filename,
                               const PrecacheSourceCatalog& catalog,
                               const PrecacheBNSHParseLimits& limits) {
    const auto bytes = SerializePrecacheSourceCatalog(catalog, limits);
    if (!bytes) {
        return false;
    }
    auto temporary = filename;
    temporary += ".tmp";
    auto backup = filename;
    backup += ".previous";
    std::error_code error;
    std::filesystem::remove(temporary, error);
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(bytes->data()),
                   static_cast<std::streamsize>(bytes->size()));
        file.flush();
        if (!file.good()) {
            std::filesystem::remove(temporary, error);
            return false;
        }
    }
    const bool had_existing = std::filesystem::exists(filename, error) && !error;
    if (had_existing) {
        if (std::filesystem::exists(backup, error) && !error) {
            std::filesystem::remove(temporary, error);
            return false;
        }
        std::filesystem::rename(filename, backup, error);
        if (error) {
            std::filesystem::remove(temporary, error);
            return false;
        }
    }
    std::filesystem::rename(temporary, filename, error);
    if (error) {
        if (had_existing) {
            std::error_code restore_error;
            std::filesystem::rename(backup, filename, restore_error);
        }
        std::filesystem::remove(temporary, error);
        return false;
    }
    if (had_existing) {
        std::filesystem::remove(backup, error);
    }
    return true;
}

PrecacheSourceCatalogLoadResult LoadPrecacheSourceCatalog(
    const std::filesystem::path& filename, const PrecacheBNSHParseLimits& limits) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return {.rejection = PrecacheSourceCatalogRejection::Malformed};
    }
    const auto size = file.tellg();
    constexpr u64 MAX_CATALOG_METADATA_PER_GROUP = 256;
    const u64 max_file_size = SOURCE_CATALOG_HEADER_SIZE + static_cast<u64>(limits.max_total_bytes) +
                              static_cast<u64>(limits.max_groups) * MAX_CATALOG_METADATA_PER_GROUP;
    if (size <= 0 || static_cast<u64>(size) > max_file_size) {
        return {.rejection = PrecacheSourceCatalogRejection::Capacity};
    }
    std::vector<u8> bytes(static_cast<size_t>(size));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file.good()) {
        return {.rejection = PrecacheSourceCatalogRejection::Malformed};
    }
    return DeserializePrecacheSourceCatalog(bytes, limits);
}

PrecacheDependencyProvenance PrecacheDependencyLedger::Get(
    PrecacheDependency dependency) const noexcept {
    return provenance[static_cast<size_t>(dependency)];
}

void PrecacheDependencyLedger::Set(PrecacheDependency dependency,
                                   PrecacheDependencyProvenance value) noexcept {
    provenance[static_cast<size_t>(dependency)] = value;
}

PrecacheContractDecision ClassifyPrecacheSourceProgramGroup(
    const PrecacheSourceProgramGroup& group, const PrecacheDependencyLedger& ledger) noexcept {
    if (group.stages.size() != 2 || group.stages[0].stage != Shader::Stage::VertexB ||
        group.stages[1].stage != Shader::Stage::Fragment) {
        return {.primary_rejection = PrecacheContractRejection::UnsupportedStageShape};
    }
    if (!group.stages[0].HasExactProgramIdentity() || !group.stages[1].HasExactProgramIdentity()) {
        return {.primary_rejection = PrecacheContractRejection::MissingProgramIdentity};
    }
    u64 missing{};
    for (u8 index = 0; index < static_cast<u8>(PrecacheDependency::Count); ++index) {
        const auto dependency = static_cast<PrecacheDependency>(index);
        if (!IsExact(ledger.Get(dependency))) {
            missing |= PrecacheDependencyBit(dependency);
        }
    }
    if (missing != 0) {
        return {.primary_rejection = PrecacheContractRejection::MissingDependency,
                .missing_dependencies = missing};
    }
    return {.eligible = true};
}

PrecacheContractDecision ClassifyPrecacheCapturedGraphicsPipeline(
    std::span<const PrecacheCapturedGraphicsStage> stages,
    const PrecacheDependencyLedger& ledger) noexcept {
    if (stages.empty()) {
        return {.primary_rejection = PrecacheContractRejection::UnsupportedStageShape};
    }
    std::array<bool, static_cast<size_t>(Shader::Stage::VertexA) + 1> seen{};
    for (const auto& stage : stages) {
        const size_t index = static_cast<size_t>(stage.stage);
        if (stage.program_identity == 0 || index >= seen.size() ||
            stage.stage == Shader::Stage::Compute || seen[index]) {
            return {.primary_rejection = stage.program_identity == 0
                            ? PrecacheContractRejection::MissingProgramIdentity
                            : PrecacheContractRejection::UnsupportedStageShape};
        }
        seen[index] = true;
    }
    u64 missing{};
    for (u8 index = 0; index < static_cast<u8>(PrecacheDependency::Count); ++index) {
        const auto dependency = static_cast<PrecacheDependency>(index);
        if (!IsExact(ledger.Get(dependency))) {
            missing |= PrecacheDependencyBit(dependency);
        }
    }
    if (missing != 0) {
        return {.primary_rejection = PrecacheContractRejection::MissingDependency,
                .missing_dependencies = missing};
    }
    return {.eligible = true};
}

} // namespace VideoCommon
