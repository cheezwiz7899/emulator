// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "video_core/precache_compiler_target.h"

namespace {

class TemporaryTargetFile {
public:
    TemporaryTargetFile() {
        static std::atomic<u64> next_id{};
        const auto base = std::filesystem::temp_directory_path();
        std::error_code ec;
        for (u64 attempt = 0; attempt < 128; ++attempt) {
            directory = base / ("citron_precache_target_contract_" + std::to_string(++next_id));
            if (std::filesystem::create_directory(directory, ec)) {
                path = directory / "target.bin";
                return;
            }
            ec.clear();
        }
        throw std::runtime_error("could not create private temporary target directory");
    }

    ~TemporaryTargetFile() {
        std::error_code ec;
        std::filesystem::remove_all(directory, ec);
    }

    std::filesystem::path path;
    std::filesystem::path directory;
};

VideoCommon::PrecacheCompilerTarget MakeTarget() {
    VideoCommon::PrecacheCompilerTarget target{};
    target.profile.supported_spirv = 0x00010600;
    target.profile.unified_descriptor_binding = true;
    target.profile.support_int16 = true;
    target.profile.force_fragment_relaxed_precision = true;
    target.profile.min_ssbo_alignment = 16;
    target.host_info.support_float16 = true;
    target.host_info.min_ssbo_alignment = 16;
    target.shader_recompiler_revision = 7;
    target.descriptor_abi_version = 13;
    target.specialization_schema_version = 29;
    target.gpu_accuracy_mode = 2;
    return target;
}

TEST_CASE("Pre-cache compiler target rejects incomplete contracts", "[video_core]") {
    TemporaryTargetFile file;
    const auto expected = MakeTarget();
    const auto expected_fingerprint = VideoCommon::CompilerTargetFingerprint(expected);

    REQUIRE(VideoCommon::SavePrecacheCompilerTarget(file.path, expected));
    const auto loaded = VideoCommon::LoadPrecacheCompilerTarget(file.path);
    REQUIRE(loaded.has_value());
    REQUIRE(VideoCommon::CompilerTargetFingerprint(*loaded) == expected_fingerprint);
    REQUIRE(loaded->profile.supported_spirv == expected.profile.supported_spirv);
    REQUIRE(loaded->profile.unified_descriptor_binding);
    REQUIRE(loaded->profile.force_fragment_relaxed_precision);
    REQUIRE(loaded->host_info.support_float16);
    REQUIRE(loaded->shader_recompiler_revision == expected.shader_recompiler_revision);
    REQUIRE(loaded->descriptor_abi_version == expected.descriptor_abi_version);
    REQUIRE(loaded->specialization_schema_version == expected.specialization_schema_version);
    REQUIRE(loaded->gpu_accuracy_mode == expected.gpu_accuracy_mode);

    // A per-game override may select any supported accuracy level. Every
    // encoded mode must round-trip and occupy its own final-module namespace.
    std::array<u64, 4> accuracy_fingerprints{};
    for (size_t mode_index = 0; mode_index < accuracy_fingerprints.size(); ++mode_index) {
        const auto mode = static_cast<u8>(mode_index);
        auto variant = expected;
        variant.gpu_accuracy_mode = mode;
        REQUIRE(VideoCommon::SavePrecacheCompilerTarget(file.path, variant));
        const auto variant_loaded = VideoCommon::LoadPrecacheCompilerTarget(file.path);
        REQUIRE(variant_loaded.has_value());
        REQUIRE(variant_loaded->gpu_accuracy_mode == mode);
        accuracy_fingerprints[mode_index] = VideoCommon::CompilerTargetFingerprint(*variant_loaded);
    }
    for (size_t lhs = 0; lhs < accuracy_fingerprints.size(); ++lhs) {
        for (size_t rhs = lhs + 1; rhs < accuracy_fingerprints.size(); ++rhs) {
            REQUIRE(accuracy_fingerprints[lhs] != accuracy_fingerprints[rhs]);
        }
    }

    // Scanner must fingerprint the decoded complete contract, not reconstruct
    // one from profile/host_info and accidentally reset these revisions.
    const VideoCommon::PrecacheCompilerTarget profile_only{expected.profile, expected.host_info};
    REQUIRE(VideoCommon::CompilerTargetFingerprint(profile_only) != expected_fingerprint);

    auto changed = expected;
    changed.profile.has_split_descriptor_sets = true;
    REQUIRE(VideoCommon::CompilerTargetFingerprint(changed) != expected_fingerprint);
    changed = expected;
    changed.profile.has_broken_spirv_subgroup_mask_vector_extract_dynamic = true;
    REQUIRE(VideoCommon::CompilerTargetFingerprint(changed) != expected_fingerprint);
    changed = expected;
    changed.profile.support_native_ndc = true;
    REQUIRE(VideoCommon::CompilerTargetFingerprint(changed) != expected_fingerprint);
    changed = expected;
    changed.profile.force_fragment_relaxed_precision = false;
    REQUIRE(VideoCommon::CompilerTargetFingerprint(changed) != expected_fingerprint);
    changed = expected;
    changed.host_info.min_ssbo_alignment = 32;
    REQUIRE(VideoCommon::CompilerTargetFingerprint(changed) != expected_fingerprint);
    changed = expected;
    ++changed.shader_recompiler_revision;
    REQUIRE(VideoCommon::CompilerTargetFingerprint(changed) != expected_fingerprint);
    changed = expected;
    ++changed.descriptor_abi_version;
    REQUIRE(VideoCommon::CompilerTargetFingerprint(changed) != expected_fingerprint);
    changed = expected;
    ++changed.specialization_schema_version;
    REQUIRE(VideoCommon::CompilerTargetFingerprint(changed) != expected_fingerprint);
    changed = expected;
    ++changed.gpu_accuracy_mode;
    REQUIRE(VideoCommon::CompilerTargetFingerprint(changed) != expected_fingerprint);

    // The final byte is the GPU-accuracy enum. A malformed future enum must
    // invalidate the complete target, never silently fall back to a setting
    // that could publish scanner modules into the wrong namespace.
    {
        std::fstream corrupt_accuracy{file.path, std::ios::binary | std::ios::in | std::ios::out};
        REQUIRE(corrupt_accuracy.is_open());
        corrupt_accuracy.seekp(-1, std::ios::end);
        const char invalid_accuracy{4};
        corrupt_accuracy.write(&invalid_accuracy, sizeof(invalid_accuracy));
    }
    REQUIRE_FALSE(VideoCommon::LoadPrecacheCompilerTarget(file.path).has_value());

    REQUIRE(VideoCommon::SavePrecacheCompilerTarget(file.path, expected));
    {
        std::ofstream append{file.path, std::ios::binary | std::ios::app};
        const char corrupt{};
        append.write(&corrupt, sizeof(corrupt));
    }
    REQUIRE_FALSE(VideoCommon::LoadPrecacheCompilerTarget(file.path).has_value());

    REQUIRE(VideoCommon::SavePrecacheCompilerTarget(file.path, expected));
    std::filesystem::resize_file(file.path, std::filesystem::file_size(file.path) - 1);
    REQUIRE_FALSE(VideoCommon::LoadPrecacheCompilerTarget(file.path).has_value());

    // Interrupted replacement can leave only the last-known-good backup. It must remain usable.
    REQUIRE(VideoCommon::SavePrecacheCompilerTarget(file.path, expected));
    const auto backup = file.path.parent_path() / (file.path.filename().string() + ".bak");
    std::filesystem::copy_file(file.path, backup,
                               std::filesystem::copy_options::overwrite_existing);
    std::filesystem::remove(file.path);
    const auto recovered = VideoCommon::LoadPrecacheCompilerTarget(file.path);
    REQUIRE(recovered.has_value());
    REQUIRE(VideoCommon::CompilerTargetFingerprint(*recovered) == expected_fingerprint);
}

} // Anonymous namespace
