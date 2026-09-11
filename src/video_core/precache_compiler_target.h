// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <optional>

#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/profile.h"

namespace VideoCommon {

// Compiler settings captured from a real Vulkan renderer. ROM scanning runs outside
// the renderer, so it must never invent this contract for final SPIR-V modules.
struct PrecacheCompilerTarget {
    Shader::Profile profile;
    Shader::HostTranslateInfo host_info;
    // Bump these when a shader-recompiler change, descriptor layout change, or
    // specialization ABI change can alter final SPIR-V without changing a
    // Profile/HostTranslateInfo field. They are part of the final-module
    // namespace, not descriptive metadata.
    u32 shader_recompiler_revision{1};
    u32 descriptor_abi_version{1};
    u32 specialization_schema_version{1};
    // Effective Settings::GpuAccuracy value captured from the renderer boot.
    // Keep this explicit even though Low currently also changes Profile: an
    // accuracy setting is a compiler namespace decision in its own right, and
    // per-game overrides must never share final modules with global settings.
    u8 gpu_accuracy_mode{};
};

[[nodiscard]] bool SavePrecacheCompilerTarget(const std::filesystem::path& filename,
                                              const PrecacheCompilerTarget& target);
[[nodiscard]] std::optional<PrecacheCompilerTarget> LoadPrecacheCompilerTarget(
    const std::filesystem::path& filename);
[[nodiscard]] u64 CompilerTargetFingerprint(const PrecacheCompilerTarget& target) noexcept;

} // namespace VideoCommon
