// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <vector>

#include "shader_recompiler/environment.h"

namespace VideoCommon {

// On-disk format (phase4_prototype_slots.bin, one per game -- lives next to that game's own
// spirv_cache.bin, see PipelineCache::LoadDiskResources, vk_pipeline_cache.cpp):
//   8 bytes  magic "citrp4s\0"
//   u32      version (see PHASE4_PROTOTYPE_SLOTS_VERSION in the .cpp)
//   u32      count
//   count * (u32 cbuf_index, u32 cbuf_offset)
//
// Deliberately its own small file, not folded into spirv_cache.bin or gated by
// SPIRV_CACHE_VERSION: this data feeds INTO texture_key computation (via
// Shader::IsPhase4PrototypeSlot) rather than being a translation result cached BY texture_key,
// and its own on-disk format can change independently of the SPIR-V cache's -- see
// Shader::ActivePhase4PrototypeSlots's doc comment (environment.h) for why growing this file's
// contents between sessions needs no SPIRV_CACHE_VERSION bump at all, format changes to THIS
// file are a different concern from that.
//
// Stores every coordinate THIS specific game has taught the mechanism, full stop -- no
// separate hardcoded-default list exists anywhere to merge back in (see
// Shader::ActivePhase4PrototypeSlots's doc comment, environment.h, for why that was removed),
// so a corrupt, missing, or version-mismatched file simply costs a game everything it's
// learned so far, same as a profile that's never hit this path before.

// Returns this game's previously-learned coordinates, or an empty vector on any error
// (missing file, bad magic, version mismatch, truncated/corrupt data) -- never throws, and a
// failure here is not itself a bug to fix reactively: it just means this session starts with
// no active slots at all, exactly like a first-ever run would, which is always a safe,
// already-proven state to fall back to (ordinary pre-Phase-4 behavior for every coordinate).
[[nodiscard]] std::vector<Shader::Phase4PrototypeSlot>
LoadPhase4PrototypeSlots(const std::filesystem::path& filename);

// Overwrites filename with exactly `slot_list` (the caller's responsibility to have already
// merged whatever was loaded with whatever got learned this session -- see
// VideoCommon::TakePhase4PrototypeCandidates, shader_environment.cpp, and
// PipelineCache::~PipelineCache, vk_pipeline_cache.cpp, for the real call site). Silently does
// nothing on a write failure (matches SpirvCache::Save's own fail-quiet convention,
// spirv_cache.cpp) rather than crashing a session over a diagnostic/optimization file that was
// never load-bearing for correctness.
void SavePhase4PrototypeSlots(const std::filesystem::path& filename,
                               const std::vector<Shader::Phase4PrototypeSlot>& slot_list);

} // namespace VideoCommon
