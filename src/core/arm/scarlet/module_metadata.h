// SPDX-FileCopyrightText: 2026 citron-neo Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>
#include <vector>

namespace Core::Scarlet {

// Extracts the same function-entry seeds ScarletNX's NSO/NRO loader gathers from MOD0:
// dynamic function symbols, relative relocations, and .eh_frame_hdr function entries.
std::vector<std::uint64_t> DiscoverModuleEntryPoints(
    const std::vector<std::uint8_t>& image, std::uint64_t image_address, std::uint64_t text_offset,
    std::uint64_t text_size, std::size_t& unwind_entries, std::size_t& function_symbols,
    std::size_t& relocation_entries);

} // namespace Core::Scarlet
