// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>

#include "core/file_sys/vfs/vfs.h"
#include "core/file_sys/xc2_arh_archive.h"

namespace FileSys {

namespace {
constexpr u32 kArhMagic = 0x31687261u; // "arh1" read as LE u32
constexpr size_t kEntrySize = 24;      // u64 ard_offset, u32 comp_size, u32 decomp_size, u32 flags, u32 file_id
} // namespace

std::vector<ArhSubFile> TryEnumerateArhSubFiles(const VirtualFile& file, u64 ard_file_size) {
    if (!file || file->GetSize() < 0x24) {
        return {};
    }

    // Header fields needed: magic (0x00), table_offset (0x1c), file_count (0x20).
    // Everything between is present in real files but not needed here.
    const auto header = file->ReadBytes(0x24, 0);
    if (header.size() != 0x24) {
        return {};
    }
    u32 magic{};
    std::memcpy(&magic, header.data(), 4);
    if (magic != kArhMagic) {
        return {}; // Not an .arh file — the expected case for every other title.
    }
    u32 table_offset{}, file_count{};
    std::memcpy(&table_offset, header.data() + 0x1c, 4);
    std::memcpy(&file_count, header.data() + 0x20, 4);

    // Sanity bound: the real XC2 sample examined during this investigation has
    // 36,941 entries; a few million is still a plausible ceiling for a larger
    // title without accepting an obviously-corrupt/hostile count.
    constexpr u32 kMaxPlausibleEntries = 4'000'000;
    if (file_count == 0 || file_count > kMaxPlausibleEntries) {
        return {};
    }
    const u64 table_bytes_needed = static_cast<u64>(file_count) * kEntrySize;
    if (static_cast<u64>(table_offset) + table_bytes_needed > static_cast<u64>(file->GetSize())) {
        return {};
    }

    const auto table = file->ReadBytes(table_bytes_needed, table_offset);
    if (table.size() != table_bytes_needed) {
        return {};
    }

    std::vector<ArhSubFile> result;
    result.reserve(file_count);
    for (u32 i = 0; i < file_count; ++i) {
        const size_t off = static_cast<size_t>(i) * kEntrySize;
        u64 ard_offset{};
        u32 comp_size{}, decomp_size{}, flags{};
        std::memcpy(&ard_offset, table.data() + off, 8);
        std::memcpy(&comp_size, table.data() + off + 8, 4);
        std::memcpy(&decomp_size, table.data() + off + 12, 4);
        std::memcpy(&flags, table.data() + off + 16, 4);
        // table.data()+off+20 is file_id — not needed here, nothing after it read.

        if (ard_file_size != 0 && ard_offset + comp_size > ard_file_size) {
            continue; // Corrupt entry relative to the real paired file — skip, don't abort the whole table.
        }
        result.push_back(ArhSubFile{ard_offset, comp_size, decomp_size, flags});
    }
    return result;
}

} // namespace FileSys
