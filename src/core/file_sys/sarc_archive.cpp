// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>

#include "core/file_sys/sarc_archive.h"

namespace FileSys {

namespace {

bool ReadU16(std::span<const u8> data, size_t offset, u16& out) {
    if (offset + sizeof(u16) > data.size()) return false;
    std::memcpy(&out, data.data() + offset, sizeof(u16));
    return true;
}

bool ReadU32(std::span<const u8> data, size_t offset, u32& out) {
    if (offset + sizeof(u32) > data.size()) return false;
    std::memcpy(&out, data.data() + offset, sizeof(u32));
    return true;
}

} // namespace

SarcArchive SarcArchive::Parse(std::vector<u8> owned_bytes) {
    SarcArchive archive;
    archive.owned_bytes_ = std::move(owned_bytes);
    const std::span<const u8> data{archive.owned_bytes_};

    // -- Main SARC header (0x14 / 20 bytes total) --
    if (data.size() < 0x20) return archive; // Not even enough for header + SFAT tag.
    if (std::memcmp(data.data(), "SARC", 4) != 0) return archive;
    // Bytes 4 through 7: header_size (u16, normally 0x14) + byte-order-mark (u16) — not
    // needed for reading; every real SARC uses little-endian + BOM 0xFEFF.
    u32 data_offset{};
    if (!ReadU32(data, 0x0C, data_offset)) return archive;
    // Bytes 0x10 through 0x13: version (u16) + reserved (u16).

    // -- SFAT node table header --
    if (std::memcmp(data.data() + 0x14, "SFAT", 4) != 0) return archive;
    // Bytes 0x18-0x19: SFAT header size (u16) — expected 0x0C, not checked.
    u16 node_count{};
    if (!ReadU16(data, 0x1A, node_count)) return archive;
    // Bytes 0x1C-0x1F: hash key multiplier (u32) — not needed for reading.

    struct Node {
        u32 attributes{};
        u32 data_start{};
        u32 data_end{};
    };
    std::vector<Node> nodes(node_count);

    size_t cursor = 0x20;
    for (u16 i = 0; i < node_count; ++i) {
        u32 hash{}, attributes{}, data_start{}, data_end{};
        if (!ReadU32(data, cursor, hash) || !ReadU32(data, cursor + 4, attributes) ||
            !ReadU32(data, cursor + 8, data_start) || !ReadU32(data, cursor + 12, data_end)) {
            return archive;
        }
        nodes[i] = Node{attributes, data_start, data_end};
        cursor += 16;
    }

    // -- SFNT string table header --
    if (cursor + 8 > data.size() || std::memcmp(data.data() + cursor, "SFNT", 4) != 0) {
        return archive;
    }
    cursor += 8; // magic(4) + header_size(u16) + reserved(u16)

    if (data_offset < cursor || data_offset > data.size()) return archive;
    const size_t string_table_begin = cursor;
    const size_t string_table_size = data_offset - cursor;

    archive.entries_.reserve(node_count);
    for (u16 i = 0; i < node_count; ++i) {
        const u32 string_offset = (nodes[i].attributes & 0xFFFFu) * 4u;
        if (string_offset >= string_table_size) continue;

        std::string name;
        const size_t name_start = string_table_begin + string_offset;
        for (size_t j = name_start; j < data.size() && data[j] != 0; ++j) {
            name += static_cast<char>(data[j]);
        }

        const size_t abs_start = static_cast<size_t>(data_offset) + nodes[i].data_start;
        const size_t abs_end = static_cast<size_t>(data_offset) + nodes[i].data_end;
        if (abs_start > abs_end || abs_end > data.size()) continue;

        archive.entries_.push_back(Entry{
            std::move(name),
            std::span<const u8>{data.data() + abs_start, abs_end - abs_start},
        });
    }

    archive.ok_ = true;
    return archive;
}

const SarcArchive::Entry* SarcArchive::Find(std::string_view name) const {
    for (const auto& entry : entries_) {
        if (entry.name == name) return &entry;
    }
    return nullptr;
}

} // namespace FileSys
