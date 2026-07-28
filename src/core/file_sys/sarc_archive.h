// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"

namespace FileSys {

// Minimal read-only parser for Nintendo's SARC archive format, as used across
// Wii U/Switch titles (BOTW/TotK/Splatoon/etc.) for bundling named resource
// blobs — most relevantly here, romfs/Pack/ZsDic.pack.zs, which decompresses
// (with no zstd dictionary — it's the bootstrap for every other dictionary)
// into a SARC containing one *.zsdic entry per resource type (zs.zsdic,
// pack.zsdic, bcett.byml.zsdic, and — the one this exists to find — a
// shader-archive dictionary, expected to follow the same "<type>.zsdic"
// naming convention).
//
// Only reading is implemented; citron has no need to write SARC archives.
//
// Format (all integers big-endian... actually little-endian on Switch/Wii U
// titles built with the standard toolchain; see SarcArchive::Parse for the
// exact byte layout, based on the widely-documented SARC header):
//   offset 0x00: magic "SARC" (4 bytes)
//   offset 0x04: header size (u16, normally 0x14), byte-order-mark (u16)
//   offset 0x08: file size (u32)
//   offset 0x0C: data offset (u32) — where entry payload bytes begin
//   offset 0x10: version (u16), reserved (u16)
//   -- SFAT node table follows immediately --
//   offset 0x14: SFAT magic "SFAT" (4 bytes), header size (u16), node count (u16)
//   offset 0x1C: hash multiplier (u32)
//   then `node count` entries of 16 bytes each:
//     u32 name hash, u32 attributes (low 16 bits = string-table offset / 4,
//     high 16 bits = flags), u32 data start (relative to data offset),
//     u32 data end (relative to data offset)
//   -- SFNT string table follows --
//     magic "SFNT" (4 bytes), header size (u16), reserved (u16), then
//     NUL-terminated entry names back to back, aligned as a block.
class SarcArchive {
public:
    struct Entry {
        std::string name;
        std::span<const u8> data; // Points into the archive's own owned buffer.
    };

    SarcArchive() = default;
    // Entry::data spans point into this object's own owned_bytes_. Moving is
    // safe — std::vector's move constructor transfers the buffer pointer
    // without reallocating, so the buffer's address (and every span into it)
    // stays valid. Copying is NOT safe — it would reallocate owned_bytes_ at
    // a new address while the copied entries_ still point at the original's
    // buffer — so copying is deleted rather than left as a latent bug.
    SarcArchive(const SarcArchive&) = delete;
    SarcArchive& operator=(const SarcArchive&) = delete;
    SarcArchive(SarcArchive&&) = default;
    SarcArchive& operator=(SarcArchive&&) = default;

    // Returns std::nullopt (empty vector, ok()==false) if bytes doesn't start
    // with a valid SARC header. Does not throw.
    static SarcArchive Parse(std::vector<u8> owned_bytes);

    [[nodiscard]] bool Ok() const { return ok_; }
    [[nodiscard]] const std::vector<Entry>& Entries() const { return entries_; }

    // Returns nullptr if no entry with this exact name exists.
    [[nodiscard]] const Entry* Find(std::string_view name) const;

private:
    bool ok_{false};
    std::vector<u8> owned_bytes_; // Entry::data spans point into this.
    std::vector<Entry> entries_;
};

} // namespace FileSys
