// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>
#include <vector>

#include "common/common_types.h"
#include "core/file_sys/vfs/vfs_types.h"

namespace FileSys {

// Parser for CRI Middleware's CPK archive format (One Piece: Pirate
// Warriors 3's `rom*.cpk` files, and broadly across CRIWARE-licensed
// titles). CPK wraps a generic "@UTF" table format (also used by CRI's
// USM/ACB formats, not handled here) for all of its own metadata,
// including its file listing.
//
// Format, all multi-byte integers big-endian *inside* UTF table payloads,
// little-endian in the outer packet headers wrapping them:
//   Each named section ("CPK ", "TOC ", "ITOC", "ETOC", ...) is a packet:
//     offset 0x00: char magic[4]
//     offset 0x04: u32 enc_flag (little-endian) — 0xFF means the payload
//                  that follows is NOT masked; any other value means it
//                  is, and must be XORed with a repeating 64-byte mask
//                  before parsing as a UTF table (see kUtfMask below).
//     offset 0x08: u32 utf_size (little-endian) — payload length.
//     offset 0x0C: u32 reserved.
//     offset 0x10: utf_size bytes of (possibly masked) UTF table data.
//   A UTF table itself starts with "@UTF" (after any masking is undone):
//     magic(4), table_size(u32), version(u16), rows_offset(u16),
//     strings_offset(u32), data_offset(u32), name_offset(u32),
//     column_count(u16), row_width(u16), row_count(u32) — all big-endian,
//     offsets relative to the byte right after table_size (i.e. byte 8 of
//     the table). Followed by column_count column descriptors (a flags
//     byte, then a name string offset if the Name flag is set, then a
//     default value inline if the Default flag is set), then row_count
//     rows of row_width bytes each, laid out per the column schema.
//   The "CPK " table's rows give the offsets/sizes of the "TOC "/"ITOC"/
//   "ETOC" sections; the "TOC " table's rows are file entries directly:
//   named columns "DirName", "FileName", "FileSize", "ExtractSize",
//   "FileOffset". ExtractSize > FileSize (or the entry's own first 8 bytes
//   being the literal "CRILAYLA" magic) means the entry needs CRILAYLA
//   decompression — see crilayla_compression.h.
//
// PROVENANCE AND CONFIDENCE: this container format (CPK packet headers,
// the UTF table format, the 64-byte XOR mask scheme) is cross-verified
// against two independent public sources — a public technical gist
// documenting the Valkyria Chronicles 3 CPK-derived format family, and
// `Youjose/PyCriCodecs`'s C++ implementation (fetched directly for
// understanding only; the repository has no LICENSE file, so nothing is
// copied from it, only used to inform a clean-room implementation of the
// same publicly-documented field layout) — both agree on the same field
// layout independently. This container format is trusted at the same
// level as this investigation's other container parsers (SARC, arh1,
// the pairtable format). The CRILAYLA compression algorithm itself is a
// meaningfully different, lower-confidence case — see
// crilayla_compression.h and docs/precache-scanner/FINDINGS.md section 8a
// for why.
struct CpkFileEntry {
    std::string dir_name;  // May be empty.
    std::string file_name;
    u64 file_offset;   // Absolute byte offset within the CPK file.
    u64 file_size;     // Bytes to read starting at file_offset.
    u64 extract_size;  // Expected size after decompression, if compressed
                        // (equals file_size when the entry isn't compressed).
};

// Returns every file entry listed in a CPK archive's "TOC " table, or an
// empty vector if `file` doesn't start with the "CPK " magic or the
// header/TOC table fails to parse. Does not read file *contents* — only
// the (small, always at the start of the archive) index tables needed to
// locate them; real entry data is read separately via ranged reads once
// an entry's (file_offset, file_size) is known, the same pattern as this
// investigation's other large-archive parsers.
[[nodiscard]] std::vector<CpkFileEntry> TryEnumerateCpkFiles(const VirtualFile& file);

} // namespace FileSys
