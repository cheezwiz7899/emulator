// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>

#include "common/common_types.h"
#include "core/file_sys/vfs/vfs_types.h"

namespace FileSys {

// Parser for Monolith Soft's "arh1" index format (Xenoblade Chronicles 2 and
// likely sibling titles — the same studio's engine spans the series). The
// real data lives in a separate, paired file (conventionally same base name,
// ".ard" extension instead of ".arh") that this parser never reads — only
// the small index file. Entries whose real ard-file bytes turn out to start
// with an "xbc1" header (Monolith's own zlib/zstd-wrapped compression
// container, confirmed via a real, GPL-3.0-licensed open-source
// implementation — PredatorCZ/XenoLib, consulted for understanding only;
// nothing from it is copied here, both because of its license and because
// this is a small enough struct to write cleanly from the spec) still need
// that second layer unwrapped by the caller; see docs/precache-scanner/FINDINGS.md
// for the byte layout this was derived from and verified against.
//
// Provenance note: unlike this investigation's other container parsers
// (SARC, Yaz0, the SSBU ARC index, HWDE's pairtable), the exact table
// offset/entry-count field positions here were independently re-derived and
// verified against real game data (a real .arh file's header, and the
// table's own internal size self-consistency: file_count * 24 bytes almost
// exactly matches the file's remaining length past the table offset) before
// being used, not taken on faith from an external claim.
struct ArhSubFile {
    u64 ard_offset;   // Absolute byte offset within the paired .ard file.
    u32 comp_size;    // Bytes to read starting at ard_offset.
    u32 decomp_size;  // Expected size after XBC1 decompression, if flags indicate compression.
    u32 flags;        // 2 observed to mean "xbc1-wrapped compressed"; other values passed through unmodified.
};

// Returns every entry in an .arh file's index, or an empty vector if `file`
// doesn't start with the "arh1" magic or the table fails to parse (a
// truncated read, a declared file_count that doesn't fit the remaining
// buffer, etc.). `ard_file_size`, if provided (nonzero), is used to reject
// entries whose declared (ard_offset, comp_size) would read past the end of
// the real data file — pass 0 to skip that check (e.g. if the paired .ard
// file's size isn't known yet).
[[nodiscard]] std::vector<ArhSubFile> TryEnumerateArhSubFiles(const VirtualFile& file,
                                                                u64 ard_file_size = 0);

} // namespace FileSys
