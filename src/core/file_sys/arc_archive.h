// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>

#include "common/common_types.h"
#include "core/file_sys/vfs/vfs_types.h"

namespace FileSys {

// Minimal, read-only index parser for Super Smash Bros. Ultimate's data.arc
// format ("ARC" below) — NOT a general-purpose archive library. This exists
// for exactly one purpose: enumerate the real, individually-sized,
// individually zstd-compressed sub-files packed inside data.arc, so the
// precache shader scanner can queue each as its own independent unit of work
// instead of treating a single ~13GB file as one monolithic, single-threaded
// work item (which is what makes SSBU's scan hang — see
// docs/precache-scanner/FINDINGS.md for the investigation).
//
// UNLIKE this codebase's other format-detection code added for the same
// investigation (Yaz0, SARC recursion — see yaz0_compression.h and
// sarc_archive.h), this parser has NOT been verified against real file
// bytes. data.arc is far too large (10+ GB) to fetch into the environment
// that investigation was done in, so this implementation is built entirely
// from the format's primary, authoritative open-source reference
// implementation instead: jam1garner/smash-arc (MIT licensed, the library
// underlying most of the Smash Ultimate modding ecosystem — ARCropolis,
// ArcExplorer, etc.). Every struct layout and offset-resolution formula
// below is transcribed field-for-field from that project's src/filesystem.rs,
// src/arc_file.rs, and src/lookups.rs (fetched 2026; see those files for the
// original Rust/binrw source this was derived from). This should be treated
// as considerably lower-confidence than the rest of this investigation's
// fixes until it's actually run against a real dump.
//
// On-disk layout (all integers little-endian):
//   offset 0x00: magic u64 = 0xABCD_EF98_7654_3210
//   offset 0x08: stream_section_offset (u64)
//   offset 0x10: file_section_offset (u64)  — regular (non-stream) sub-file
//                 byte ranges are relative to this
//   offset 0x18: shared_section_offset (u64)
//   offset 0x20: u64 absolute file offset of the compressed filesystem
//                 table. At that offset: a 16-byte header (u32 magic
//                 0x10, u32 decomp_size, u32 comp_size, u32 section_size)
//                 followed by comp_size bytes of a *standalone* (no
//                 dictionary) zstd frame — decompresses to the FileSystem
//                 table walked below. (A second FilePtr64 + compressed
//                 blob for the *search* filesystem follows this one; not
//                 needed here and not parsed.)
//
// This only walks the regular (non-"stream") file table — the "stream"
// section is Smash Ultimate's separate audio/movie storage, addressed
// directly by absolute offset with no compression, and shader data has no
// reason to live there. See ParseFileSystemTable's implementation for the
// exact sequential field layout of the decompressed table itself.
struct ArcSubFile {
    u64 offset;        // Absolute byte offset within the .arc file.
    u32 comp_size;      // Bytes to read starting at `offset`.
    u32 decomp_size;    // Size after decompression, if is_compressed.
    bool is_compressed;
    bool uses_zstd;     // The reference implementation only ever supports
                         // zstd for compressed entries (see read_file_data in
                         // lookups.rs, which errors out on any other
                         // compression flag combination) — kept as a
                         // separate field anyway so a caller can tell
                         // "compressed but not zstd" (which would need
                         // different handling) apart from "not compressed".
};

// Returns the enumerated, de-duplicated (by absolute offset — many entries
// can share one underlying FileData, e.g. shared textures between
// characters) list of regular sub-files in `file`, or an empty vector if
// `file` doesn't start with the ARC magic or the table fails to parse
// (truncated read, a count that doesn't fit the remaining buffer, etc. —
// this never throws, and never trusts a count from the file without
// checking it against the actual decompressed buffer size first).
[[nodiscard]] std::vector<ArcSubFile> TryEnumerateArcSubFiles(const VirtualFile& file);

} // namespace FileSys
