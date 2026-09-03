// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>

#include "common/common_types.h"
#include "core/file_sys/vfs/vfs_types.h"

namespace FileSys {

// Parser for Metroid Prime Remastered's MaterialArchive.arc — Retro
// Studios' own "RFRM" container format (used across their engine's
// titles), specifically the "MTRL" (material archive) variant.
//
// Format, verified field-for-field against the real file during this
// investigation (see docs/precache-scanner/FINDINGS.md section 3):
//   FormDescriptor (32 bytes): FourCC type("RFRM"), u64 size, u64 unk1,
//     FourCC id("MTRL"), u32 reader_version, u32 writer_version. Only
//     reader_version 22 or 10 is the MaterialArchive layout this parser
//     understands; other values (168, 12, observed in the wild per the
//     struct definitions this was derived from) are a different, raw
//     layout not handled here.
//   MaterialArchive(size): repeated MaterialBackend structs until `size`
//     bytes (from FormDescriptor) are consumed.
//   MaterialBackend: a ChunkDescriptor (FourCC id, u64 size, u32 unk,
//     u64 skip — skip that many more bytes before the real content
//     starts), whose `id` names the backend's shader ISA: "SDX " for
//     standard Microsoft DXBC (a different instruction set entirely —
//     see FINDINGS.md, not extracted here), "SNVN" for Nintendo Switch's
//     own native NVN format — the real target. Then: u32 flags; if
//     nonzero, a chain of variable-length tables (buffer entries,
//     material entries with nested per-material shader-technique
//     entries) this parser walks through but doesn't otherwise use, only
//     to keep its read cursor correctly positioned; then u32
//     shaderSourceCount and that many SShaderSource entries — u8 state,
//     an optional length-prefixed string, then a u32-length-prefixed raw
//     data blob. For "SNVN" backends specifically, that data blob itself
//     starts with a small header (u32 unused, u32 decompressed_size,
//     u8 flag, u32 compressed_size — 13 bytes) followed by a zlib stream;
//     decompressing it exposes a bare, unwrapped Maxwell shader
//     recognized by this scanner's general raw-NVN-shader convention
//     (see main.cpp) with no further MPR-specific detection code needed.
//
// PROVENANCE: struct definitions transcribed field-for-field from
// PrimeDecomp/PrimeRemasterStructs (MIT licensed, fetched directly via
// GitHub tarball), then independently verified against the real
// MaterialArchive.arc file before being trusted further (FormDescriptor's
// own declared size plus its own header exactly matches the real file's
// total size; reader_version correctly selects this layout; internal
// count self-consistency — bufferEntryCount equals shaderSourceCount in
// both real backends found).
struct MprShaderSource {
    u64 data_offset;  // Absolute byte offset of this SShaderSource's raw
                       // data blob (the u32-length-prefixed one, i.e. right
                       // after its length field) within the file.
    u32 data_size;     // Length of that raw data blob.
};

// Returns every "SNVN"-backend shader source's raw data blob location, or
// an empty vector if `file` doesn't parse as a recognized MaterialArchive
// (wrong magic, unsupported reader_version, a truncated/malformed table,
// etc.). Deliberately does not return "SDX "-backend (DXBC) entries — see
// the type-level comment above and FINDINGS.md for why that's a
// different instruction set out of scope for this scanner. Each returned
// blob is a raw, still-need-zlib-decompression region (see the 13-byte
// header described above); this function only locates them, the same
// "enumerate, then let the caller read+decompress via ranged reads"
// pattern as this investigation's other large-archive parsers.
[[nodiscard]] std::vector<MprShaderSource> TryEnumerateMprSnvnShaderSources(const VirtualFile& file);

} // namespace FileSys
